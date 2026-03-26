"""
MiniMind-V 在 LIBERO 仿真环境上的评估脚本。
支持对 SFT 后的 VLA checkpoint 进行 rollout 评测。
"""

import json
import os
import sys
import time
from pathlib import Path

# 必须在导入 mujoco/libero 之前设置，以避免 EGL 初始化错误
os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio
import numpy as np
import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

# rich_helpers 和本脚本同目录，动态加路径确保 import 可靠
# 项目根目录 (含 model/) 也加入路径以便 from model.model_vla import VLAModel
_script_dir = Path(__file__).parent
_root_dir = _script_dir.parent
sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(_root_dir))
from rich_helpers import ok, warn, err, fatal, section, kv_table, console

import tyro
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

# 预设置 LIBERO 资产路径缓存，避免每次自动尝试从 HuggingFace 下载
import libero.libero
_libero_pkg_dir = Path(libero.libero.__file__).parent
_libero_assets_dir = _libero_pkg_dir / "assets"
if _libero_assets_dir.exists():
    libero.libero._assets_path_cache = str(_libero_assets_dir)

# VLA 推理封装
from model.model_vla import VLAModel


# ==============================================================================
# 辅助函数
# ==============================================================================

def _flip_frame(frame: np.ndarray) -> np.ndarray:
    """翻转帧以修正 MuJoCo offscreen 渲染的坐标系（原点左下→左上）"""
    return np.flipud(frame)


def generate_markdown_report(summary: dict, output_dir: str, cfg) -> str:
    """生成 markdown 格式的评测报告，返回报告文件路径"""
    md_lines = []
    md_lines.append("# LIBERO 评测报告\n")

    # 评测配置
    md_lines.append("## 评测配置\n")
    md_lines.append("| 配置项 | 值 |")
    md_lines.append("|--------|-----|")
    md_lines.append(f"| Task Suite | {summary['task_suite']} |")
    md_lines.append(f"| Checkpoint | `{summary['checkpoint']}` |")
    md_lines.append(f"| 评测任务数 | {summary['num_tasks_evaluated']} |")
    md_lines.append(f"| 最大步数 | {cfg.max_steps} |")
    md_lines.append(f"| 种子 | {cfg.seed} |")
    md_lines.append(f"| Device | {cfg.device} |")
    md_lines.append(f"| Max New Tokens | {cfg.max_new_tokens} |")
    md_lines.append("")

    # 总体结果
    md_lines.append("## 总体结果\n")
    md_lines.append("| 指标 | 值 |")
    md_lines.append("|------|-----|")
    md_lines.append(f"| 成功数 | {summary['success_count']} / {summary['num_tasks_evaluated']} |")
    md_lines.append(f"| **成功率** | **{summary['success_rate']}** |")
    md_lines.append(f"| 平均奖励 | {summary['mean_episode_return']} |")
    md_lines.append(f"| 平均步数 | {summary['mean_episode_len']} |")
    md_lines.append("")

    # 各任务详细结果
    md_lines.append("## 各任务详细结果\n")
    md_lines.append("| # | 任务指令 | 状态 | 奖励 | 步数 | 耗时 |")
    md_lines.append("|---|---------|------|------|------|------|")
    for r in summary["per_task_results"]:
        status = "✅" if r["success"] else "❌"
        instruction = r["instruction"]
        if len(instruction) > 40:
            instruction = instruction[:40] + "..."
        md_lines.append(
            f"| {r['task_idx']:02d} | {instruction} | {status} | "
            f"{r['episode_return']:.3f} | {r['episode_len']} | {r['elapsed_s']:.1f}s |"
        )
    md_lines.append("")

    # 写入文件
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return report_path


# ==============================================================================
# 评测流程
# ==============================================================================

def evaluate_task(vla: VLAModel, env: OffScreenRenderEnv, instruction: str, max_steps: int, save_dir: str, fps: int = 30, verbose: bool = True) -> dict:
    """在单个任务上执行一个 episode，可选保存帧图像和视频"""
    obs = env.reset()
    episode_return = 0.0
    episode_len = 0
    done = False
    success = False
    frames = []

    # 打印任务指令
    if verbose:
        section(f"任务指令: {instruction}", "yellow")
        console.print()

    for step in range(max_steps):
        rgb = obs["agentview_image"]
        wrist_rgb = obs["robot0_eye_in_hand_image"]

        # VLA 推理（双图像模式：主视角 + 腕部视角）
        front_img = Image.fromarray(rgb)
        wrist_img = Image.fromarray(wrist_rgb)
        action_tensor = vla.predict(instruction, [front_img, wrist_img])
        action = action_tensor.cpu().numpy()
        # 防御性：确保 action 是 1D (7,) 而非 (1, 7) 或其他形状
        if action.ndim > 1:
            action = action.squeeze()

        # 调试输出：显示 VLA 推理结果
        if verbose:
            console.print(f"[dim]Step {step:3d}[/dim] | Action: {action.round(3)}")

        obs, reward, done, info_ = env.step(action)
        episode_return += reward
        episode_len += 1

        # 采集帧（所有帧用于合成视频，按需保存 PNG）
        if save_dir:
            frames.append(_flip_frame(rgb))
            if step % 50 == 0:
                p = Path(save_dir) / f"step_{step:04d}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(_flip_frame(rgb)).save(p)

        if done or info_.get("success", False):
            success = info_.get("success", False)
            break

    # 保存视频（帧已在采集时翻转，直接写入）
    if save_dir and frames:
        video_path = Path(save_dir) / "rollout.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(video_path, fps=fps, codec="libx264", pixelformat="yuv420p") as writer:
            for frame in frames:
                writer.append_data(frame)
        warn(f"视频已保存: {video_path}")

    return {"success": success, "episode_return": episode_return, "episode_len": episode_len}


def run_eval(checkpoint: str, model_id: str, task_suite: str, num_episodes: int,
             max_steps: int, max_new_tokens: int, output_dir: str, device: str,
             seed: int, verbose: bool = True) -> dict:

    # 种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(output_dir, exist_ok=True)

    # 加载 VLA 模型
    section("加载 VLA 模型", "cyan")
    vla = VLAModel(checkpoint, device=device, dual_image=True, max_new_tokens=max_new_tokens)
    ok(f"模型已加载到 {device}")
    kv_table("模型信息", [
        ("checkpoint", checkpoint),
        ("device", device),
        ("processor", vla.processor.__class__.__name__),
        ("max_new_tokens", str(max_new_tokens)),
    ])
    console.print()

    # 环境
    section("创建 LIBERO 环境", "cyan")
    benchmark = get_benchmark(task_suite)()
    tasks = benchmark.tasks
    ok(f"Suite: {task_suite}, 共 {len(tasks)} 个任务，评测前 {num_episodes} 个")
    console.print()

    # 逐任务评测 + 实时表格
    results = []
    task_table = Table(title="评测进度", box=None)
    task_table.add_column("#", style="cyan", justify="right", width=3)
    task_table.add_column("任务指令 (截断)", style="white")
    task_table.add_column("结果", style="green", width=10)
    task_table.add_column("奖励", style="magenta", width=8)
    task_table.add_column("步数", style="yellow", width=5)
    task_table.add_column("耗时", style="dim", width=6)

    section("开始评测", "cyan")
    for task_idx, task in enumerate(tasks[:num_episodes]):
        instruction = task.language
        bddl_path = benchmark.get_task_bddl_file_path(task_idx)
        task_env = OffScreenRenderEnv(bddl_file_name=bddl_path)
        save_dir = os.path.join(output_dir, f"task_{task_idx:02d}") if True else ""

        t0 = time.time()
        result = evaluate_task(
            vla=vla, env=task_env,
            instruction=instruction,
            max_steps=max_steps, save_dir=save_dir,
            verbose=verbose,
        )
        elapsed = time.time() - t0

        result["task_idx"] = task_idx
        result["instruction"] = instruction
        result["elapsed_s"] = elapsed
        results.append(result)

        status = "[green]✓ SUCCESS[/green]" if result["success"] else "[red]✗ FAIL[/red]"
        task_table.add_row(
            f"{task_idx:02d}",
            instruction[:50] + "...",
            status,
            f"{result['episode_return']:.3f}",
            str(result["episode_len"]),
            f"{elapsed:.1f}s",
        )
        console.print(task_table, end="\r")

    console.print()  # 换行

    # 汇总
    success_count = sum(r["success"] for r in results)
    success_rate = success_count / len(results) * 100
    mean_return = np.mean([r["episode_return"] for r in results])
    mean_len = np.mean([r["episode_len"] for r in results])

    summary = {
        "task_suite": task_suite,
        "checkpoint": checkpoint,
        "num_tasks_evaluated": len(results),
        "success_count": success_count,
        "success_rate": f"{success_rate:.1f}%",
        "mean_episode_return": f"{mean_return:.3f}",
        "mean_episode_len": f"{mean_len:.1f}",
        "per_task_results": results,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 生成 markdown 报告
    _cfg = type("EvalCfg", (), {
        "max_steps": max_steps,
        "seed": seed,
        "device": device,
        "max_new_tokens": max_new_tokens,
    })()
    report_path = generate_markdown_report(summary, output_dir, _cfg)
    warn(f"Markdown 报告已生成: {report_path}")

    # 汇总输出
    section("评测结果汇总", "green")
    result_table = Table(box=None)
    result_table.add_column("指标", style="yellow bold")
    result_table.add_column("值", style="cyan")
    result_table.add_row("任务数量", str(len(results)))
    result_table.add_row("成功数", f"{success_count}/{len(results)}")
    result_table.add_row("成功率", f"[green]{success_rate:.1f}%[/green]")
    result_table.add_row("平均奖励", f"{mean_return:.3f}")
    result_table.add_row("平均步数", f"{mean_len:.1f}")
    result_table.add_row("结果文件", summary_path)
    console.print(result_table)

    return summary


# ==============================================================================
# tyro CLI
# ==============================================================================

if __name__ == "__main__":
    import tyro
    from dataclasses import dataclass

    @dataclass
    class EvalConfig:
        checkpoint: str = "./checkpoints/final"
        model_id: str = "jingyaogong/minimind2-v"
        task_suite: str = "LIBERO_OBJECT"
        num_episodes: int = 10
        max_steps: int = 400
        max_new_tokens: int = 64
        output_dir: str = "./eval_results"
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        seed: int = 42
        verbose: bool = True  # 是否显示 VLA 推理输出

    tyro.extras.set_accent_color("cyan")
    cfg = tyro.cli(EvalConfig)
    run_eval(
        checkpoint=cfg.checkpoint,
        model_id=cfg.model_id,
        task_suite=cfg.task_suite,
        num_episodes=cfg.num_episodes,
        max_steps=cfg.max_steps,
        max_new_tokens=cfg.max_new_tokens,
        output_dir=cfg.output_dir,
        device=cfg.device,
        seed=cfg.seed,
        verbose=cfg.verbose,
    )
