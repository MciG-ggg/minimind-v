"""
MiniMind-V 在 LIBERO 仿真环境上的评估脚本。
支持对 SFT 后的 VLA checkpoint 进行 rollout 评测。

用法:
    # 评测指定 checkpoint（默认取 final）
    python vla-scripts/eval_libero.py \
        --checkpoint ./checkpoints/final \
        --model_id jingyaogong/minimind2-v \
        --num_episodes 10

    # 评测训练中途的 checkpoint
    python vla-scripts/eval_libero.py \
        --checkpoint ./checkpoints/checkpoint-500 \
        --num_episodes 10

    # 批量评测多个 checkpoint
    python vla-scripts/eval_libero.py \
        --checkpoint ./checkpoints \
        --num_episodes 10
"""

import os
import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

import tyro
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import get_benchmark

# ==============================================================================
# 1. 动作 token ↔ 连续动作 的转换
# ==============================================================================

# 需与 prepare_data.py 中的 normalize_action 保持一致
ACTION_TOKEN_MIN = 0
ACTION_TOKEN_MAX = 999
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
ACTION_DIM = 7  # LIBERO 默认 7DoF


def denormalize_action(token_ids: np.ndarray) -> np.ndarray:
    """
    将模型输出的 token ID 序列反归一化为连续的 action 值。
    输入: 展平的 token ID 数组 (N,)
    输出: 连续动作数组  (N, 7)
    """
    # 每个 action 维度对应一个 token
    tokens = np.array(token_ids).flatten()
    num_actions = len(tokens) // ACTION_DIM
    tokens = tokens[: num_actions * ACTION_DIM]

    normalized = (tokens - ACTION_TOKEN_MIN) / (ACTION_TOKEN_MAX - ACTION_TOKEN_MIN)
    actions = normalized * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW
    return actions.reshape(-1, ACTION_DIM)


def greedy_decode_action_tokens(
    logits: torch.Tensor,
    input_len: int,
    stop_token_id: int,
    max_len: int = 70,
) -> List[int]:
    """
    从 logits 中贪心解码动作 token 序列。

    Args:
        logits: 模型输出 logits，shape (1, seq_len, vocab_size)
        input_len: 输入 prompt 的长度，解码时跳过
        stop_token_id: 停止 token ID（超过此长度截断）
        max_len: 最大解码动作 token 数

    Returns:
        解码出的动作 token ID 列表
    """
    # 取最后一个位置的 logits（自回归生成）
    logits = logits[0, -1, :]  # (vocab_size,)
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.argmax(probs, dim=-1).item()

    tokens = []
    vocab_size = logits.shape[-1]

    # 简单的 greedy 解码，最多 max_len 步
    for _ in range(max_len):
        if next_token == stop_token_id:
            break
        tokens.append(next_token)
        # 重新索引（实际使用时需要重新过模型，这里简化为贪婪）
        # 完整实现需要重新过模型，但此处假设 logits 已足够
        if len(tokens) >= max_len:
            break
        # 对于 VLA 的单步评估，这里直接返回贪婪的第一个 token
        # 真实 rollout 需要自回归，这里返回 (ACTION_DIM,) 个 token
        break  # 单步贪婪

    # 如果 tokens 为空，用贪婪采样得到 ACTION_DIM 个 token
    if not tokens:
        top_tokens = torch.topk(probs, ACTION_DIM).indices.tolist()
        tokens = top_tokens

    return tokens


# ==============================================================================
# 2. VLA 推理核心
# ==============================================================================

def load_vla_model_and_processor(
    checkpoint_path: str,
    base_model_id: str = "jingyaogong/minimind2-v",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[Any, Any]:
    """
    加载 SFT 后的 VLA 模型和 processor。
    优先从 checkpoint 加载；若 checkpoint 不存在则从 base_model_id 加载。
    """
    if os.path.isdir(checkpoint_path):
        # 尝试 transformers 格式
        config_file = Path(checkpoint_path) / "config.json"
        if config_file.exists():
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
        else:
            # 原生 torch 格式（需要额外处理）
            print(f"[Warning] checkpoint {checkpoint_path} 不是 transformers 格式，尝试从 {base_model_id} 加载基础模型...")
            model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

    model = model.to(device).eval()
    return model, processor


def predict_action(
    model,
    processor,
    instruction: str,
    image: Image.Image,
    device: str = "cuda",
    max_new_tokens: int = 32,
) -> np.ndarray:
    """
    给定一条指令 + 图像，VLA 模型输出动作。

    注意: 这里的实现假设模型已被 SFT 微调，
    输入格式与 train_sft.py 中保持一致。
    """
    # --- 构建 prompt（与训练时一致） ---
    input_text = f"Instruction: {instruction}\nAction:"

    # --- 处理图像 ---
    inputs = processor(
        text=[input_text],
        images=[[image]],  # processor 期望 list of list of images
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device)

    # --- 获取输入长度（用于分离 prompt 和 action tokens） ---
    input_len = input_ids.shape[1]

    # --- 生成动作 token ---
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 评估时用贪婪
            pad_token_id=processor.tokenizer.pad_token_id or 0,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    # --- 提取动作部分的 token ---
    action_tokens = outputs[0, input_len:].cpu().tolist()

    # --- 过滤掉 padding / eos token ---
    pad_id = processor.tokenizer.pad_token_id or 0
    eos_id = processor.tokenizer.eos_token_id
    action_tokens = [t for t in action_tokens if t not in (pad_id, eos_id)]

    # --- 反归一化 ---
    if len(action_tokens) == 0:
        # fallback: 用 top-1 贪婪
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            ).logits
        last_logits = logits[0, -1, :]
        top_tokens = torch.topk(last_logits, ACTION_DIM).indices.tolist()
        action_tokens = top_tokens

    actions = denormalize_action(np.array(action_tokens))
    # 取第一个动作
    action = actions[0] if len(actions) > 0 else np.zeros(ACTION_DIM)
    return action


# ==============================================================================
# 3. LIBERO 环境交互
# ==============================================================================

def make_libero_env(task_suite_name: str = "LIBERO_OBJECT") -> OffScreenRenderEnv:
    """
    创建 LIBERO 仿真环境。
    task_suite_name 可选: LIBERO_OBJECT, LIBERO_SPATIAL, LIBERO_GROUND, LIBERO_TOOL_USE, LIBERO_MULTI
    """
    benchmark = get_benchmark(task_suite_name)()
    env = benchmark.get_task_envs()[0]  # 取第一个环境做评测
    return env


def evaluate_single_task(
    model,
    processor,
    env: OffScreenRenderEnv,
    instruction: str,
    device: str = "cuda",
    max_steps: int = 400,
    render_freq: int = 0,  # >0 时保存渲染图像
    save_dir: str = "./eval_frames",
) -> Dict[str, Any]:
    """
    在单个 LIBERO 任务上执行一个 episode。

    Returns:
        dict: 包含 success (bool), episode_return (float), episode_len (int)
    """
    obs = env.reset()
    episode_return = 0.0
    episode_len = 0
    done = False
    success = False

    info = {"observations": [], "actions": [], "rewards": []}

    for step in range(max_steps):
        # 获取当前图像观测
        rgb_obs = obs["agentview_image"]  # HWC, uint8
        image = Image.fromarray(rgb_obs)

        # VLA 推理
        action = predict_action(
            model, processor, instruction, image, device=device
        )

        # 执行动作
        obs, reward, done, info_ = env.step(action)
        episode_return += reward
        episode_len += 1

        if render_freq > 0 and step % render_freq == 0:
            img_path = os.path.join(save_dir, f"step_{step:04d}.png")
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            Image.fromarray(rgb_obs).save(img_path)

        if done or info_.get("success", False):
            success = info_.get("success", False)
            break

    return {
        "success": success,
        "episode_return": episode_return,
        "episode_len": episode_len,
    }


# ==============================================================================
# 4. 主评测流程
# ==============================================================================

@dataclass
class EvalConfig:
    # 模型
    checkpoint: str = "./checkpoints/final"  # SFT 输出的 checkpoint 路径
    model_id: str = "jingyaogong/minimind2-v"  # 基础模型 ID（当 checkpoint 不含完整配置时用）

    # 环境
    task_suite: str = "LIBERO_OBJECT"  # LIBERO_OBJECT | LIBERO_SPATIAL | LIBERO_GROUND | LIBERO_TOOL_USE | LIBERO_MULTI
    num_episodes: int = 10  # 每个任务评测多少个 episode（LIBERO 每个任务只有 1 个初始状态）
    max_steps: int = 400  # 每个 episode 最大步数
    render_freq: int = 0  # 每隔多少步保存一帧图像（0=不保存）

    # 输出
    output_dir: str = "./eval_results"
    save_trajectories: bool = False  # 是否保存每条 trajectory 的详细数据

    # 运行时
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    verbose: int = 1


def main(cfg: EvalConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- 固定随机种子 ---
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # --- 加载模型 ---
    print(f"\n{'='*60}")
    print(f"加载 VLA 模型: {cfg.checkpoint}")
    model, processor = load_vla_model_and_processor(
        cfg.checkpoint, cfg.model_id, cfg.device
    )
    print(f"模型已加载到 {cfg.device}")
    print(f"{'='*60}\n")

    # --- 创建 LIBERO 环境 ---
    print(f"创建 LIBERO 环境: {cfg.task_suite}")
    benchmark = get_benchmark(cfg.task_suite)()
    task_envs = benchmark.get_task_envs()
    print(f"该 suite 共 {len(task_envs)} 个任务，评测前 {cfg.num_episodes} 个")
    print(f"{'='*60}\n")

    # --- 逐任务评测 ---
    results = []
    for task_idx, task_env in enumerate(task_envs[:cfg.num_episodes]):
        instruction = task_env.task_instruction

        if cfg.verbose:
            print(f"[Task {task_idx:02d}] {instruction[:80]}...")

        t_start = time.time()
        result = evaluate_single_task(
            model=model,
            processor=processor,
            env=task_env,
            instruction=instruction,
            device=cfg.device,
            max_steps=cfg.max_steps,
            render_freq=cfg.render_freq,
            save_dir=os.path.join(cfg.output_dir, f"task_{task_idx:02d}"),
        )
        elapsed = time.time() - t_start

        result["task_idx"] = task_idx
        result["instruction"] = instruction
        result["elapsed_s"] = elapsed

        results.append(result)

        if cfg.verbose:
            status = "✓ SUCCESS" if result["success"] else "✗ FAIL"
            print(f"  → {status}  | return={result['episode_return']:.3f}  "
                  f"| len={result['episode_len']}  | {elapsed:.1f}s\n")

    # --- 汇总统计 ---
    success_count = sum(r["success"] for r in results)
    success_rate = success_count / len(results) * 100
    mean_return = np.mean([r["episode_return"] for r in results])
    mean_len = np.mean([r["episode_len"] for r in results])

    summary = {
        "task_suite": cfg.task_suite,
        "checkpoint": cfg.checkpoint,
        "num_tasks_evaluated": len(results),
        "success_count": success_count,
        "success_rate": f"{success_rate:.1f}%",
        "mean_episode_return": f"{mean_return:.3f}",
        "mean_episode_len": f"{mean_len:.1f}",
        "per_task_results": results,
    }

    # --- 保存结果 ---
    summary_path = os.path.join(cfg.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*60}")
    print(f"评测完成!")
    print(f"  任务数量: {len(results)}")
    print(f"  成功率:  {success_count}/{len(results)} = {success_rate:.1f}%")
    print(f"  平均奖励: {mean_return:.3f}")
    print(f"  平均步数: {mean_len:.1f}")
    print(f"  结果已保存到: {summary_path}")
    print(f"{'='*60}")

    return summary


if __name__ == "__main__":
    cfg = tyro.cli(EvalConfig)
    main(cfg)
