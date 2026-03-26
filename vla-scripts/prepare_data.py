"""
LIBERO 数据集预览与 VLA0 格式转换
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from datasets import Dataset, load_dataset, load_from_disk
from rich.table import Table

from rich_helpers import ok, info, err, fatal, section, kv_table, cols_table, sample_table, console

# ---------------------------------------------------------------------------
# TASK_MAP
# ---------------------------------------------------------------------------

from libero.libero import benchmark

TASK_MAP: dict[int, tuple[str, str]] = {}
_idx = 0
for _name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    _ts = benchmark.get_benchmark_dict()[_name]()
    for _i in range(_ts.n_tasks):
        _t = _ts.get_task(_i)
        TASK_MAP[_idx] = (_name, _t.language)
        _idx += 1


# ---------------------------------------------------------------------------
# 核心逻辑
# ---------------------------------------------------------------------------

def normalize_action(
    action,
    low: float = -1,
    high: float = 1,
    token_range: tuple[int, int] = (0, 999),
) -> np.ndarray:
    action = np.array(action, dtype=np.float64)
    return (
        (action - low) / (high - low) * (token_range[1] - token_range[0]) + token_range[0]
    ).astype(int)


def convert_sample(sample: dict) -> dict:
    _, language = TASK_MAP.get(sample["task_index"], (None, f"Task {sample['task_index']}"))
    norm = normalize_action(sample["action"])
    # 【修复 BPE tokenization 与 action token 不匹配问题】
    # 旧格式: "25 864 20 25 26 1713 4794" → BPE digit-by-digit split，模型学会单 digit 输出
    # 新格式: "025x864x020x025x026x713x494" → 3-digit zero-padded + x 分隔，
    #   每个 action value = 3 digits + 1 separator (x) = 4 tokens，7 actions = 28 tokens
    #   x 是 tokenizer 的单 token (ID=90)，digit 是单 token (0-9 → IDs 18-27)
    # 推理时 decode → regex 提取 3-digit groups → 正确还原 action tokens
    action_str = "x".join(f"{int(a):03d}" for a in norm)
    # 保留双视角图像：主视角 + 腕部视角（训练时在 DataCollator 中拼接为 6 通道）
    return {
        "input_text": f"Instruction: {language}\nAction:",
        "action_text": action_str,
        "images": [sample["observation.images.image"], sample["observation.images.image2"]],
    }


# ---------------------------------------------------------------------------
# 共享预览组件
# ---------------------------------------------------------------------------

def show_task_distribution(dataset: Dataset, n: int = 1000) -> None:
    """打印 task_index 分布表"""
    task_indices: set[int] = set()
    for s in dataset.take(min(n, len(dataset))):
        task_indices.add(s["task_index"])

    tbl = Table(box=None)
    tbl.add_column("task_index", style="cyan", justify="right")
    tbl.add_column("benchmark", style="yellow")
    tbl.add_column("语言指令", style="white")
    for ti in sorted(task_indices):
        bm, lang = TASK_MAP.get(ti, ("?", "???"))
        tbl.add_row(str(ti), bm, f"{lang[:55]}..." if len(lang) > 55 else lang)
    console.print(tbl)


def show_raw_samples(dataset: Dataset, n: int = 3) -> None:
    """打印原始数据集前 n 条样本"""
    info(f"总样本数: {len(dataset):,}\n")
    cols_table(dataset.column_names)
    console.print()

    for i in range(min(n, len(dataset))):
        s = dataset[i]
        img0 = np.array(s["observation.images.image"])
        img1 = np.array(s["observation.images.image2"])
        _, lang = TASK_MAP.get(s["task_index"], ("?", "???"))
        sample_table(
            f"原始样本 #{i}",
            [
                ("task_index", f"{s['task_index']}  →  {TASK_MAP.get(s['task_index'], ('?',))[0]}"),
                ("语言指令", lang),
                ("action", str(s["action"])),
                ("action 范围", f"min={min(s['action']):.4f}  max={max(s['action']):.4f}"),
                ("主视角图像", f"shape={img0.shape}  dtype={img0.dtype}  range=[{img0.min()},{img0.max()}]"),
                ("腕部图像", f"shape={img1.shape}  dtype={img1.dtype}"),
                ("robot_state", str(s["observation.state"])[:80]),
                ("meta", f"episode={s['episode_index']}  frame={s['frame_index']}  ts={s['timestamp']}"),
            ],
        )
        console.print()


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = str(Path(__file__).parent.parent / "vla0_libero_object_train")


@dataclass
class Show:
    """预览原始 + 转换后数据集样本"""

    n: int = 3
    converted_path: str = ""

    def run(self) -> None:
        if not self.converted_path:
            self.converted_path = DEFAULT_OUTPUT_DIR

        section("加载原始数据集", "cyan")
        dataset = load_dataset("HuggingFaceVLA/libero", "default", split="train")
        ok(f"加载完成: {len(dataset):,} 条\n")

        section("task_index 分布 (前 1000 条)", "cyan")
        show_task_distribution(dataset)
        console.print()

        section("原始样本预览", "cyan")
        show_raw_samples(dataset, n=self.n)

        section("转换后 VLA0 样本预览", "green")
        try:
            converted = load_from_disk(self.converted_path)
        except FileNotFoundError:
            err(f"找不到转换后数据集: {self.converted_path}，请先运行 convert 命令")
            return

        info(f"总样本数: {len(converted):,}\n")
        cols_table(converted.column_names)
        console.print()
        for i in range(min(self.n, len(converted))):
            s = converted[i]
            imgs = s["images"]  # list of 2 PIL.Image
            img0, img1 = [np.array(img) for img in imgs]
            sample_table(
                f"转换后样本 #{i}",
                [
                    ("input_text", s["input_text"].replace("\n", " ")[:80]),
                    ("action_text", s["action_text"]),
                    ("主视角图像", f"shape={img0.shape}  dtype={img0.dtype}  range=[{img0.min()},{img0.max()}]"),
                    ("腕部图像", f"shape={img1.shape}  dtype={img1.dtype}  range=[{img1.min()},{img1.max()}]"),
                ],
                style="green",
            )
            console.print()


@dataclass
class Convert:
    """执行完整转换流程"""

    output: str = ""
    task_index: int = 0

    def run(self) -> None:
        if not self.output:
            self.output = DEFAULT_OUTPUT_DIR

        section("加载数据集", "cyan")
        dataset = load_dataset("HuggingFaceVLA/libero", "default", split="train")
        ok(f"加载完成: {len(dataset):,} 条\n")

        kv_table("LIBERO 数据集概览", [
            ("数据集名称", "HuggingFaceVLA/libero"),
            ("配置 / 划分", "default / train"),
            ("原始样本总数", f"{len(dataset):,}"),
            ("task_index 映射范围", f"0 – {len(TASK_MAP) - 1}  ({len(TASK_MAP)} tasks)"),
            ("过滤条件", f"task_index == {self.task_index}"),
        ])
        console.print()

        filtered = dataset.filter(lambda x: x["task_index"] == self.task_index)
        if len(filtered) == 0:
            fatal(f"task_index={self.task_index} 没有样本，请先运行 show 查看可用分布")
        ok(f"过滤后: {len(filtered):,} 条\n")

        section("格式转换 (VLA0)", "cyan")
        converted = filtered.map(convert_sample, remove_columns=filtered.column_names, desc="转换中")
        ok(f"转换完成: {len(converted):,} 条\n")

        section("保存到磁盘", "cyan")
        converted.save_to_disk(self.output)
        ok(f"已保存: {self.output}\n")

        Show(n=3, converted_path=self.output).run()


@dataclass
class All:
    """预览 + 转换"""

    output: str = ""
    task_index: int = 0
    n: int = 3

    def run(self) -> None:
        if not self.output:
            self.output = DEFAULT_OUTPUT_DIR

        section("加载数据集", "cyan")
        dataset = load_dataset("HuggingFaceVLA/libero", "default", split="train")
        ok(f"加载完成: {len(dataset):,} 条\n")

        section("task_index 分布 (前 1000 条)", "cyan")
        show_task_distribution(dataset)
        console.print()

        section("原始样本预览", "cyan")
        show_raw_samples(dataset, n=self.n)

        Convert(output=self.output, task_index=self.task_index).run()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

Commands = Show | Convert | All

if __name__ == "__main__":
    tyro.extras.set_accent_color("cyan")
    tyro.cli(Commands).run()
