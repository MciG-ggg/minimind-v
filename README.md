# miniVLA

<div align="center">

[![GitHub Repo stars](https://img.shields.io/github/stars/MciG-ggg/miniVLA?style=social)](https://github.com/MciG-ggg/miniVLA/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/MciG-ggg/miniVLA?v=1)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/MciG-ggg/miniVLA)](https://github.com/MciG-ggg/miniVLA/commits/master)

</div>

<div align="center">

中文 | [English](./README_en.md)

</div>

基于 [MiniMind2-V](https://huggingface.co/jingyaogong/minimind2-v) 的轻量级 VLA（Vision-Language-Action）项目。通过行为克隆（Behavioral Cloning）将视觉-语言模型扩展为机器人策略模型，在 [LIBERO](https://libero-project.github.io/) 仿真平台上实现端到端的视觉引导机械臂控制。

> 本项目 fork 自 [minimind-v](https://github.com/jingyaogong/minimind-v)，仅保留 VLA 相关代码，移除了原始 VLM 训练流程。

---

## 核心思路

VLA 将动作视为一种特殊的"语言 token"，让 VLM 直接输出可执行的机器人动作序列：

```
VLM 输入: 图像 + 文本指令  →  VLM 输出:  动作文本  →  反归一化  →  机器人执行
```

动作离散化：每维动作归一化到 `[0, 999]`（1000 个 bin），7 维动作（6 关节 + 夹爪）拼接为 `"123 456 78 234 567 89 345"` 形式的文本序列。

架构上不做任何修改，仅通过 **数据设计 + system prompt + 后处理管道** 实现动作输出。

> 详细设计文档见 [docs/VLA-structure.md](./docs/VLA-structure.md) 和 [docs/VLA0.md](./docs/VLA0.md)。

---

## 环境安装

```bash
# 基础依赖（minimind-v requirements.txt）
pip install -r requirements.txt

# VLA / LIBERO 额外依赖
pip install libero==0.0.1 mujoco==3.2.3 mujoco-mjx==3.2.3 imageio==2.37.0
```

> `libero` 安装后会自动从 HuggingFace 下载资产。如需离线部署，可将 `~/.cache/huggingface/modules/modules--libero-libero-*/assets` 目录缓存到目标机器，并设置环境变量 `LIBERO_ASSET_DIR`。

---

## 快速开始

### 1. 下载基础模型权重

VLA SFT 以 [jingyaogong/minimind2-v](https://huggingface.co/jingyaogong/minimind2-v) 作为基础模型：

```bash
# 方式1：从 HuggingFace 下载（推荐）
git lfs install
git clone https://huggingface.co/jingyaogong/minimind2-v ./checkpoints/minimind2-v

# 方式2：从 ModelScope 下载
git clone https://www.modelscope.cn/models/jingyaogong/minimind2-v ./checkpoints/minimind2-v
```

> **为什么要下载基础模型？** 训练脚本从 `--model_path_or-id` 加载 `MiniMindVLM` 架构（含 `processor` 和视觉投影层），而不是裸的 LLM。直接用 `jingyaogong/minimind2-v` 作为起点进行 SFT。

下载后目录结构应包含 `config.json`、`pytorch_model.bin`、`processor/` 等文件。

### 2. 数据准备（LIBERO → VLA0 格式）

LIBERO 原始数据从 HuggingFace 加载，经格式转换后用于训练。

```bash
cd vla-scripts

# 预览原始数据集和转换后格式
python prepare_data.py show

# 执行格式转换（以 libero_object 为例）
# --task_index 指定过滤的任务 ID（每个 ID 对应一个独立任务）
python prepare_data.py convert --task_index 0 --output ./vla0_libero_object_train
```

转换后的数据集格式：

```json lines
{
  "input_text": "Instruction: pick up the black bowl\nAction:",
  "action_text": "123 456 78 234 567 89 345",
  "images": <PIL.Image>
}
```

`prepare_data.py` 支持以下子命令：

| 命令 | 说明 |
|------|------|
| `show` | 预览原始 + 转换后数据集样本 |
| `convert` | 执行完整转换流程 |
| `all` | 预览 + 转换（一步完成）|

支持的 LIBERO 任务套件：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`。

### 3. SFT 训练

```bash
cd vla-scripts

# 预热阶段：从基础 VLM 开始 SFT
python train_sft.py \
    --model_path_or_id jingyaogong/minimind2-v \
    --dataset_path ./vla0_libero_object_train \
    --output_dir ./checkpoints \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-6 \
    --bf16

# 继续微调：从上一次 checkpoint 恢复
python train_sft.py \
    --model_path_or_id ./checkpoints/final \
    --dataset_path ./vla0_libero_object_train \
    --output_dir ./checkpoints \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-6 \
    --bf16
```

关键参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path_or_id` | `./checkpoints/final` | HuggingFace model_id 或本地 checkpoint 路径 |
| `--dataset_path` | `./vla0_libero_object_train` | VLA0 格式数据集路径 |
| `--output_dir` | `./checkpoints` | 权重输出目录 |
| `--epochs` | 3 | 训练轮数 |
| `--batch_size` | 4 | 每卡 batch size |
| `--learning_rate` | 5e-6 | 学习率 |
| `--bf16` | True | 使用 bf16 混合精度 |
| `--resume` | "" | 从 checkpoint 恢复训练 |
| `--device` | `cuda` | 设备（支持 `cuda`, `cuda:0`, `cuda:1` 等）|

训练完成后最终权重保存在 `{output_dir}/final/`。

### 4. LIBERO 评测

```bash
cd vla-scripts

python eval_libero.py \
    --checkpoint ./checkpoints/final \
    --model_id jingyaogong/minimind2-v \
    --task_suite LIBERO_OBJECT \
    --num_episodes 10 \
    --max_steps 400 \
    --output_dir ./eval_results
```

关键参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | `./checkpoints/final` | VLA 模型权重路径 |
| `--model_id` | `jingyaogong/minimind2-v` | 基础模型 ID（用于加载 processor） |
| `--task_suite` | `LIBERO_OBJECT` | LIBERO 任务套件 |
| `--num_episodes` | 10 | 评测任务数 |
| `--max_steps` | 400 | 每 episode 最大步数 |
| `--output_dir` | `./eval_results` | 结果输出目录 |

评测结果汇总到 `eval_results/summary.json`，包含每个任务的成功状态、奖励、步数，以及整体成功率。

支持的 LIBERO 任务套件：`LIBERO_SPATIAL`、`LIBERO_OBJECT`、`LIBERO_GOAL`、`LIBERO_10`。

---

## 项目结构

```
miniVLA/
├── model/
│   └── model_vla.py           # VLAModel 推理封装 + load_vla_model
├── vla-scripts/
│   ├── train_sft.py           # VLA SFT 训练脚本 (HuggingFace Trainer)
│   ├── eval_libero.py         # LIBERO 仿真评测脚本
│   ├── prepare_data.py        # 数据集预览与格式转换
│   └── rich_helpers.py        # 终端美化输出工具
├── docs/
│   ├── VLA-structure.md        # VLA 架构设计空间分析
│   ├── VLA0.md               # VLA-0 行为克隆详解
│   └── sft.md                 # SFT on LIBERO 原理
└── checkpoints/              # 模型权重输出目录
    ├── minimind2-v/           # 基础 VLM 权重（需手动下载）
    └── final/                 # SFT 后的 VLA 权重
```

---

## 技术细节

### System Prompt

训练与推理统一使用以下 system prompt：

```
System Prompt. Analyze the input image and predict robot actions
for the next H timesteps. Each action has D dimensions. Output a single
sequence of H × D integers (0 B each), representing the H timesteps
sequentially. Provide only space-separated numbers. Nothing else.
```

### 动作格式

| 维度 | 含义 |
|------|------|
| 0-5 | 6 自由度机械臂关节 |
| 6 | 夹爪（0=打开, 999=关闭）|

动作归一化范围：`[ACTION_LOW=-1.0, ACTION_HIGH=1.0]` → 离散化到 `[0, 999]`。

### Loss Mask

训练时仅对 **动作部分 token** 计算交叉熵损失，图像和指令部分的 label 设为 `-100`（忽略）。这是行为克隆的核心——输入部分不参与梯度更新。

---

## License

本项目基于 [Apache-2.0 License](./LICENSE)。基础模型 [MiniMind2-V](https://huggingface.co/jingyaogong/minimind2-v) 遵循其原始许可证。
