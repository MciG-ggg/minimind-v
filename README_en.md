# miniVLA

<div align="center">

[![GitHub Repo stars](https://img.shields.io/github/stars/MciG-ggg/miniVLA?style=social)](https://github.com/MciG-ggg/miniVLA/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/MciG-ggg/miniVLA?v=1)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/MciG-ggg/miniVLA)](https://github.com/MciG-ggg/miniVLA/commits/master)

</div>

<div align="center">

[中文](./README.md) | English

</div>

A lightweight VLA (Vision-Language-Action) project built on top of [MiniMind2-V](https://huggingface.co/jingyaogong/minimind2-v). Through Behavioral Cloning, the vision-language model is extended into a robot policy model that performs end-to-end vision-guided arm manipulation on the [LIBERO](https://libero-project.github.io/) simulation platform.

> This project is forked from [minimind-v](https://github.com/jingyaogong/minimind-v), keeping only VLA-related code and removing the original VLM training pipeline.

---

## Core Idea

VLA treats actions as a special kind of "language token", enabling the VLM to directly output executable robot action sequences:

```
VLM input:  image + text instruction  →  VLM output:  action text  →  denormalize  →  robot execution
```

Action discretization: each action dimension is normalized to `[0, 999]` (1000 bins), and 7-dimensional actions (6 joints + gripper) are concatenated into text sequences like `"123 456 78 234 567 89 345"`.

No architecture modifications are made. The VLA capability is achieved purely through **data design + system prompt + post-processing pipeline**.

> See [docs/VLA-structure.md](./docs/VLA-structure.md) and [docs/VLA0.md](./docs/VLA0.md) for detailed design documentation.

---

## Environment Setup

```bash
# Base dependencies (minimind-v requirements.txt)
pip install -r requirements.txt

# VLA / LIBERO extra dependencies
pip install libero==0.0.1 mujoco==3.2.3 mujoco-mjx==3.2.3 imageio==2.37.0
```

> `libero` automatically downloads assets from HuggingFace on install. For offline deployment, cache the `~/.cache/huggingface/modules/modules--libero-libero-*/assets` directory to the target machine and set `LIBERO_ASSET_DIR`.

---

## Quick Start

### 1. Download Base Model Weights

VLA SFT uses [jingyaogong/minimind2-v](https://huggingface.co/jingyaogong/minimind2-v) as the base model:

```bash
# Option 1: From HuggingFace (recommended)
git lfs install
git clone https://huggingface.co/jingyaogong/minimind2-v ./checkpoints/minimind2-v

# Option 2: From ModelScope
git clone https://www.modelscope.cn/models/jingyaogong/minimind2-v ./checkpoints/minimind2-v
```

> **Why download the base model?** The training script loads the `MiniMindVLM` architecture (including `processor` and vision projection layers) from `--model_path_or_id`, not a bare LLM. Starting SFT from `jingyaogong/minimind2-v` initializes all required components.

After downloading, the directory should contain `config.json`, `pytorch_model.bin`, `processor/`, etc.

### 2. Data Preparation (LIBERO → VLA0 Format)

LIBERO raw data is loaded from HuggingFace and converted to the VLA0 training format.

```bash
cd vla-scripts

# Preview raw and converted dataset samples
python prepare_data.py show

# Run format conversion (using libero_object as example)
# --task_index filters by task ID (each ID corresponds to one independent task)
python prepare_data.py convert --task_index 0 --output ./vla0_libero_object_train
```

Converted dataset format:

```json lines
{
  "input_text": "Instruction: pick up the black bowl\nAction:",
  "action_text": "123 456 78 234 567 89 345",
  "images": <PIL.Image>
}
```

`prepare_data.py` subcommands:

| Command | Description |
|---------|-------------|
| `show` | Preview raw + converted dataset samples |
| `convert` | Run full conversion pipeline |
| `all` | Preview + convert in one step |

Supported LIBERO task suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.

### 3. SFT Training

```bash
cd vla-scripts

# Warmup: start SFT from base VLM
python train_sft.py \
    --model_path_or_id jingyaogong/minimind2-v \
    --dataset_path ./vla0_libero_object_train \
    --output_dir ./checkpoints \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-6 \
    --bf16

# Continue fine-tuning: resume from previous checkpoint
python train_sft.py \
    --model_path_or_id ./checkpoints/final \
    --dataset_path ./vla0_libero_object_train \
    --output_dir ./checkpoints \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-6 \
    --bf16
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path_or_id` | `./checkpoints/final` | HuggingFace model_id or local checkpoint path |
| `--dataset_path` | `./vla0_libero_object_train` | VLA0 format dataset path |
| `--output_dir` | `./checkpoints` | Weight output directory |
| `--epochs` | 3 | Number of training epochs |
| `--batch_size` | 4 | Per-device batch size |
| `--learning_rate` | 5e-6 | Learning rate |
| `--bf16` | True | Use bf16 mixed precision |
| `--resume` | "" | Resume from checkpoint |
| `--device` | `cuda` | Device (supports `cuda`, `cuda:0`, `cuda:1`, etc.) |

Final checkpoint is saved to `{output_dir}/final/`.

### 4. LIBERO Evaluation

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

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | `./checkpoints/final` | VLA model checkpoint path |
| `--model_id` | `jingyaogong/minimind2-v` | Base model ID (for loading processor) |
| `--task_suite` | `LIBERO_OBJECT` | LIBERO task suite |
| `--num_episodes` | 10 | Number of evaluation tasks |
| `--max_steps` | 400 | Max steps per episode |
| `--output_dir` | `./eval_results` | Results output directory |

Evaluation results are summarized in `eval_results/summary.json`, including per-task success status, reward, episode length, and overall success rate.

Supported LIBERO task suites: `LIBERO_SPATIAL`, `LIBERO_OBJECT`, `LIBERO_GOAL`, `LIBERO_10`.

---

## Project Structure

```
miniVLA/
├── model/
│   └── model_vla.py           # VLAModel inference wrapper + load_vla_model
├── vla-scripts/
│   ├── train_sft.py           # VLA SFT training script (HuggingFace Trainer)
│   ├── eval_libero.py         # LIBERO simulation evaluation script
│   ├── prepare_data.py        # Dataset preview and format conversion
│   └── rich_helpers.py        # Terminal UI utilities
├── docs/
│   ├── VLA-structure.md        # VLA architecture design space analysis
│   ├── VLA0.md               # VLA-0 behavioral cloning deep dive
│   └── sft.md                 # SFT on LIBERO principles
└── checkpoints/              # Model weight output directory
    ├── minimind2-v/           # Base VLM weights (to be downloaded manually)
    └── final/                 # SFT'd VLA weights
```

---

## Technical Details

### System Prompt

Training and inference share a unified system prompt:

```
System Prompt. Analyze the input image and predict robot actions
for the next H timesteps. Each action has D dimensions. Output a single
sequence of H × D integers (0 B each), representing the H timesteps
sequentially. Provide only space-separated numbers. Nothing else.
```

### Action Format

| Dimension | Meaning |
|-----------|---------|
| 0-5 | 6-DOF robotic arm joints |
| 6 | Gripper (0=open, 999=close) |

Action normalization range: `[ACTION_LOW=-1.0, ACTION_HIGH=1.0]` → discretized to `[0, 999]`.

### Loss Mask

During training, cross-entropy loss is computed **only on action tokens**. Image and instruction token labels are set to `-100` (ignored). This is the core of behavioral cloning — input portions do not contribute to gradient updates.

---

## License

This project is licensed under [Apache-2.0 License](./LICENSE). The base model [MiniMind2-V](https://huggingface.co/jingyaogong/minimind2-v) follows its original license.
