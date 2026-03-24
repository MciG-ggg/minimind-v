# VLA-0: 基于 MiniMind-V 的视觉-语言-动作模型

> 本文档描述如何在 MiniMind-V (VLM) 基础上实现 VLA-0 (Vision-Language-Action) 范式，使模型能够根据视觉观察和语言指令输出机器人可执行的动作。

---

## 1. VLA-0 概述

### 1.1 什么是 VLA？

VLA (Vision-Language-Action) 是对 VLM 的自然扩展。VLM 将图像作为"外语"翻译为 LLM 可理解的 token 序列；VLA 则更进一步——将动作（Action）也视为一种特殊的"语言 token"，让模型直接输出可执行的机器人动作序列。

### 1.2 VLA-0 的设计哲学

VLA-0 是最简单的 VLA 实现方式：**不做任何架构修改，仅通过数据设计和后处理管道让现有 VLM 输出动作**。这验证了"token 即一切"的范式——只要能序列化，就能自回归生成。

```
VLM 输入:  图像 + 文本指令  →  VLM 输出:  动作文本  →  反归一化  →  机器人执行
```

### 1.3 与现有 VLM 的关系

MiniMind-V 的核心架构（Vision Encoder + Projection + LLM）完全适用于 VLA-0：

| 组件 | VLM 模式 | VLA-0 模式 |
|------|----------|------------|
| Vision Encoder | CLIP (冻结) | CLIP (冻结) |
| Projection | vision_proj | vision_proj (可训练) |
| LLM | 输出文本 token | 输出动作数字 token |
| 后处理 | 无 | 文本解析 + 反归一化 |

---

## 2. 动作生成与处理

VLA-0 输出的文字需要后处理才能转换为机器人可执行的动作。

### 2.1 完整流程

```
VLM 输出文字 → 解析数字字符串 → 反归一化 → 动作向量 → 机器人执行
```

### 2.2 输出格式定义

训练时，让模型输出固定格式的动作字符串。动作值在 `[0, 999]` 范围内离散化（每维 1000 个 bin），用空格分隔不同关节，用 `|` 分隔不同时间步。

```python
# 单步动作（7个关节 + 1个夹爪 = 8维）
"123 456 78 234 567 89 345 12"

# 动作块（3步，每步8个值，用 | 分隔）
"123 456 78 234 567 89 345 12 | 124 457 79 235 568 90 346 13 | 125 458 80 236 569 91 347 14"
```

**动作维度说明**（以 LIBERO 机械臂为例）：

| 维度 | 含义 | 典型范围 |
|------|------|----------|
| 0-5 | 6自由度机械臂关节角度或末端位置 | 视具体环境 |
| 6 | 夹爪开合 | 0（开）/ 999（关）|

> 注：实际维度根据机器人平台而定，VLA-0 不限制维度数量，只需在数据准备和解析时保持一致即可。

### 2.3 推理时解析

```python
import numpy as np

def parse_action_text(text, chunk_size=1, num_dims=8, joint_limits=None):
    """
    将 VLM 输出的文本解析为动作数组

    Args:
        text: 模型输出的字符串，如 "123 456 78 234 567 89 345 12 | 124 ..."
        chunk_size: 动作块长度（几步）
        num_dims: 每个动作的维度（如 7关节+1夹爪=8）
        joint_limits: 每个维度的 [min, max] 范围，如 [[-1.0, 1.0], ...]

    Returns:
        actions: shape (chunk_size, num_dims) 的 numpy 数组
    """
    if joint_limits is None:
        # 默认：每个维度归一化到 [0, 1]
        joint_limits = [[0.0, 1.0]] * num_dims

    # 1. 按 | 分割动作块
    chunks = text.strip().split("|")

    actions = []
    for chunk in chunks[:chunk_size]:
        # 2. 按空格分割数字
        tokens = chunk.strip().split()
        if len(tokens) != num_dims:
            # 容错：如果输出不完整，用默认值填充
            print(f"Warning: expected {num_dims} tokens, got {len(tokens)}")
            tokens = tokens[:num_dims]
            while len(tokens) < num_dims:
                tokens.append("500")  # 中间值填充

        # 3. 转换为整数
        int_actions = [int(t) for t in tokens]

        # 4. 反归一化：从 [0, 999] 映射回原始动作范围
        real_actions = []
        for i, val in enumerate(int_actions):
            normalized = val / 999.0          # 归一化到 [0, 1]
            min_val, max_val = joint_limits[i]
            real_val = min_val + normalized * (max_val - min_val)
            real_actions.append(real_val)

        actions.append(real_actions)

    return np.array(actions)
```

### 2.4 带容错机制的增强版解析

100M 模型生成多步长数字字符串时，容易出现数字不完整、格式错误等问题。以下增强版解析加入了更强的容错能力：

```python
import re

def robust_parse_action_text(text, num_dims=8, default_value=500):
    """
    带容错机制的解析

    适用于：
    - 输出中混入无关文本（VLM 可能添加解释性文字）
    - 数字之间缺少空格
    - 动作块数量不足
    """
    # 1. 清理文本：只保留数字、空格、|
    clean_text = re.sub(r'[^0-9\s\|]', '', text)

    # 2. 提取所有数字序列
    chunks = clean_text.split("|")

    actions = []
    for chunk in chunks:
        # 提取所有连续数字
        numbers = re.findall(r'\d+', chunk)
        if len(numbers) >= num_dims:
            # 取前 num_dims 个
            int_vals = [int(n) for n in numbers[:num_dims]]
        else:
            # 不足则用默认值填充
            int_vals = [int(n) for n in numbers]
            int_vals.extend([default_value] * (num_dims - len(numbers)))

        actions.append(int_vals)

    return np.array(actions)
```

### 2.5 完整推理与执行流程

```python
def inference_and_execute(vlm, image, instruction, robot, chunk_size=3,
                           num_dims=8, joint_limits=None):
    """
    完整的 VLA-0 推理与执行流程

    Args:
        vlm: MiniMindVLM 模型实例
        image: PIL.Image 或已处理的图像 tensor
        instruction: 自然语言指令，如 "把红色的积木放到绿色位置"
        robot: 机器人控制接口（需实现 execute(action) 方法）
        chunk_size: 预测的动作步数
    """
    # 1. 图像预处理
    if isinstance(image, Image.Image):
        pixel_values = MiniMindVLM.image2tensor(
            image, vlm.processor
        ).unsqueeze(0).to(vlm.device)

    # 2. 构建 prompt
    messages = [
        {"role": "user", "content": f"Instruction: {instruction}\nAction:"}
    ]
    inputs_text = vlm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = vlm.tokenizer(inputs_text, return_tensors="pt").to(vlm.device)

    # 3. VLM 生成动作文本
    #    每步 8 个数字 + 空格 ≈ 9 token，| 分隔符 ≈ 1 token
    #    chunk_size * (8 + 1) * 4 (每数字平均 4 位)
    generated_ids = vlm.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=chunk_size * 50,          # 保守估计
        do_sample=True,
        temperature=0.1,                          # 低温度，更确定性
        top_p=0.9,
        pad_token_id=vlm.tokenizer.pad_token_id,
        eos_token_id=vlm.tokenizer.eos_token_id,
        pixel_values=pixel_values,
        logits_processor=action_logits_processor()  # 可选：约束解码
    )

    # 4. 解码为文本
    output_text = vlm.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # 5. 解析动作
    actions = robust_parse_action_text(output_text, num_dims=num_dims)

    # 6. 逐步执行
    for action in actions:
        robot.execute(action)
        time.sleep(0.033)  # 30Hz 控制频率
```

---

## 3. 数据准备

VLA-0 的数据格式在 VLM SFT 格式基础上扩展，增加动作标签。

### 3.1 数据集格式

```json lines
{
  "conversations": [
    {
      "role": "user",
      "content": "把桌上的苹果拿起来\n<image>"
    },
    {
      "role": "assistant",
      "content": "123 456 78 234 567 89 345 12"
    }
  ],
  "image": "robot_scene_001.jpg",
  "actions": [
    [123, 456, 78, 234, 567, 89, 345, 12],
    [124, 457, 79, 235, 568, 90, 346, 13],
    [125, 458, 80, 236, 569, 91, 347, 14]
  ],
  "action_chunk_size": 3
}
```

### 3.2 动作归一化

原始机器人动作在训练前需要离散化到 `[0, 999]`：

```python
def normalize_action(action, action_min, action_max, num_bins=1000):
    """
    将连续动作离散化到 [0, num_bins-1]

    Args:
        action: 原始动作值，标量或数组
        action_min: 各维度的最小值列表
        action_max: 各维度的最大值列表
        num_bins: 离散化 bin 数量（默认 1000）

    Returns:
        归一化后的整数动作，范围 [0, 999]
    """
    normalized = (action - action_min) / (action_max - action_min)
    normalized = np.clip(normalized, 0, 1)
    return (normalized * (num_bins - 1)).astype(int)
```

### 3.3 数据来源推荐

| 数据集 | 描述 | 链接 |
|--------|------|------|
| LIBERO | 4 个场景共 480 个任务，6400 条轨迹 | [link](https://libero-project.github.io/) |
| RT-1 | 130K 机器人演示数据 | [link](https://robotics-transformer.github.io/) |
| ManiSkill2 | SAPIEN 模拟器的 2000+ 任务 | [link](https://github.com/haosulab/ManiSkill2) |
| BridgeData | 多机器人多场景跨域数据 | [link](https://bridge-data.github.io/) |

### 3.4 VLA 数据集构建脚本

以下脚本以 LIBERO 为例展示完整的数据转换流程：

```python
# scripts/convert_vla_dataset.py
import json
import numpy as np
from pathlib import Path

# LIBERO 关节范围（示例）
LIBERO_JOINT_LIMITS = [
    [-2.87, 2.87], [-1.97, 1.97], [-2.87, 2.87],
    [-2.87, 2.87], [-2.87, 2.87], [-2.87, 2.87], [0, 1]
]
NUM_BINS = 1000

def normalize_action(action):
    """LIBERO 动作离散化"""
    result = []
    for val, (low, high) in zip(action, LIBERO_JOINT_LIMITS):
        norm = np.clip((val - low) / (high - low), 0, 1)
        result.append(int(norm * (NUM_BINS - 1)))
    return result

def convert_libero_to_vla(libero_json_path, output_path, image_dir, chunk_size=3):
    """
    将 LIBERO 数据转换为 VLA-0 训练格式
    """
    with open(libero_json_path) as f:
        data = json.load(f)

    results = []
    for episode in data["episodes"]:
        instruction = episode["instruction"]
        frames = episode["frames"]

        # 按 chunk_size 分组动作
        for i in range(0, len(frames) - chunk_size + 1, chunk_size):
            chunk = frames[i:i + chunk_size]
            actions = [normalize_action(f["action"]) for f in chunk]

            # 拼接为字符串格式
            action_texts = [" ".join(map(str, a)) for a in actions]
            action_str = " | ".join(action_texts)

            results.append({
                "conversations": [
                    {"role": "user", "content": f"{instruction}\n<image>"},
                    {"role": "assistant", "content": action_str}
                ],
                "image": chunk[0]["image_path"],
                "actions": actions,
                "action_chunk_size": chunk_size
            })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Converted {len(results)} VLA samples -> {output_path}")


if __name__ == "__main__":
    convert_libero_to_vla(
        libero_json_path="data/libero_train.json",
        output_path="../dataset/vla_libero.parquet",
        image_dir="data/images/",
        chunk_size=3
    )
```

---

## 4. 训练

### 4.1 与 VLM SFT 的差异

VLA-0 训练与 VLM SFT 的核心差异在于：

| 方面 | VLM SFT | VLA-0 训练 |
|------|---------|------------|
| 训练目标 | 图像描述 + 对话 | 视觉引导的动作预测 |
| 数据 | 图文对话数据 | 视觉-指令-动作三元组 |
| Loss mask | 对话响应部分 | 动作字符串部分 |
| 可选微调 | vision_proj + LLM | vision_proj + LLM（全部可训练）|
| 数据规模 | 300K 条 | 10K-100K 条轨迹即可验证 |

### 4.2 数据集类

```python
# dataset/vla_dataset.py
import io
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from model.model_vlm import MiniMindVLM

class VLADataset(Dataset):
    def __init__(self, parquet_path, tokenizer, preprocess=None,
                 max_length=512, image_special_token='@' * 196,
                 num_dims=8, chunk_size=3):
        super().__init__()
        import pyarrow.parquet as pq
        self.table = pq.read_table(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.num_dims = num_dims
        self.chunk_size = chunk_size

        # 构建动作 prompt 模板
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.table)

    def create_vla_prompt(self, instruction):
        """构建 VLA 专用的 prompt"""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{instruction}\n{self.image_token}"}],
            tokenize=False, add_generation_prompt=True
        )

    def generate_labels(self, input_ids, action_ids):
        """仅对动作部分计算 loss，图像和指令部分 mask 掉"""
        labels = [-100] * len(input_ids)

        # 找到 assistant 开始位置
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1

        # 如果有独立的动作 token，将动作 token 的 label 也设回动作 ID
        # （与 input_ids 一致，但仅在 assistant 段内）
        action_start = None
        for i in range(len(input_ids) - len(action_ids)):
            if input_ids[i:i + len(action_ids)] == action_ids:
                action_start = i
                break
        if action_start is not None:
            for j in range(len(action_ids)):
                labels[action_start + j] = action_ids[j]

        return labels

    def __getitem__(self, index: int):
        row = {
            "conversations": json.loads(self.table['conversations'][index].as_py()),
            "image_bytes": self.table['image_bytes'][index].as_py(),
        }
        if not isinstance(row["image_bytes"], list):
            row["image_bytes"] = [row["image_bytes"]]

        instruction = row["conversations"][0]["content"].replace("<image>", "")
        action_str = row["conversations"][1]["content"]

        # 构建含图像的 prompt
        prompt = self.create_vla_prompt(instruction)
        prompt_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_length).input_ids

        # 动作 token（作为 label，不作为新的 input_ids）
        action_str = action_str.replace(" | ", " ")  # 展平动作块
        action_ids = self.tokenizer(action_str + self.tokenizer.eos_token, add_special_tokens=False).input_ids

        # 拼接 input_ids：prompt + 动作
        input_ids = prompt_ids + action_ids
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        else:
            input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        labels = self.generate_labels(input_ids, action_ids)

        # 图像处理
        image_tensor = torch.stack([
            MiniMindVLM.image2tensor(
                Image.open(io.BytesIO(img)), self.preprocess
            ) for img in row["image_bytes"]
        ])

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            image_tensor
        )
```

### 4.3 训练脚本

VLA-0 训练脚本可直接基于 `train_sft_vlm.py` 修改，核心差异在于使用 `VLADataset` 而非 `VLMDataset`：

```python
# trainer/train_vla.py
# --- 仅展示关键差异，完整代码参考 train_sft_vlm.py ---
from dataset.vla_dataset import VLADataset

# 数据集：使用 VLA 格式而非普通 SFT 格式
train_ds = VLADataset(
    args.data_path,
    tokenizer,
    preprocess=preprocess,
    image_special_token=vlm_config.image_special_token,
    max_length=vlm_config.max_seq_len,
    num_dims=args.num_dims,
    chunk_size=args.chunk_size
)

# 训练过程与 train_sft_vlm.py 完全一致
# loss = res.loss + res.aux_loss
```

启动训练：

```bash
python -m trainer.train_vla \
    --data_path ../dataset/vla_libero.parquet \
    --from_weight sft_vlm \
    --save_weight vla_0 \
    --epochs 5 \
    --batch_size 4 \
    --learning_rate 3e-5 \
    --num_dims 7 \
    --chunk_size 3 \
    --max_seq_len 1536
```

### 4.4 训练参数建议

| 参数 | VLM SFT | VLA-0 推荐 | 说明 |
|------|---------|------------|------|
| learning_rate | 1e-6 | 3e-5 ~ 5e-5 | VLA 需要更强的更新幅度 |
| epochs | 2 | 5 ~ 10 | 取决于数据规模 |
| freeze_vision_encoder | True | True | CLIP 始终冻结 |
| freeze_llm | 最后1层 | False | VLA 需要 LLM 理解动作语义 |
| chunk_size | N/A | 3 ~ 5 | 建议不超过 5，避免过长输出 |
| num_dims | N/A | 7 ~ 8 | 视机器人平台而定 |

---

## 5. 推理与执行

### 5.1 推理接口

```python
# eval_vla.py
import argparse
import time
import torch
from PIL import Image
from model.model_vlm import MiniMindVLM, VLMConfig
from transformers import AutoTokenizer

def init_vla_model(args):
    """初始化 VLA 模型"""
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    model = MiniMindVLM(
        VLMConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe)
        ),
        vision_model_path="./model/vision_model/clip-vit-base-patch16"
    )
    state_dict = torch.load(args.weight_path, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    return model.eval().to(args.device), tokenizer


def vla_generate(model, image, instruction, tokenizer,
                 chunk_size=3, num_dims=8, max_tokens=200):
    """VLA 推理：图像 + 指令 -> 动作"""
    # 图像处理
    pixel_values = MiniMindVLM.image2tensor(
        image, model.processor
    ).unsqueeze(0).to(model.device)

    # Prompt 构建
    messages = [{"role": "user", "content": f"{instruction}\n{model.params.image_special_token}"}]
    inputs_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(inputs_text, return_tensors="pt").to(model.device)

    # 生成
    generated_ids = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        pixel_values=pixel_values
    )

    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', required=True)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--chunk_size', default=3, type=int)
    parser.add_argument('--num_dims', default=8, type=int)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    model, tokenizer = init_vla_model(args)
    image = Image.open(args.image_path).convert('RGB')

    output = vla_generate(
        model, image, args.instruction, tokenizer,
        chunk_size=args.chunk_size, num_dims=args.num_dims
    )
    print(f"VLM output: {output}")

    actions = robust_parse_action_text(output, num_dims=args.num_dims)
    print(f"Parsed actions:\n{actions}")
```

### 5.2 与机器人控制接口集成

```python
class RobotController:
    """机器人控制接口（示例，需根据实际机器人 SDK 实现）"""

    def __init__(self, joint_limits):
        self.joint_limits = joint_limits
        # self.arm = YourRobotArmSDK()
        # self.gripper = YourGripperSDK()

    def execute(self, action):
        """
        执行一个动作向量

        Args:
            action: shape (num_dims,) 的 numpy 数组
                   假设前 6 维为关节角度，后 1 维为夹爪
        """
        # 夹爪特殊处理
        gripper_value = action[-1]  # 0=打开, 1=关闭
        joint_positions = action[:-1]

        # self.arm.set_joint_positions(joint_positions)
        # self.gripper.set_position(gripper_value)
        print(f"Executing: joints={joint_positions}, gripper={gripper_value:.3f}")

    def close(self):
        # self.arm.disconnect()
        pass
```

---

## 6. 针对 100M VLM 的优化建议

### 6.1 挑战分析

100M 规模的 VLM 在生成动作字符串时面临以下挑战：

| 问题 | 表现 | 影响 |
|------|------|------|
| 输出格式不稳定 | 数字不完整、缺少空格、混入解释性文字 | 解析失败 |
| 动作块太长 | 生成 10 步以上时后期数字质量下降 | 自回归误差累积 |
| 推理不确定性 | 高温度下数值波动大 | 动作抖动 |
| 动作维度多 | 8+ 维输出同时保证质量困难 | 联合分布建模不足 |

### 6.2 解决方案

| 问题 | 解决方案 |
|------|----------|
| 输出格式不稳定 | **Constrained Decoding**：实现 custom logits_processor，只允许数字、空格和 `\|` token 输出 |
| 动作块太长 | 减小 chunk_size（从 10 降到 3-5），用 MPC/ILQR 在低层做轨迹优化 |
| 动作块太长 | **Action Binning**：将连续值映射为固定离散 token（如 0-999 作为独立 vocab 项）|
| 推理不确定性 | **Beam Search**：多路径生成，取与指令最一致的轨迹 |
| 推理不确定性 | **多次采样取平均**：同一指令生成 3-5 次，对动作取均值/中位数 |

### 6.3 Constrained Decoding 实现

```python
from transformers import LogitsProcessor

class ActionTokenLogitsProcessor(LogitsProcessor):
    """
    约束解码：只允许输出动作字符

    允许的 token：
    - 数字 0-9
    - 空格
    - 管道符 |
    - 换行符
    - EOS
    """
    ALLOWED_CHARS = set("0123456789 |")
    NUM_TOKEN_IDS = set(range(10))  # 假设 tokenizer 中 '0'-'9' 的 token id

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # 构建允许的 token id 集合
        self.allowed_ids = set()
        for text in ["0","1","2","3","4","5","6","7","8","9"," ", "|", "\n"]:
            ids = tokenizer(text, add_special_tokens=False).input_ids
            self.allowed_ids.update(ids)
        self.allowed_ids.add(tokenizer.eos_token_id)

    def __call__(self, input_ids, scores):
        # 将不允许的 token 概率设为 -inf
        for i in range(scores.shape[-1]):
            if i not in self.allowed_ids:
                scores[..., i] = float("-inf")
        return scores
```

### 6.4 Beam Search 集成

```python
def vla_beam_generate(model, image, instruction, tokenizer,
                      num_beams=3, chunk_size=3, num_dims=8):
    """
    使用 Beam Search 生成多个候选动作，取最优
    """
    pixel_values = MiniMindVLM.image2tensor(image, model.processor).unsqueeze(0)
    messages = [{"role": "user", "content": f"{instruction}\n{model.params.image_special_token}"}]
    inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs_text, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=chunk_size * 50,
        do_sample=False,
        num_beams=num_beams,
        pixel_values=pixel_values,
    )

    # 收集所有 beam 的结果
    all_outputs = []
    for ids in generated_ids:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        actions = robust_parse_action_text(text, num_dims=num_dims)
        all_outputs.append((text, actions))

    # 简单策略：取第一个完整解析的 beam
    for text, actions in all_outputs:
        if len(actions) >= chunk_size:
            return text, actions

    # Fallback：返回第一个结果
    return all_outputs[0]
```

---

## 7. 动作动作空间与归一化参考

### 7.1 LIBERO 机器人

LIBERO 是 VLA 验证的理想平台，提供了预归一化的动作空间。

```python
LIBERO_ACTION_CONFIG = {
    "num_dims": 7,  # 6 joints + 1 gripper
    "joint_limits": [  # 已归一化到 [0, 1]
        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]
    ],
    "gripper_mapping": {
        "open": 0.0,
        "close": 1.0,
        # 离散化后：open=0, close=999
    }
}
```

### 7.2 实际机械臂（如 UR5）

```python
UR5_ACTION_CONFIG = {
    "num_dims": 8,  # 6 joints + 2 gripper fingers
    "joint_limits": [
        [-2.87, 2.87],   # shoulder_pan
        [-1.97, 1.97],   # shoulder_lift
        [-2.87, 2.87],   # elbow
        [-2.87, 2.87],   # wrist1
        [-2.87, 2.87],   # wrist2
        [-2.87, 2.87],   # wrist3
        [0.0, 1.0],      # gripper_left
        [0.0, 1.0],      # gripper_right
    ],
    "num_bins": 1000,
}
```

---

## 8. 总结与路线图

### 8.1 VLA-0 的定位

VLA-0 是验证 VLM 到 VLA 迁移可行性的**最小可行实现**。它不需要任何架构修改，仅通过数据设计 + 后处理管道实现动作输出。

### 8.2 优势与局限

| 优势 | 局限 |
|------|------|
| 零架构修改，复用全部 VLM 代码 | 动作质量受限于语言模型的数值建模能力 |
| 训练成本低，10K 轨迹即可验证 | 长动作序列的自回归误差累积 |
| 与 VLM SFT 共用同一套训练框架 | Constrained Decoding 需要额外的 vocab 控制 |
| 可快速迭代数据策略 | 100M 模型生成 8+ 维数值序列的精度有限 |

### 8.3 演进路线

```
VLA-0 (当前)          →  VLA-0.5             →  VLA-1 (SmolVLA 风格)
最小可行验证           增强版：Constrained     轻量化专用 VLA 架构
                       Decoding + Action     - Action Expert
                       Binning               - Action chunk prediction
                                             - 专用动作 head
```

### 8.4 核心原则

> **100M 规模的 VLM 在 LIBERO 等仿真平台上完全可用，但要合理设计动作块长度和容错机制。先跑通 VLA-0 基线验证数据流程，再用 SmolVLA 架构提升性能，这是最稳妥的路径。**

### 8.5 快速验证清单

- [ ] 数据格式正确转换（离散化 + 字符串拼接）
- [ ] 模型能生成格式正确的动作字符串
- [ ] `robust_parse_action_text` 能正确解析输出
- [ ] 动作反归一化后落在合理范围内
- [ ] 仿真环境中机器人能执行动作
- [ ] 多次采样结果一致性检查
- [ ] 端到端任务成功率评估

---

## 附录 A：完整项目文件清单

```
minimind-v/
├── model/
│   ├── model_minimind.py          # LLM 核心（冻结，无需修改）
│   ├── model_vlm.py               # VLM 包装（冻结，无需修改）
│   └── vision_model/              # CLIP Vision Encoder（冻结，无需修改）
├── dataset/
│   ├── lm_dataset.py               # VLM 数据集（参考）
│   ├── vla_dataset.py              # VLA 数据集（新增）
│   └── vla_libero.parquet          # VLA 训练数据（需生成）
├── trainer/
│   ├── trainer_utils.py           # 训练工具（通用）
│   ├── train_sft_vlm.py           # VLM SFT（参考）
│   └── train_vla.py               # VLA 训练（新增）
├── scripts/
│   ├── convert_vla_dataset.py     # 数据转换脚本（新增）
│   ├── web_demo_vlm.py            # VLM WebUI（参考）
│   └── web_demo_vla.py            # VLA WebUI（新增，待实现）
├── eval_vlm.py                    # VLM 评测（参考）
├── eval_vla.py                    # VLA 评测（新增）
└── docs/
    └── VLA.md                     # 本文档
```
