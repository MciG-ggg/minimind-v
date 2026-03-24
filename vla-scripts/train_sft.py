"""
MiniMind-V SFT 训练脚本 (HuggingFace Trainer 版)
数据集格式: input_text (str), action_text (str), images (PIL.Image 或 list[PIL.Image])
训练方式: 只对 action 部分计算损失，输入部分 labels 设为 -100
"""

import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import DataCollatorWithPadding
from datasets import load_from_disk
import tyro


# ==============================================================================
# 1. 自定义 DataCollator（将 collate_fn 逻辑封装为 HF Trainer 可用的形式）
# ==============================================================================

@dataclass
class VLADataCollator(DataCollatorWithPadding):
    """
    批量处理 VLA 样本，统一调用 processor 保留 pixel_values。

    - 输入文本 + 图像 → processor → input_ids
    - 动作文本（不加特殊 token）→ tokenizer → action_ids
    - 拼接 input_ids + action_ids
    - labels: 输入部分=-100，动作部分=action_ids(padding=-100)
    """
    processor: Any = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_texts = [f["input_text"] for f in features]
        action_texts = [f["action_text"] for f in features]
        images = [f["images"] for f in features]

        # --- 处理输入文本 + 图像 ---
        inputs = self.processor(
            text=input_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # --- 处理动作文本（不加特殊 token）---
        action_ids = self.processor.tokenizer(
            action_texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )["input_ids"]

        # --- 拼接 input_ids + action_ids ---
        full_input_ids = torch.cat([input_ids, action_ids], dim=1)

        # --- 构造 attention_mask ---
        action_mask = torch.ones_like(action_ids)
        full_attention_mask = torch.cat([attention_mask, action_mask], dim=1)

        # --- 构造 labels ---
        pad_token_id = self.processor.tokenizer.pad_token_id or 0
        labels = torch.full_like(full_input_ids, -100)

        input_len = input_ids.size(1)
        for i in range(action_ids.size(0)):
            action_len = (action_ids[i] != pad_token_id).sum().item()
            labels[i, input_len:input_len + action_len] = action_ids[i, :action_len]

        # --- 提取 pixel_values ---
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            raise ValueError("processor 输出中缺少 pixel_values，请确认模型支持图像输入")

        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }


# ==============================================================================
# 2. CLI 配置
# ==============================================================================

@dataclass
class SFTConfig:
    # 模型和数据
    model_id: str = "jingyaogong/minimind2-v"
    dataset_path: str = "./vla0_libero_object_train"
    output_dir: str = "./checkpoints"

    # 训练超参数（直接映射到 TrainingArguments）
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    seed: int = 42
    bf16: bool = True

    # 运行环境
    resume: str = ""  # HF Trainer checkpoint 路径


# ==============================================================================
# 3. 主入口
# ==============================================================================

def main(cfg: SFTConfig):
    # --- 加载数据集 ---
    dataset = load_from_disk(cfg.dataset_path)
    print(f"数据集加载完成: {len(dataset)} 条样本, 字段: {dataset.column_names}")

    # --- 加载模型和 processor ---
    print(f"加载模型: {cfg.model_id}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"模型总参数量: {num_params:.2f}M, 可训练: {trainable_params:.2f}M")

    # --- DataCollator ---
    data_collator = VLADataCollator(processor=processor)

    # --- TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        bf16=cfg.bf16,
        report_to=["none"],  # 关闭默认 wandb 等，可改为 ["wandb"]
        optim="adamw_torch",  # 避免 HF 自动选择 LOMO
        remove_unused_columns=False,  # 保留 input_text/action_text/images 供 DataCollator 使用
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        # 注意：VLM 的 compute_metrics 通常在推理阶段单独做，此处不传
    )

    # --- 训练 & 保存 ---
    print(f"\n开始训练: {cfg.epochs} epochs, batch_size={cfg.batch_size}, "
          f"lr={cfg.learning_rate}, warmup_ratio={cfg.warmup_ratio}")
    print("-" * 80)

    trainer.train(resume_from_checkpoint=cfg.resume if cfg.resume else None)

    print("\n训练完成!")

    # --- 最终保存为 HuggingFace 格式 ---
    trainer.save_model(os.path.join(cfg.output_dir, "final"))
    print(f"模型已保存到 {os.path.join(cfg.output_dir, 'final')}")


if __name__ == "__main__":
    cfg = tyro.cli(SFTConfig)
    main(cfg)
