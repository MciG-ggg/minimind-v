"""
MiniMind-V SFT 训练脚本 (HuggingFace Trainer 版)
数据集格式: input_text (str), action_text (str), images (PIL.Image)
训练方式: 只对 action 部分计算损失，输入部分 labels 设为 -100
"""

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn
torch.backends.cudnn.enabled = False  # 避免 cuDNN 自动调优导致的内部错误
from PIL import Image as PILImage
from datasets import load_from_disk
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.data.data_collator import DataCollatorWithPadding

# 项目根目录加入 sys.path（使 vla_scripts 成为可导入的包）
sys.path.insert(0, str(Path(__file__).parent.parent))
from vla_scripts.rich_helpers import ok, warn, info, section, kv_table, console

import tyro

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from model_vla import load_vla_model

# 【方案 A】训练和推理均不使用 SYSTEM_PROMPT，避免引入未见过的 token
SYSTEM_PROMPT = ""

# ==============================================================================
# DataCollator
# ==============================================================================

@dataclass
class VLADataCollator(DataCollatorWithPadding):
    """
    批量处理 VLA 样本，输入部分 loss mask 为 -100，只训练 action。
    使用 MiniMind tokenizer（与模型 config.bos_token_id=1, eos_token_id=2, vocab_size=6400 一致），
    替代 CLIP tokenizer（bos=49406, eos=49407, vocab=49408）。
    序列结构：[BOS=1, image_id=34×196, text_tokens, EOS=2]

    【双图像支持】images 字段为 list of 2 PIL.Image（主视角 + 腕部视角），
    分别 resize+normalize 后传入模型，由 ImageFusionModule 融合。
    """
    processor: Any = None
    tokenizer: Any = None
    num_image_tokens: int = 196  # VLM 模型期望的图像 token 数量（image_ids 长度）
    system_prompt: str = SYSTEM_PROMPT
    # MiniMind tokenizer 的特殊 token（与 config.json 一致）
    _bos_id: int = 1
    _eos_id: int = 2
    _pad_id: int = 0
    _image_id: int = 34  # model.params.image_ids[0]

    def __init__(self, processor=None, tokenizer=None, num_image_tokens=196, system_prompt=SYSTEM_PROMPT, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.processor = processor
        self._tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens
        self.system_prompt = system_prompt

    def _preprocess_image(self, img: "PIL.Image.Image") -> torch.Tensor:
        """单张 PIL.Image → resize → normalize → [3, 224, 224]"""
        import numpy as np
        img = img.resize((224, 224), PILImage.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.481, 0.458, 0.408], dtype=np.float32)
        std = np.array([0.269, 0.261, 0.276], dtype=np.float32)
        arr = (arr - mean) / std
        return torch.from_numpy(arr).permute(2, 0, 1)  # [3, 224, 224]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        import numpy as np

        input_texts = [f["input_text"] for f in features]
        action_texts = [f["action_text"] for f in features]

        # --- 【双图像支持】images 是 list of 2 PIL.Image（主视角 + 腕部视角）---
        front_imgs = []
        wrist_imgs = []
        for feat in features:
            img_list = feat["images"]  # [front_PIL, wrist_PIL]
            front_imgs.append(self._preprocess_image(img_list[0]))   # [3, 224, 224]
            wrist_imgs.append(self._preprocess_image(img_list[1]))   # [3, 224, 224]
        front_stack = torch.stack(front_imgs, dim=0)   # [B, 3, 224, 224]
        wrist_stack = torch.stack(wrist_imgs, dim=0)   # [B, 3, 224, 224]
        # 始终输出 [B, 2, 3, H, W]，由模型根据 dual_image 参数决定：
        # - dual_image=True:  用 ImageFusionModule 融合
        # - dual_image=False: 在模型内部平均两个图像 embedding
        pixel_values = torch.stack([front_stack, wrist_stack], dim=1)  # [B, 2, 3, 224, 224]

        # --- 文本 tokenize: MiniMind tokenizer ---
        image_ids_batch = torch.full(
            (len(features), self.num_image_tokens),
            self._image_id,
            dtype=torch.long,
        )

        text_with_prompt = [f"{self.system_prompt}{t}" for t in input_texts]
        text_ids = self._tokenizer(
            text_with_prompt,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )["input_ids"]

        eos_batch = torch.full((len(features), 1), self._eos_id, dtype=torch.long)
        bos_batch = torch.full((len(features), 1), self._bos_id, dtype=torch.long)

        # input_ids: [BOS, image_ids, text_tokens, EOS]
        input_ids = torch.cat([bos_batch, image_ids_batch, text_ids, eos_batch], dim=1)
        attention_mask = torch.ones_like(input_ids)

        # --- Action tokenize ---
        action_ids = self._tokenizer(
            action_texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )["input_ids"]

        # --- 拼接 input + action，构建 labels ---
        full_input_ids = torch.cat([input_ids, action_ids], dim=1)
        full_attention_mask = torch.cat([attention_mask, torch.ones_like(action_ids)], dim=1)

        labels = torch.full_like(full_input_ids, -100)
        input_len = input_ids.size(1)
        for i in range(action_ids.size(0)):
            action_len = (action_ids[i] != self._pad_id).sum().item()
            labels[i, input_len:input_len + action_len] = action_ids[i, :action_len]

        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }


# ==============================================================================
# tyro CLI
# ==============================================================================

@dataclass
class SFTConfig:
    # 本地 VLA checkpoint 路径 或 HuggingFace model_id
    # 首次训练（SFT 预热）→ 填 base VLM（如 "jingyaogong/minimind2-v"）
    # 继续微调（基于已有 VLA）→ 填训练输出的 checkpoint（如 "./checkpoints/final"）
    model_path_or_id: str = "jingyaogong/minimind2-v"
    dual_image: bool = True   # 启用双视角图像融合（主视角 + 腕部视角）
    # 数据集路径：对齐 eval_libero.py 评测的 LIBERO_OBJECT benchmark（task_index 10-19）
    # 使用 prepare_data.py 的 task_index="10-19" 生成
    dataset_path: str = "./vla0_libero_object_train"
    output_dir: str = "./checkpoints"
    epochs: int = 10         # 方案 A 新格式：增加到 10 epochs，loss 仍在下降
    batch_size: int = 4
    lr_llm: float = 5e-6      # LLM 学习率（预训练权重，需较小 LR 保持泛化）
    lr_vision: float = 5e-5   # VisionProj 学习率（少量参数，可适当放大）
    lr_fusion: float = 2e-4   # ImageFusionModule 学习率（加大，加速收敛）
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0  # 梯度裁剪，防止 NaN
    logging_steps: int = 10
    save_steps: int = 200        # 缩短保存间隔，便于观察训练进度
    save_total_limit: int = 3
    seed: int = 42
    bf16: bool = True  # bf16 动态范围更大（exp 8bits vs fp16 5bits），优先使用
    resume: str = ""
    device: str = "cuda"  # 设备类型，可选: "cuda", "cuda:0", "cuda:1", "cpu" 等


# ==============================================================================
# 主入口
# ==============================================================================

def main(cfg: SFTConfig) -> None:
    # GPU 设备选择：如果 cfg.device 明确指定了 cuda:N，则设置 CUDA_VISIBLE_DEVICES
    # 否则保持用户通过环境变量传入的 CUDA_VISIBLE_DEVICES 不变
    if cfg.device.startswith("cuda") and ":" in cfg.device:
        gpu_id = cfg.device.split(":")[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        # load_vla_model 只需基础设备名，ordinal 由 CUDA_VISIBLE_DEVICES 控制
        model_device = "cuda"
    else:
        model_device = cfg.device

    # 数据集
    section("加载数据集", "cyan")
    dataset = load_from_disk(cfg.dataset_path)
    ok(f"样本数: {len(dataset):,}  |  字段: {dataset.column_names}")
    console.print()

    # 模型 & processor（统一通过 load_vla_model 加载，与 VLAModel 推理保持一致）
    section("加载模型", "cyan")
    print(f"[DEBUG] dual_image={cfg.dual_image}, fusion_type=gated_sum")
    loaded = load_vla_model(cfg.model_path_or_id, device=model_device, dual_image=cfg.dual_image)
    model = loaded["model"]

    # 【修复 NaN】VisionProj (768→512 Linear) 使用 PyTorch 默认初始化，
    # 权重范围 ±0.036 可能导致 fp16/bf16 前向溢出。在模型加载后显式 Xavier 初始化。
    for name, param in model.named_parameters():
        if "vision_proj" in name:
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif param.dim() == 1:
                nn.init.zeros_(param)  # bias 用零初始化，数值更稳定
            print(f"[Init] VisionProj {name}: shape={param.shape}, range=[{param.min():.4f},{param.max():.4f}]")

    # 【NaN 诊断】注册 forward hooks，定位 NaN 产生的层
    _nan_hook_count = [0]
    def _make_hook(tag):
        def hook(module, input, output):
            if _nan_hook_count[0] >= 10:
                return
            _nan_hook_count[0] += 1
            o = output[0] if isinstance(output, tuple) else output
            if o is not None and torch.isnan(o).any():
                print(f"[NaN@Hook {tag}] shape={getattr(o, 'shape', '?')}, NaN={torch.isnan(o).sum().item()}/{o.numel()}")
        return hook

    # 注册到关键层
    model.model.embed_tokens.register_forward_hook(_make_hook("embed_tokens"))
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(_make_hook(f"layer{i}"))
    model.model.norm.register_forward_hook(_make_hook("final_norm"))
    print("[Hooks] NaN 诊断 hooks 已注册（静默模式：无 NaN 不输出）")

    processor = loaded["processor"]
    num_image_tokens = len(loaded["image_ids"])

    # 加载 MiniMind tokenizer（与模型 config.vocab_size=6400, bos=1, eos=2 一致）
    # 注意：processor.tokenizer 是 CLIP tokenizer（vocab=49408），不能用！
    from transformers import PreTrainedTokenizerFast
    minimind_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(__file__).parent.parent / "model" / "tokenizer.json")
    )
    minimind_tokenizer.bos_token = "<|im_start|>"
    minimind_tokenizer.eos_token = "<|im_end|>"
    minimind_tokenizer.pad_token = "<|endoftext|>"
    minimind_tokenizer.bos_token_id = 1
    minimind_tokenizer.eos_token_id = 2
    minimind_tokenizer.pad_token_id = 0

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    kv_table("模型概览", [
        ("model_path_or_id", cfg.model_path_or_id),
        ("dual_image", str(cfg.dual_image)),
        ("总参数量", f"{num_params:.1f}M"),
        ("可训练参数", f"{trainable:.1f}M"),
        ("processor", processor.__class__.__name__),
    ])
    console.print()

    # TrainingArguments
    section("训练配置", "cyan")
    kv_table("超参数", [
        ("epochs", str(cfg.epochs)),
        ("batch_size", str(cfg.batch_size)),
        ("lr_llm", str(cfg.lr_llm)),
        ("lr_vision", str(cfg.lr_vision)),
        ("lr_fusion", str(cfg.lr_fusion)),
        ("weight_decay", str(cfg.weight_decay)),
        ("warmup_ratio", str(cfg.warmup_ratio)),
        ("max_grad_norm", str(cfg.max_grad_norm)),
        ("save_steps", str(cfg.save_steps)),
        ("precision", "bf16" if cfg.bf16 else "fp16"),
    ])
    console.print()

    # bf16 / fp16 二选一，互斥；bf16 回退到 fp16（如果当前 GPU 不支持 bf16）
    if cfg.bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        warn("当前 GPU 不支持 bf16，自动回退到 fp16")
        cfg.bf16 = False
    amp_args = {"bf16": cfg.bf16} if cfg.bf16 else {"fp16": True}

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.lr_llm,  # optimizer_groups 由 create_optimizer 覆盖，这里只作为默认值
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,  # 梯度裁剪，防止 NaN
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to=["none"],
        optim="adamw_torch",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        save_safetensors=False,  # 模型有权重重绑，不支持 safetensors
        disable_tqdm=True,       # 用自定义 tqdm，进度条更简洁
        **amp_args,
    )

    # 用 TrainerCallback 实现 tqdm 进度条（只在 logging 时更新）
    class _TqdmCallback(TrainerCallback):
        def __init__(self, pbar):
            self.pbar = pbar

        def on_log(self, args, state, control, model=None, logs=None, **kwargs):
            # 推进进度 + 原地更新 postfix（不换行）
            self.pbar.update(args.logging_steps)
            if logs:
                loss = logs.get("loss")
                lr = logs.get("learning_rate")
                postfix = {}
                if loss is not None:
                    postfix["loss"] = f"{loss:.4f}"
                if lr is not None:
                    postfix["lr"] = f"{lr:.2e}"
                self.pbar.set_postfix(postfix, refresh=True)

        def on_train_end(self, args, state, control, **kwargs):
            # 训练结束，推进 pbar 到 100%
            self.pbar.update(self.pbar.total - self.pbar.n)
            self.pbar.close()

    # 自定义 Trainer：手动 compute_loss + 静默内置 logging（用 tqdm 展示）
    class VLATrainer(Trainer):
        def __init__(self, *args, pbar=None, lr_llm=5e-6, lr_vision=5e-5, lr_fusion=2e-4, **kwargs):
            super().__init__(*args, **kwargs)
            self._pbar = pbar
            self._lr_llm = lr_llm
            self._lr_vision = lr_vision
            self._lr_fusion = lr_fusion
            self._debug_count = 0

        def create_optimizer(self):
            """Per-parameter-group LR：fusion > vision_proj > LLM"""
            if self.optimizer is not None:
                return self.optimizer

            opt_model = self.model
            # 按模块名分组
            fusion_params = []
            vision_params = []
            llm_params = []
            for name, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue
                if "image_fusion" in name:
                    fusion_params.append(p)
                elif "vision_proj" in name:
                    vision_params.append(p)
                else:
                    llm_params.append(p)

            optimizer_grouped_parameters = [
                {"params": fusion_params, "lr": self._lr_fusion},
                {"params": vision_params, "lr": self._lr_vision},
                {"params": llm_params, "lr": self._lr_llm},
            ]
            # 过滤空组
            optimizer_grouped_parameters = [
                g for g in optimizer_grouped_parameters if g["params"]
            ]
            assert len(optimizer_grouped_parameters) > 0, (
                "所有参数组都为空！检查模型是否正确加载且包含可训练参数。"
            )

            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
            return self.optimizer

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            labels = inputs.pop("labels")
            # 【NaN Debug】只在异常时打印输入信息
            pv = inputs.get("pixel_values")
            if pv is not None and self._debug_count < 3 and (torch.isnan(pv).any() or torch.isinf(pv).any()):
                self._debug_count += 1
                print(f"  [INPUT DEBUG #{self._debug_count}] pixel_values: shape={pv.shape}, dtype={pv.dtype}, NaN={torch.isnan(pv).any()}, range=[{pv.min():.4f},{pv.max():.4f}]")
                print(f"  [INPUT DEBUG #{self._debug_count}] input_ids: shape={inputs['input_ids'].shape}, dtype={inputs['input_ids'].dtype}")
                print(f"  [INPUT DEBUG #{self._debug_count}] attention_mask: shape={inputs['attention_mask'].shape}")
                print(f"  [INPUT DEBUG #{self._debug_count}] labels: NaN={torch.isnan(labels).any()}, -100 ratio={(labels==-100).float().mean():.4f}")
            outputs = model(**inputs, use_cache=False)
            logits = outputs.logits

            # 【NaN Debug】逐层检测
            if torch.isnan(logits).any():
                nan_mask = torch.isnan(logits)
                print(f"WARNING: logits NaN at {nan_mask.sum().item()} positions, total={logits.numel()}")
                if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                    hs = outputs.hidden_states[-1]
                    print(f"  hidden_states NaN: {torch.isnan(hs).any()}, stats: min={hs.min():.4f} max={hs.max():.4f}")
                return logits.sum() * 0.0

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if torch.isnan(shift_labels).any() or (shift_labels == -100).all():
                print("WARNING: labels invalid")
                return logits.sum() * 0.0

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            if torch.isnan(loss):
                print(f"WARNING: loss NaN! logits stats: min={logits.min():.4f} max={logits.max():.4f}")
                return logits.sum() * 0.0

            # 【梯度 NaN 追踪】反向传播一次，检测哪层梯度爆炸
            if loss.item() == 0.0 and self.state.global_step < 5:
                dummy_loss = loss
                model.zero_grad()
                dummy_loss.backward()
                for name, p in model.named_parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        print(f"  GRAD_NAN in: {name}, grad_stats: min={p.grad.nanmin():.4f} max={p.grad.nanmax():.4f}")
                model.zero_grad()

            return (loss, outputs) if return_outputs else loss

        def _inner_training_loop(self, *args, **kwargs):
            # 训练前：静默 transformers logger，防止直接 print dict 等日志
            import logging
            tf_logger = logging.getLogger("transformers.trainer")
            self._old_log_level = tf_logger.level
            tf_logger.setLevel(logging.CRITICAL + 1)
            for h in tf_logger.handlers[:]:
                tf_logger.removeHandler(h)
            # 禁用 Trainer 内置的 tqdm
            self.args.disable_tqdm = True
            try:
                return super()._inner_training_loop(*args, **kwargs)
            finally:
                # 确保异常时 logger 也恢复
                tf_logger.setLevel(self._old_log_level)

        def log(self, logs, batch_size=None, num_examples=None):
            # 静默内置的 log dict 打印，但必须保留 callback 触发（tqdm 需要 on_log）
            if self.state.epoch is not None:
                logs["epoch"] = self.state.epoch
            if self.args.include_num_input_tokens_seen != "no":
                logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            output = {**logs, **{"step": self.state.global_step}}
            self.state.log_history.append(output)
            # 触发 on_log callbacks（tqdm 进度条在此更新），但不解包给 logger
            self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    # 训练前估算总步数
    steps_per_epoch = math.ceil(len(dataset) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs

    # 先创建 tqdm，再传给 trainer
    pbar = tqdm(total=total_steps, desc="Training", unit="step", dynamic_ncols=True, mininterval=0.5)

    trainer = VLATrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=VLADataCollator(
            processor=processor,
            tokenizer=minimind_tokenizer,
            num_image_tokens=num_image_tokens,
            system_prompt=SYSTEM_PROMPT,
        ),
        pbar=pbar,
        lr_llm=cfg.lr_llm,
        lr_vision=cfg.lr_vision,
        lr_fusion=cfg.lr_fusion,
    )
    trainer.add_callback(_TqdmCallback(pbar))

    # 训练
    section("开始训练", "cyan")
    info(f"{cfg.epochs} epochs · {steps_per_epoch} steps/epoch · batch_size={cfg.batch_size} · lr_llm={cfg.lr_llm} lr_vision={cfg.lr_vision} lr_fusion={cfg.lr_fusion}")
    console.print()

    trainer.train(resume_from_checkpoint=cfg.resume if cfg.resume else None)
    ok("训练完成")

    # 保存
    final_path = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_path)
    ok(f"模型已保存: {final_path}")


if __name__ == "__main__":
    tyro.extras.set_accent_color("cyan")
    main(tyro.cli(SFTConfig))
