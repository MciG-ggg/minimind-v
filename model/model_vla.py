"""
MiniMind-V VLA 推理封装

用法:
    vla = VLAModel("./checkpoints/final", device="cuda")
    action = vla.predict("pick up the object", image_pil)  # → torch.Tensor (7,)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM


# ---------------------------------------------------------------------------
# 动作离散化参数
# ---------------------------------------------------------------------------
ACTION_TOKEN_MIN = 0
ACTION_TOKEN_MAX = 999
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
ACTION_DIM = 7
BINS_PER_ACTION = 3
# 生成 token 总数（每个 action 值 3 位 + 空格分隔）
MAX_GENERATION_TOKENS = ACTION_DIM * BINS_PER_ACTION + ACTION_DIM - 1
# Fallback bin=500（中间值）用于异常情况
_FALLBACK_DIGITS = [5, 0, 0] * ACTION_DIM


def denormalize_action(token_ids):
    """离散动作 token → 连续动作，始终返回 2D 数组 (N, ACTION_DIM)

    输入格式：纯数字字符列表（ACTION_DIM * BINS_PER_ACTION 个 0-9 整数）。
    将每 BINS_PER_ACTION=3 个相邻数字合并为 1 个 bin 值 (0-999)，再反归一化。
    """
    import numpy as np
    tokens = np.array(token_ids).flatten()

    if len(tokens) == 0:
        tokens = np.zeros(ACTION_DIM * BINS_PER_ACTION, dtype=int)

    num_complete_bins = len(tokens) // BINS_PER_ACTION
    remainder = len(tokens) % BINS_PER_ACTION

    bin_values = []
    for i in range(num_complete_bins):
        chunk = tokens[i * BINS_PER_ACTION:(i + 1) * BINS_PER_ACTION]
        bin_val = sum(d * (10 ** (BINS_PER_ACTION - 1 - j)) for j, d in enumerate(chunk))
        bin_values.append(bin_val)

    if remainder > 0:
        tail = tokens[num_complete_bins * BINS_PER_ACTION:]
        pad = [tail[-1]] * (BINS_PER_ACTION - remainder)
        chunk = np.concatenate([tail, pad])
        bin_val = sum(d * (10 ** (BINS_PER_ACTION - 1 - j)) for j, d in enumerate(chunk))
        bin_values.append(bin_val)

    if len(bin_values) == 0:
        bin_values = [0]
    while len(bin_values) < ACTION_DIM:
        bin_values.append(bin_values[-1])

    bin_values = np.array(bin_values[:ACTION_DIM], dtype=np.float64)
    normalized = (bin_values - ACTION_TOKEN_MIN) / (ACTION_TOKEN_MAX - ACTION_TOKEN_MIN)
    continuous = normalized * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW
    return continuous.reshape(-1, ACTION_DIM)


# ---------------------------------------------------------------------------
# VLAModel
# ---------------------------------------------------------------------------

def load_vla_model(
    model_path_or_id: str,
    device: str = "cuda",
    *,
    eval_mode: bool = True,
    dual_image: bool = False,
    fusion_type: str = "gated_sum",
):
    """
    加载 VLA 模型并返回 (model, processor, tokenizer, image_ids, bos_id, eos_id, pad_id)

    Args:
        model_path_or_id: 本地 VLA checkpoint 路径（如 "./checkpoints/final"）
                         或 HuggingFace model_id（如 "jingyaogong/minimind2-v"）
        device: 加载设备
        eval_mode: True 时调用 .to(device).eval()，False 时只 .to(device)
        dual_image: 启用双视角图像融合（主视角 + 腕部视角）
        fusion_type: 融合方式，gated_sum（门控加权，默认）或 concat_mlp

    Returns:
        包含模型和关键 tokenizer/processor 属性的 dict
    """
    if not model_path_or_id or not model_path_or_id.strip():
        raise ValueError("model_path_or_id 不能为空，请传入本地 checkpoint 路径或 HuggingFace model_id")

    if dual_image:
        # 加载 checkpoint 的 config，然后覆盖 dual_image 相关字段
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path_or_id, trust_remote_code=True)
        config.dual_image = True
        config.fusion_type = fusion_type
        vision_abs_path = str(Path(__file__).parent / "vision_model" / "clip-vit-base-patch16")
        config.vision_model_path = vision_abs_path
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_id, config=config, trust_remote_code=True
        )
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path_or_id, trust_remote_code=True)
        vision_abs_path = str(Path(__file__).parent / "vision_model" / "clip-vit-base-patch16")
        config.vision_model_path = vision_abs_path
        model = AutoModelForCausalLM.from_pretrained(model_path_or_id, config=config, trust_remote_code=True)

    if eval_mode:
        model = model.to(device).eval()
    else:
        model = model.to(device)

    processor = getattr(model, "processor", None)
    if processor is None:
        raise RuntimeError(
            f"model_path_or_id {model_path_or_id} 的模型实例不包含 .processor 属性，"
            "可能是非 VLA 模型"
        )

    # 使用 MiniMind tokenizer（vocab=6400）替代 CLIP tokenizer
    _model_dir = Path(model_path_or_id) if Path(model_path_or_id).is_dir() else Path("./checkpoints/final")
    _tokenizer_path = _model_dir / "tokenizer.json"
    if not _tokenizer_path.exists():
        _tokenizer_path = Path(__file__).parent / "tokenizer.json"
    if not _tokenizer_path.exists():
        raise FileNotFoundError(f"MiniMind tokenizer not found at {_tokenizer_path}")

    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(_tokenizer_path))
    tokenizer.bos_token = "<|im_start|>"
    tokenizer.eos_token = "<|im_end|>"          # 按训练配置 (config: eos_id=2)
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.bos_token_id = 1                  # 与 model.config 一致
    tokenizer.eos_token_id = 2                  # 与 model.config 一致
    tokenizer.pad_token_id = 0

    return {
        "model": model,
        "processor": processor,
        "tokenizer": tokenizer,
        "image_ids": model.params.image_ids,  # [34] * 196
        "bos_id": tokenizer.bos_token_id,      # 1
        "eos_id": tokenizer.eos_token_id,      # 2
        "pad_id": tokenizer.pad_token_id,      # 0
    }


class VLAModel:
    """
    MiniMindVLA 推理接口

    MiniMindVLA 使用 CLIP 编码图像，通过 image_ids=[34]*196 注入到 LLM。
    输入序列结构: [BOS] + [IMAGE_ID*196] + [text_tokens] + [EOS]
    """

    def __init__(
        self,
        model_path_or_id: str,
        device: str = "cuda",
        max_new_tokens: int = MAX_GENERATION_TOKENS,
        do_sample: bool = False,
        temperature: float = 0.3,
        dual_image: bool = False,
    ):
        self.model_path_or_id = model_path_or_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.dual_image = dual_image

        loaded = load_vla_model(model_path_or_id, device, eval_mode=True, dual_image=dual_image)
        self.model = loaded["model"]
        self.processor = loaded["processor"]
        self.tokenizer = loaded["tokenizer"]
        self.image_ids = loaded["image_ids"]
        self.bos_id = loaded["bos_id"]
        self.eos_id = loaded["eos_id"]
        self.pad_id = loaded["pad_id"]

        # 预计算常量 tensor，避免 predict 每次重建
        self._bos_tensor = torch.tensor([[self.bos_id]], device=device)
        self._eos_tensor = torch.tensor([[self.eos_id]], device=device)
        self._image_ids_tensor = torch.tensor([self.image_ids], device=device)

    # ------------------------------------------------------------------
    # 核心推理
    # ------------------------------------------------------------------

    def predict(
        self,
        instruction: str,
        image: "Image.Image | list[Image.Image]",
    ) -> torch.Tensor:
        """
        给定指令 + 图像，返回连续动作向量

        【方案 A】训练时用拼接格式 "025864020025026713494"，
        推理时模型自由生成数字文本，regex 提取所有 digit，
        每 3 个相邻 digit 合并为 1 个 bin 值 (0-999)，再反归一化。

        Args:
            instruction: 文本指令
            image: 单张 PIL.Image 或 list of 2 PIL.Image（双图模式）

        Returns:
            torch.Tensor: shape (ACTION_DIM,), dtype float32, range [ACTION_LOW, ACTION_HIGH]
        """
        import numpy as np

        inputs = self._build_inputs(instruction, image)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            pad_token_id=self.pad_id,
            eos_token_id=self.eos_id,
        )
        if self.do_sample and self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.9
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # 【方案 A】从生成文本中提取所有 digit（忽略空格和文字）
        generated_text = self.tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
        all_digits = [int(c) for c in generated_text if c.isdigit()]

        # 防御性：digit 不足 21 个时用 fallback 填充
        if len(all_digits) < ACTION_DIM * BINS_PER_ACTION:
            all_digits = _FALLBACK_DIGITS[:]

        actions = denormalize_action(np.array(all_digits))
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        return torch.from_numpy(actions[0]).float()

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _preprocess_image(self, img: Image.Image) -> torch.Tensor:
        """单张 PIL.Image → [3, 224, 224] float32 (CLIP normalize)"""
        import numpy as np
        img = img.resize((224, 224), Image.BICUBIC)
        if img.mode in ("RGBA", "LA"):
            img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        # 使用 CLIP 预处理器官方参数 (与 preprocessor_config.json 一致)
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        arr = (arr - mean) / std
        return torch.from_numpy(arr).permute(2, 0, 1)  # [3, 224, 224]

    def _build_inputs(
        self,
        instruction: str,
        image: "Image.Image | list[Image.Image]",
    ) -> dict[str, torch.Tensor]:
        text = f"Instruction: {instruction}\nAction:"

        text_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"][0]

        input_ids = torch.cat(
            [
                self._bos_tensor,
                self._image_ids_tensor,
                text_tokens.unsqueeze(0),
                self._eos_tensor,
            ],
            dim=1,
        )
        attention_mask = torch.ones_like(input_ids)

        if self.dual_image:
            front, wrist = image[0], image[1]
            front_t = self._preprocess_image(front).unsqueeze(0)
            wrist_t = self._preprocess_image(wrist).unsqueeze(0)
            pixel_values = torch.stack([front_t, wrist_t], dim=1)
        else:
            pixel_values = self._preprocess_image(image).unsqueeze(0).unsqueeze(0)

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "pixel_values": pixel_values.to(self.device),
        }
