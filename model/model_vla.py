"""
MiniMind-V VLA 推理封装

抽象出 VLAModel 类，封装:
- 模型加载 (checkpoint → MiniMindVLM + CLIPProcessor)
- 输入预处理 (instruction + PIL.Image → input_ids + pixel_values)
- forward / generate
- 动作 token 提取 + 反归一化

用法:
    vla = VLAModel("./checkpoints/final", device="cuda")  # 加载本地 VLA checkpoint
    action = vla.predict("pick up the object", image_pil)  # → torch.Tensor (7,)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM


# ---------------------------------------------------------------------------
# 动作离散化参数（与 prepare_data.py 保持一致）
# ---------------------------------------------------------------------------
ACTION_TOKEN_MIN = 0
ACTION_TOKEN_MAX = 999
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
ACTION_DIM = 7


def denormalize_action(token_ids):
    """离散动作 token → 连续动作，始终返回 2D 数组 (N, ACTION_DIM)"""
    import numpy as np
    tokens = np.array(token_ids).flatten()
    num_actions = len(tokens) // ACTION_DIM
    tokens = tokens[: num_actions * ACTION_DIM]
    normalized = (tokens - ACTION_TOKEN_MIN) / (ACTION_TOKEN_MAX - ACTION_TOKEN_MIN)
    # 始终 reshape 成 (N, 7)，确保上游 actions[0] 索引行为一致
    continuous = (normalized * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW)
    return continuous.reshape(-1, ACTION_DIM)


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "System Prompt. Analyze the input image and predict robot actions "
    "for the next H timesteps. Each action has D dimensions. Output a single "
    "sequence of H × D integers (0 B each), representing the H timesteps "
    "sequentially. Provide only space-separated numbers. Nothing else."
)


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
        # 【修复】vision_model_path 使用绝对路径，避免相对路径在 HuggingFace cache 环境下解析错误
        vision_abs_path = str(Path(__file__).parent / "vision_model" / "clip-vit-base-patch16")
        config.vision_model_path = vision_abs_path
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_id, config=config, trust_remote_code=True
        )
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path_or_id, trust_remote_code=True)
        # 【修复】vision_model_path 使用绝对路径
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

    # =========================================================================
    # 【核心修复】使用正确的 MiniMind Tokenizer（替换 CLIPProcessor 的 tokenizer）
    #
    # 问题：CLIPProcessor 的 tokenizer 有 49408 个 token（BOS=49406, EOS=49407），
    #       产生的 text token ID 可达 37695+，远超模型 vocab_size=6400。
    #       根本原因：模型训练时使用的是 MiniMind tokenizer（vocab=6400），
    #       而推理时错误地使用了 CLIP tokenizer。
    #
    # 方案：加载与训练时一致的 MiniMind tokenizer（来自 model/tokenizer.json）
    #       - vocab_size: 6400
    #       - bos_token_id: 1 (<|im_start|>)
    #       - eos_token_id: 2 (<|im_end|>)
    #       - pad_token_id: 0 (<|endoftext|>)
    #       CLIPProcessor 保留（模型内部 get_vision_model 仍需要它）。
    # =========================================================================
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
        max_new_tokens: int = 64,
        do_sample: bool = False,
        dual_image: bool = False,
    ):
        self.model_path_or_id = model_path_or_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.system_prompt = SYSTEM_PROMPT
        self.dual_image = dual_image

        # 统一通过 load_vla_model 加载
        loaded = load_vla_model(model_path_or_id, device, eval_mode=True, dual_image=dual_image)
        self.model = loaded["model"]
        self.processor = loaded["processor"]
        self.tokenizer = loaded["tokenizer"]
        self.image_ids = loaded["image_ids"]
        self.bos_id = loaded["bos_id"]
        self.eos_id = loaded["eos_id"]
        self.pad_id = loaded["pad_id"]

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

        Args:
            instruction: 文本指令
            image: 单张 PIL.Image（单图模式）或 list of 2 PIL.Image（双图模式：主视角 + 腕部视角）

        Returns:
            torch.Tensor: shape (ACTION_DIM,), dtype float32, range [ACTION_LOW, ACTION_HIGH]
        """
        import numpy as np
        import re

        inputs = self._build_inputs(instruction, image)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                pad_token_id=self.pad_id,
                eos_token_id=self.eos_id,
            )

        # 【修复 BPE Tokenizer 与 Action Token 不匹配问题】
        # 训练格式: action values → normalize → "025x864x020x025x026x713x494"
        #   (3-digit zero-padded + x 分隔，每个 action = 3 digits + 1 separator)
        # 推理时: model 输出 tokenizer vocab token IDs → decode → parse → action token IDs (0-999)
        generated_text = self.tokenizer.decode(outputs[0, input_len:].tolist(), skip_special_tokens=True)
        action_ids = re.findall(r"\d{3}", generated_text)

        if len(action_ids) < ACTION_DIM:
            # 极少情况：用 logits top-k 补齐
            logits = self.model(**inputs).logits[0, -1, :]
            logits_ids = torch.topk(logits, ACTION_DIM).indices.tolist()
            action_ids = action_ids + logits_ids[len(action_ids):]

        actions = denormalize_action(np.array([int(x) for x in action_ids[:ACTION_DIM]]))
        # 防御性：确保 actions 是 2D (N, 7)，防止极端情况下 reshape 后变成 1D
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        action = torch.from_numpy(actions[0]).float()  # 提取第一行 → (7,)
        return action

    def forward(
        self,
        instruction: str,
        image: "Image.Image | list[Image.Image]",
    ) -> torch.Tensor:
        """predict 的别名，保持接口一致性"""
        return self.predict(instruction, image)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _preprocess_image(self, img: Image.Image) -> torch.Tensor:
        """单张 PIL.Image → [3, 224, 224] float32 normalized"""
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
        """构造 LLM 输入: input_ids + attention_mask + pixel_values"""
        text = f"{self.system_prompt}\n\nInstruction: {instruction}\nAction:"

        # 1. 文本 tokenize（不含特殊 token，由我们手动插入 BOS/EOS）
        text_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"][0]

        # 2. 构造完整 input_ids: BOS + image_ids + text_tokens + EOS
        input_ids = torch.cat(
            [
                torch.tensor([[self.bos_id]]),
                torch.tensor([self.image_ids]),
                text_tokens.unsqueeze(0),
                torch.tensor([[self.eos_id]]),
            ],
            dim=1,
        )
        attention_mask = torch.ones_like(input_ids)

        # 3. 图像预处理
        if self.dual_image:
            # 【双图像模式】image 是 list of 2 PIL.Image（主视角 + 腕部视角）
            front, wrist = image[0], image[1]
            front_t = self._preprocess_image(front).unsqueeze(0)    # [1, 3, 224, 224]
            wrist_t = self._preprocess_image(wrist).unsqueeze(0)    # [1, 3, 224, 224]
            pixel_values = torch.stack([front_t, wrist_t], dim=1) # [1, 2, 3, 224, 224]
        else:
            # 【单图像模式】
            pixel_values = self._preprocess_image(image).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 224, 224]

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "pixel_values": pixel_values.to(self.device),
        }

    def _extract_action_tokens(
        self, outputs: torch.Tensor, input_len: int
    ) -> list[int]:
        """
        从 generate 输出中提取动作 token

        输出序列 = [BOS, IMAGE*196, text_tokens, EOS, 生成_tokens...]
        动作 token 从 text_tokens 结束后开始（跳过 input_len）
        """
        skip_ids = {self.pad_id, self.eos_id}
        return [
            t.item()
            for t in outputs[0, input_len:]
            if t.item() not in skip_ids
        ]
