import os
import torch
import warnings
from .model_minimind import *
from typing import Optional, Tuple, List, Union
from torch import nn
from transformers import CLIPProcessor, CLIPModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

warnings.filterwarnings('ignore')


class VLMConfig(MiniMindConfig):
    model_type = "minimind-v"

    def __init__(
            self,
            image_special_token: str = '@' * 196,
            image_ids: List = [34] * 196,
            dual_image: bool = False,  # 双视角图像融合
            fusion_type: str = "gated_sum",  # gated_sum | concat_mlp（cross_attn 数值不稳定已废弃）
            **kwargs,
    ):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.dual_image = dual_image
        self.fusion_type = fusion_type
        super().__init__(**kwargs)


class ImageFusionModule(nn.Module):
    """
    双视角图像交互层（方案 4）

    输入: 两路 CLIP embeddings, 各 [B, 50, 768]
    输出: 融合后的单路 embedding [B, 50, 768]

    使用门控加权融合（数值稳定，替代有数值溢出风险的 cross-attention）：
    front × α + wrist × (1 - α)，α 由可学习门控网络预测
    """
    def __init__(self, dim=768, fusion_type="gated_sum"):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == "gated_sum":
            # 可学习门控：输入拼接 → Sigmoid → 标量权重
            # front × gate + wrist × (1 - gate)，gate ∈ [0, 1]
            self.gate_net = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Linear(dim, 1),
            )
        elif fusion_type == "concat_mlp":
            self.concat_mlp = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, front_emb: torch.Tensor, wrist_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            front_emb: [B, 50, 768] 主视角 CLIP embedding
            wrist_emb: [B, 50, 768] 腕部视角 CLIP embedding
        Returns:
            fused: [B, 50, 768] 融合后的 embedding
        """
        if self.fusion_type == "gated_sum":
            # 对每个 token 位置预测一个 [0,1] 门控权重
            concat = torch.cat([front_emb, wrist_emb], dim=-1)          # [B, 50, 1536]
            gate = torch.sigmoid(self.gate_net(concat))                 # [B, 50, 1]
            fused = gate * front_emb + (1 - gate) * wrist_emb            # [B, 50, 768]
            return fused

        elif self.fusion_type == "concat_mlp":
            concat = torch.cat([front_emb, wrist_emb], dim=-1)          # [B, 50, 1536]
            return self.concat_mlp(concat)                               # [B, 50, 768]


class VisionProj(nn.Module):
    def __init__(self, ve_hidden_size=768, hidden_size=512):
        super().__init__()
        self.ve_hidden_size = ve_hidden_size
        self.hidden_size = hidden_size
        self.vision_proj = nn.Sequential(
            nn.Linear(self.ve_hidden_size, self.hidden_size)
        )

    def forward(self, image_encoders):
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj


# 继承自语言模型
class MiniMindVLM(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model_path=None):
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params
        # 【修复】优先使用 config.vision_model_path（绝对路径），否则使用默认相对路径
        resolved_path = getattr(params, 'vision_model_path', None) or vision_model_path or "./model/vision_model/clip-vit-base-patch16"
        self.vision_encoder, self.processor = self.__class__.get_vision_model(resolved_path)
        self.vision_proj = VisionProj(hidden_size=params.hidden_size)
        # 【双图像支持】交互层（双视角融合）
        if params.dual_image:
            self.image_fusion = ImageFusionModule(
                dim=768,
                fusion_type=getattr(params, "fusion_type", "gated_sum"),
            )
        else:
            self.image_fusion = None

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        # 【关键修复】先创建模型结构，再显式加载权重，避免 transformers 懒加载导致 meta tensor
        clip_config = CLIPModel.config_class.from_pretrained(model_path)
        model = CLIPModel(clip_config)
        # 强制直接赋值到模型参数（解决 transformers 懒加载导致 meta tensor 的问题）
        sd_full = torch.load(os.path.join(model_path, "pytorch_model.bin"),
                              map_location="cpu", weights_only=True)
        sd_vision = {k: v.float() for k, v in sd_full.items() if k.startswith("vision_model.") and not k.endswith("_ids")}
        for name, param in model.named_parameters():
            if name in sd_vision:
                param.requires_grad_(False)
                param.copy_(sd_vision[name])
        processor = CLIPProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_encoder):
        """
        Extract image embeddings from CLIP vision encoder.

        Args:
            image_tensors: [B, 3, H, W] tensor
            vision_encoder: CLIPModel instance
        Returns:
            img_embedding: [B, 50, 768] (CLS token removed, last hidden state)
        """
        with torch.no_grad():
            # vision_encoder is CLIPModel, .vision_model is CLIPVisionTransformer
            outputs = vision_encoder.vision_model(pixel_values=image_tensors)
        # Remove CLS token ([:, 1:, :]), keep batch dimension
        img_embedding = outputs.last_hidden_state[:, 1:, :]
        return img_embedding

    @staticmethod
    def _find_image_indices(tokens, image_ids):
        """查找 input_ids 中所有连续的 image_id 序列（返回 (start, end) 列表）"""
        image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
        len_image_ids = len(image_ids)
        if len_image_ids > tokens.size(1):
            return None
        tokens_view = tokens.unfold(1, len_image_ids, 1)
        matches = (tokens_view == image_ids_tensor).all(dim=2)
        return {
            batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                        matches[batch_idx].nonzero(as_tuple=True)[0]]
            for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
        } or None

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        image_indices = self._find_image_indices(tokens, self.params.image_ids)
        if vision_tensors is not None and image_indices:
            vision_proj = self.vision_proj(vision_tensors)
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)
            new_h = []
            for i in range(h.size(0)):
                if i in image_indices:
                    h_i = h[i]
                    img_idx = 0
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):
                            h_i = torch.cat((h_i[:start_idx], vision_proj[i][img_idx], h_i[end_idx + 1:]), dim=0)[
                                  :seqlen]
                            img_idx += 1
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)
        return h

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                labels: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            if len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2)
            bs, num, c, im_h, im_w = pixel_values.shape

            # 【双图像支持】方案 4：双视角交互层融合
            if self.params.dual_image and num == 2:
                # pixel_values: [B, 2, 3, H, W] — 主视角 + 腕部视角
                # 分别通过 CLIP 编码（各输出 [B, 50, 768]）
                emb_list = [
                    MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)
                    for i in range(2)
                ]
                front_emb = emb_list[0]   # [B, 50, 768]
                wrist_emb = emb_list[1]     # [B, 50, 768]
                fused_emb = self.image_fusion(front_emb, wrist_emb)  # [B, 50, 768]
                # VisionProj 是 Linear(768→512)，对每个位置独立投影
                B2, L, D = fused_emb.shape
                vision_proj_out = self.vision_proj(fused_emb.view(-1, D)).view(B2, L, -1)  # [B, 50, 512]
                # 【关键修复】直接用投影后的结果注入到 hidden_states，跳过 count_vision_proj 里的重复 vision_proj
                image_indices = self._find_image_indices(input_ids, self.params.image_ids)
                if image_indices:
                    # 使用 clone() 避免 in-place 修改 CUDA tensor
                    hidden_states = hidden_states.clone()
                    hs_np = hidden_states.detach().cpu().float().numpy()  # 强制转 fp32 CPU
                    proj_np = vision_proj_out.detach().cpu().float().numpy()
                    for i in range(hidden_states.size(0)):
                        if i in image_indices:
                            for start_idx, end_idx in image_indices[i]:
                                proj_len = min(end_idx - start_idx + 1, vision_proj_out.size(1))
                                hs_np[i, start_idx:start_idx + proj_len, :] = proj_np[i, :proj_len, :]
                    hidden_states = torch.from_numpy(hs_np).to(hidden_states.device)
            else:
                # 单图像模式: get_image_embeddings 输出 [B, 196, 768]，stack(dim=1) 后保持 [B, 196, 768]
                vision_tensors = torch.stack([
                    MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)
                    for i in range(num)
                ], dim=1)
                hidden_states = self.count_vision_proj(
                    tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors,
                    seqlen=input_ids.shape[1])

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum([l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=presents, hidden_states=hidden_states)
        return output
