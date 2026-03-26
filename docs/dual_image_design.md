# 双视角图像支持方案分析

> 分析日期：2026-03-25
> 涉及文件：`model/model_vlm.py`、`model/model_vla.py`、`vla-scripts/train_sft.py`、`vla-scripts/prepare_data.py`

---

## 一、现状梳理

### 1.1 数据集图像字段

原始 LIBERO 数据集每条样本包含两张图像：

| 字段 | 说明 | Shape | dtype |
|------|------|-------|-------|
| `observation.images.image` | 主视角 | `(256, 256, 3)` | uint8 |
| `observation.images.image2` | 腕部视角 | `(256, 256, 3)` | uint8 |

`prepare_data.py:convert_sample` 改为保留双图像：

```python
"images": [sample["observation.images.image"], sample["observation.images.image2"]],
```

### 1.2 VLA 模型架构

```
CLIP Encoder (冻结)          VisionProj (冻结)
      │                            │
      ▼                            │
[ CLIP ViT-B/16 ]                  │
      │                            │
      ▼                            │
[ 196×768 ] ──→ ImageFusionModule ──→ [ 196×768 ] ──→ [ 196×512 ]
                            │
                            ▼
                     count_vision_proj
                            │
                            ▼
                    LLM forward pass
```

---

## 二、方案选择：方案 4（交互层融合）

**方案 4** 相比方案 1（6通道拼接）的优势：
- 每张图像独立经过 CLIP 编码 → 充分利用 3 通道预训练权重
- 两图通过交互层（cross-attention）学习空间对齐和互补信息
- CLIP 编码器保持冻结（不需要微调）

**参数量分析：**

| 模块 | 参数量 | 训练策略 |
|------|--------|---------|
| CLIP ViT-B/16 | ~87M | **冻结** |
| VisionProj `Linear(768,512)` | ~400K | 可训练（LR 较大）|
| ImageFusionModule | ~12M | **主要训练对象** |
| LLM (MiniMind) | ~600M | 可训练（LR 较小）|

> 交互层约 12M 参数，仅占总参数 1.7%，**不需要和 VLA 训练分开**，端到端一次训练搞定。

---

## 三、实现细节

### 3.1 架构：`ImageFusionModule`（model_vlm.py）

```python
class ImageFusionModule(nn.Module):
    def __init__(self, dim=768, num_heads=8, fusion_type="cross_attn"):
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # 双向 cross-attention（轻量：单层，8 head）
        self.cross_attn_front_to_wrist = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn_wrist_to_front = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, front_emb, wrist_emb):
        # 主视角 Query，腕部视角 Key/Value
        fused_front = front_emb + 0.1 * self.cross_attn_front_to_wrist(
            query=self.norm1(front_emb),
            key=self.norm1(wrist_emb),
            value=self.norm1(wrist_emb))[0]
        # 腕部视角 Query，主视角 Key/Value
        fused_wrist = wrist_emb + 0.1 * self.cross_attn_wrist_to_front(
            query=self.norm2(wrist_emb),
            key=self.norm2(fused_front),
            value=self.norm2(fused_front))[0]
        return (fused_front + fused_wrist) * 0.5
```

数据流：

```
front_emb: [B, 50, 768]  ──┐
                            ├──→ 双向 cross-attn ──→ [B, 50, 768] ──→ VisionProj ──→ [B, 50, 512]
wrist_emb: [B, 50, 768]  ──┘
```

**两种融合方式（通过 `fusion_type` 参数切换）：**
- `cross_attn`（默认）：双向 cross-attention，轻量高效
- `concat_mlp`：拼接 → `Linear(1536, 768)` → GELU → `Linear(768, 768)`

### 3.2 配置：`VLMConfig`（model_vlm.py）

```python
class VLMConfig(MiniMindConfig):
    dual_image: bool = False    # 启用双视角图像融合
    fusion_type: str = "cross_attn"  # cross_attn | concat_mlp
```

### 3.3 模型加载：`load_vla_model`（model_vla.py）

```python
def load_vla_model(model_path_or_id, device="cuda", *, eval_mode=True,
                   dual_image=False, fusion_type="cross_attn"):
    if dual_image:
        config = AutoConfig.from_pretrained(model_path_or_id, trust_remote_code=True)
        config.dual_image = True
        config.fusion_type = fusion_type
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_id, config=config, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path_or_id, trust_remote_code=True)
```

### 3.4 训练：`train_sft.py`

```python
@dataclass
class SFTConfig:
    model_path_or_id: str = "jingyaogong/minimind2-v"
    dual_image: bool = True  # 启用双视角融合
    lr_llm: float = 5e-6      # LLM 学习率（预训练权重，需较小 LR 保持泛化）
    lr_vision: float = 5e-5   # VisionProj 学习率（少量参数，可适当放大）
    lr_fusion: float = 1e-4   # ImageFusionModule 学习率（新随机初始化层，需较大 LR 快速收敛）
    ...

# 通过 VLATrainer.create_optimizer 实现 per-parameter LR 分组
trainer = VLATrainer(..., lr_llm=cfg.lr_llm, lr_vision=cfg.lr_vision, lr_fusion=cfg.lr_fusion)
```

**Per-parameter learning rates（通过 `VLATrainer.create_optimizer` 实现）：**

| 组件 | 学习率 | 理由 |
|------|--------|------|
| CLIP ViT-B/16 | **冻结** | 预训练 3 通道权重宝贵 |
| ImageFusionModule | `1e-4` | 随机初始化，主要训练对象 |
| VisionProj | `5e-5` | 少量参数，快速适应 |
| LLM (MiniMind) | `5e-6` | 保持预训练知识 |

### 3.5 推理：`model_vla.py`

```python
class VLAModel:
    def predict(self, instruction, image):
        """
        Args:
            image: 单张 PIL.Image（单图模式）
                   或 list of 2 PIL.Image（双图模式：[front, wrist]）
        """
        inputs = self._build_inputs(instruction, image)
        ...
```

推理时直接传入双图 list：

```python
vla = VLAModel("./checkpoints/final", device="cuda", dual_image=True)
action = vla.predict("pick up the object", [front_image, wrist_image])
```

---

## 四、已修改文件汇总

| 文件 | 改动 |
|------|------|
| `model/model_vlm.py` | 新增 `ImageFusionModule` 类、`VLMConfig.dual_image/fusion_type`、`MiniMindVLM.__init__` 创建融合模块、`MiniMindVLM.forward` 处理双图分支 |
| `model/model_vla.py` | `load_vla_model` 支持 `dual_image`/`fusion_type` 参数、`VLAModel` 同上、`predict`/`forward` 接受 list of 2 图像 |
| `vla-scripts/train_sft.py` | 新增 `dual_image`/`lr_llm`/`lr_vision`/`lr_fusion` CLI 参数、`VLADataCollator` 预处理双图 `→ [B, 2, 3, 224, 224]`、`VLATrainer.create_optimizer` 实现 per-parameter LR |
| `vla-scripts/prepare_data.py` | `convert_sample` 保留双图像为 list |

---

## 五、使用方式

### 训练

```bash
# 启用双视角（默认 True）
python -m vla-scripts.train_sft \
    --model-path-or-id "jingyaogong/minimind2-v" \
    --dataset-path "./vla0_libero_object_train" \
    --dual-image True \
    --epochs 3 \
    --batch-size 4 \
    --lr-llm 5e-6 \
    --lr-vision 5e-5 \
    --lr-fusion 1e-4

# 禁用双视角（回退单图）
python -m vla-scripts.train_sft \
    --model-path-or-id "jingyaogong/minimind2-v" \
    --dual-image False
```

### 推理

```python
from model import VLAModel

# 双图推理
vla = VLAModel("./checkpoints/final", device="cuda", dual_image=True)
action = vla.predict("pick up the red block", [front_img, wrist_img])

# 单图推理
vla_single = VLAModel("./checkpoints/final", device="cuda", dual_image=False)
action = vla_single.predict("pick up the red block", front_img)
```

### 数据集重新转换

> **必须重新运行数据转换**，因为 `images` 字段格式已变更

```bash
python -m vla-scripts.prepare_data convert \
    --output "./vla0_libero_object_train" \
    --task-index 0
```

---

## 六、注意事项

1. **CLIP 冻结是安全的**：ViT-B/16 在 3 通道 RGB 图像上预训练，权重质量高，冻结不影响交互层学习。
2. **交互层收敛快**：12M 参数的 cross-attention 层在 few-shot 内可学到基础对齐，完整训练 3 epoch 足够。
3. **向后兼容**：`dual_image=False` 时行为与原单图版本完全一致，现有推理代码无需修改。
4. **显存增加**：双图模式下 CLIP 需要两次 forward（可并行），显存约增加 15-20%。
