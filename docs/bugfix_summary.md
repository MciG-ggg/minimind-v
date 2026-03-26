# Bug 修复与代码改动总结

本文档记录 MiniMind-V VLA 推理 pipeline 中发现并修复的所有 bug，以及不被 Git 追踪的关键文件。

## 一、问题背景

运行 `python ./vla-scripts/eval_libero.py --device cuda:1` 时遇到 CUDA error：

```
indexSelectLargeIndex: Assertion srcIndex < srcSelectDimSize failed
```

PyTorch CUDA 层将越界索引报告为模糊的 `device-side assert`，掩盖了真实的 Python `IndexError`。使用 `CUDA_LAUNCH_BLOCKING=1` 定位到 `embed_tokens` 层。

---

## 二、发现的所有 Bug

### Bug 1：Tokenizer 与 Model Vocab 不匹配（根因）

| 组件 | Vocab Size | BOS ID | EOS ID |
|------|-----------|--------|--------|
| CLIPProcessor Tokenizer | 49408 | 49406 | 49407 |
| MiniMindVLM Embedding | **6400** | — | — |

输入序列 `[BOS=49406] + [IMAGE_ID×196] + [text_tokens] + [EOS=49407]` 中的 BOS/EOS 远超 embedding table 的 6400 行，导致越界访问。

**修复方案：** 在 `model_vla.py` 的 `load_vla_model()` 中，用 MiniMind Tokenizer（vocab=6400, BOS=1, EOS=2）替换 CLIPProcessor 的 tokenizer。

### Bug 2：`find_indices` 未定义

`model_vlm.py` 第 171 行调用了不存在的全局函数 `find_indices()`，覆盖了第 169 行的正确结果 `_find_image_indices()`。

**修复：** 删除该错误调用。

### Bug 3：`vision_model.vision_model` API 错误

在 transformers 4.57.1 中，`CLIPVisionTransformer` 没有 `.vision_model` 属性。代码错误地调用 `vision_model.vision_model(...)`。

**修复：** 改为 `vision_encoder.vision_model(...)`（vision_encoder 是 CLIPModel）。

### Bug 4：`.squeeze()` 导致 Batch 维度丢失

`get_image_embeddings` 返回 `outputs.last_hidden_state[:, 1:, :].squeeze()`，单图时 squeeze 移除了 batch 维度 `[1, 50, 768]` → `[50, 768]`。

**修复：** 移除 `.squeeze()`，保持 `[B, 50, 768]` 形状。

### Bug 5：`stack_dim` 逻辑错误

单图模式下 `stack_dim = 0` 导致 `torch.stack(..., dim=0)` 产生形状 `[196, 768]` 而不是 `[1, 196, 768]`。

**修复：** 直接使用 `dim=1`。

### Bug 6：归一化参数微小偏差

`model_vla.py` 中的 mean/std 与 CLIP 预处理器配置有微小差异。

**修复：** 对齐为 CLIP 官方参数：
```python
mean = [0.48145466, 0.4578275, 0.40821073]
std  = [0.26862954, 0.26130258, 0.27577711]
```

---

## 三、所有代码改动

### Git 追踪文件（已修改）

| 文件 | 改动说明 |
|------|----------|
| `model/model_vla.py` | **新建** — `VLAModel` 推理封装类 + `load_vla_model()`（替换 CLIP tokenizer 为 MiniMind tokenizer） |
| `model/__init__.py` | 导出 `VLAModel`, `load_vla_model`, `SYSTEM_PROMPT` |
| `model/model_vlm.py` | 修复 `get_image_embeddings`（API + squeeze）、`count_vision_proj`（删除错误 find_indices 调用）、`forward`（stack_dim=1） |
| `vla-scripts/eval_libero.py` | 启用 `dual_image=True`、传入 `[front_img, wrist_img]` 双图 |
| `vla-scripts/train_sft.py` | 对齐归一化参数（CLIP 官方值）、注释更新 |
| `vla-scripts/rich_helpers.py` | **新建** — 终端彩色输出辅助工具 |
| `docs/dual_image_design.md` | **新建** — 双视角融合设计文档 |
| `model/vision_model/clip-vit-base-patch16/` | **新建** — 本地 CLIP ViT-B/16 权重（避免每次从 HuggingFace 下载） |

### HuggingFace 缓存中的 `model_vlm.py`

当 `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` 加载时，HuggingFace 会将自定义模型代码缓存到：

```
~/.cache/huggingface/modules/transformers_modules/<model_id>/model_vlm.py
```

这个缓存副本与 `checkpoints/final/model_vlm.py` 是**独立**的，都会作为实际运行时代码。已确认 HuggingFace 缓存副本与本地文件内容一致。

---

## 四、最关键：`.venv` 下的 Auto-Patch 机制

`.venv/` 在 `.gitignore` 中被排除，不被 Git 追踪。

### 4.1 `sitecustomize.py`（最关键）

路径：`/home/mcig/minimind-v/.venv/lib/python3.13/site-packages/sitecustomize.py`

**原理：** Python 启动时，`site` 模块会自动导入 `sitecustomize.py`（在任何用户代码之前），利用这个机制在 transformers 加载之前修补 HuggingFace 缓存中的 buggy 文件。

**修补的三个 bug：**

1. 删除 `image_indices = find_indices(tokens, self.params.image_ids)`（未定义函数）
2. 修复 `vision_model.vision_model` → `vision_encoder.vision_model`
3. 修复 `dim=stack_dim)` → `dim=1`

```python
# Bug 1
content = content.replace(
    "\n        image_indices = find_indices(tokens, self.params.image_ids)\n",
    "\n"
)

# Bug 2
content = content.replace(
    "outputs = vision_model.vision_model(pixel_values=image_tensors)",
    "outputs = vision_encoder.vision_model(pixel_values=image_tensors)"
)

# Bug 3
content = content.replace(", dim=stack_dim)", ", dim=1)")
```

### 4.2 为什么需要 sitecustomize.py

之前尝试直接修改 `~/.cache/huggingface/modules/transformers_modules/final/model_vlm.py`，但文件似乎被静默恢复。这可能是因为：

- HuggingFace transformers 库在某些情况下会重新下载/验证缓存文件
- 或存在某种 auto-recovery 机制

`sitecustomize.py` 机制确保**每次 Python 启动时**都重新检查并修补，无论缓存文件当前状态如何。

---

## 五、train/eval 配置不匹配问题

| | 训练 (train_sft.py) | 评估 (旧 eval_libero.py) |
|---|---|---|
| `dual_image` | `True` | `False` ❌ |
| 图像输入 | `[front, wrist]` 双图 | 单图 |

**已修复 eval_libero.py：**
```python
vla = VLAModel(checkpoint, device=device, dual_image=True)
front_img = Image.fromarray(obs["agentview_image"])
wrist_img = Image.fromarray(obs["robot0_eye_in_hand_image"])
action_tensor = vla.predict(instruction, [front_img, wrist_img])
```

---

## 六、Git 追踪状态

```bash
# 已修改的文件
model/__init__.py           (modified)
model/model_vlm.py          (modified)
vla-scripts/eval_libero.py  (modified)
vla-scripts/prepare_data.py (modified)
vla-scripts/train_sft.py    (modified)

# 新增的 Git 追踪文件
model/model_vla.py          (new, untracked)
vla-scripts/rich_helpers.py (new, untracked)
docs/dual_image_design.md  (new, untracked)
model/vision_model/clip-vit-base-patch16/ (new, untracked dir)

# .venv 下不被 Git 追踪（已在 .gitignore）
.venv/lib/python3.13/site-packages/sitecustomize.py  ← 关键 auto-patch 文件
```

---

## 七、长期维护建议

1. **`sitecustomize.py` 是必需的文件**：即使所有 `model_vlm.py` 副本都正确修补了，下次换环境或清缓存后仍需要此文件。建议将其纳入版本控制（放到项目根目录或非 `.venv` 位置），而不是依赖 `.venv` 的隐式存在。

2. **同步三个 `model_vlm.py` 位置**：
   - `model/model_vlm.py`（开发源）
   - `checkpoints/final/model_vlm.py`（打包进 checkpoint）
   - `~/.cache/huggingface/modules/transformers_modules/final/model_vlm.py`（运行时实际执行的位置，由 `sitecustomize.py` 自动同步）

3. **`dual_image` 配置**：train 和 eval 的 `dual_image` 必须保持一致。

---

## 八、验证命令

```bash
cd /home/mcig/minimind-v

# 快速推理测试
python -c "
from model.model_vla import VLAModel
from PIL import Image
import numpy as np
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
vla = VLAModel('./checkpoints/final', device='cuda', dual_image=False)
action = vla.predict('pick up the object', img)
print('Action:', action.shape, action)
"

# 完整评估（双图模式）
python ./vla-scripts/eval_libero.py --device cuda:1 --num_episodes 10 --output_dir ./eval_results_dual
```
