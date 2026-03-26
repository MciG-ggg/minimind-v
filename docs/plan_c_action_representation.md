# 方案 C: 重新设计动作表示 (Action Representation Redesign)

## 问题根因

MiniMind tokenizer 是 **byte-level BPE**（vocab=6400），导致 3 位数字动作值被切分成多个 sub-tokens：

```
"025" → [18, 4229]   (2 tokens, 字节级 BPE 碎片)
"864" → [26, 24, 22] (3 tokens, 每个数字独立)
Space → [223]         (1 token)
```

单步 7 动作需要 **17 tokens**，模型必须完美预测这 17 个 sub-token 序列才能得到正确的 7 个动作值。任意一位错误则全部动作错误。

关键验证：
- 数字 0-9 各自是**单个 token**（token IDs 18-27）
- 空格是单个 token（223）
- "000" 已 merge 成单个 token [1909]（常见组合），但 "025" "864" 等未 merge

## 设计方案

### 核心洞察

VLA-0 (Qwen2.5-VL) 使用 Qwen tokenizer，数字 0-9 各自是**单个 token**，所以 `"025 864 ..."` 格式下每个动作维度 ≈ 1 个 token。

MiniMind tokenizer 的字节级特性导致 `"025"` 变成 2-3 个碎片 token。但数字 0-9 本身是**单个 token**，空格也是**单个 token**。

因此，将动作格式改为**每动作输出 3 个数字 token**（带空格分隔）：

```
旧格式: "025 864 020 025 026 713 494" → 17 tokens (碎片化)
新格式: "0 2 5 8 6 4 0 2 0 0 2 5 0 2 6 7 1 3 4 9 4" → 27 tokens (每数字=1 token)
```

**新格式特点**：
- 每个数字字符 = 1 token（token ID 18-27）
- 每个空格 = 1 token（token ID 223）
- 7 动作 × 3 数字 + 6 空格 = **27 tokens**
- 格式更规则：模型学习"数字数字数字空格"的循环模式

### 训练时的 Action Mask Augmentation

参考 VLA-0，在动作 token 上做随机 mask：
- 0-10% 的样本不做 mask augmentation
- 其余样本随机 mask 0-40% 的 action token
- 被 mask 的 token 在 labels 中设为 -100

## 修改清单

### 1. `model/model_vla.py`

**新增** `NumberSpaceOnlyProcessor`（与 VLA-0 一致）：
```python
class NumberSpaceOnlyProcessor(LogitsProcessor):
    """Logits processor: 约束生成空间为数字(0-9)、空格和 EOS"""
    def __init__(self, tokenizer, digit_token_ids, space_token_id, eos_token_id):
        self.allowed_tokens = set(digit_token_ids + [space_token_id, eos_token_id])
```

**更新** `denormalize_action`：
- 输入为 21 个数字字符的列表（21 digits = 7 actions × 3 digits）
- 将每 3 个相邻数字合并为一个 bin 值（0-999）
- 反归一化到 [-1, 1]

**更新** `VLAModel.predict`：
- `max_new_tokens=28`（27 action tokens + 1 buffer）
- 使用 `NumberSpaceOnlyProcessor` 约束 logits
- 从生成的文本中提取所有数字字符

**更新** `SYSTEM_PROMPT`：
- 明确说明动作格式：`"Provide space-separated digits (0-9) only."`

### 2. `vla_scripts/prepare_data.py`

**更新** `normalize_action` 保持不变（归一化到 0-999 的整数 bin）

**更新** `convert_sample` 和 `convert_sample_batch`：
```python
# 旧: action_str = " ".join(f"{int(a):03d}" for a in norm)  # "025 864 ..."
# 新: 每动作 3 个数字，空格分隔
action_str = " ".join(f"{int(a):03d}" for a in norm)  # 仍然是 "025 864 ..."
# 但推理提取时按"每 3 个数字字符"解析 → "0 2 5 8 6 4 ..."
```

实际上 prepare_data 不需要改！因为 `normalize_action` 输出还是 0-999 的整数，格式化后还是 "025 864 ..." 字符串。**改动在推理端的提取逻辑**，将 "025 864 ..." → 解析出所有数字 → 按 3 个一组解析。

**但训练时的 tokenization 会碎片化** —— 这是关键问题！

训练时 `"025 864 ..."` 被 tokenizer 切成 17 个碎片 token，labels 也是 17 个对应值。推理时用 LogitsProcessor 约束数字+空格，生成也是 27 个碎片 token。

**所以重点是：推理端需要正确从碎片化的 27 tokens 中提取 21 个数字字符。**

### 3. `vla_scripts/train_sft.py`

**更新** `VLADataCollator`：
- `max_new_tokens` 参数：设为 28（足够 21 数字 + 6 空格 + 1 buffer）
- Action mask augmentation：参考 VLA-0，10% 概率不做，其余随机 mask 0-40%

### 4. `vla_scripts/eval_libero.py`

**更新** `EvalConfig`：
- `max_new_tokens=28`（从 21 改为 28，适应新的 token 计数）

## 关键实现细节

### NumberSpaceOnlyProcessor

```python
class NumberSpaceOnlyProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        # 数字 0-9 的 token IDs
        self.allowed_tokens = set()
        for i in range(10):
            ids = tokenizer.encode(str(i), add_special_tokens=False)
            self.allowed_tokens.update(ids)
        # 空格
        space_ids = tokenizer.encode(" ", add_special_tokens=False)
        self.allowed_tokens.update(space_ids)
        # EOS
        if tokenizer.eos_token_id is not None:
            self.allowed_tokens.add(tokenizer.eos_token_id)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for tid in self.allowed_tokens:
            mask[..., tid] = 0
        return scores + mask
```

### 动作提取：从碎片化 tokens 到连续动作

推理生成的文本形如 `"0 2 5 8 6 4 ..."`（27 tokens），decode 后变成 `"0 2 5 8 6 4 0 2 0 0 2 5 0 2 6 7 1 3 4 9 4"`。

提取逻辑：
```python
# 从生成文本中提取所有数字字符
all_digits = re.findall(r'\d', generated_text)  # ['0','2','5','8','6','4',...]

# 过滤掉空格，确保只有数字
all_digits = [c for c in generated_text if c.isdigit()]

# 按 3 个一组解析为 bin 值
bin_values = []
for i in range(0, len(all_digits), 3):
    chunk = all_digits[i:i+3]
    if len(chunk) == 3:
        bin_values.append(int(''.join(chunk)))
    else:
        # 不足 3 个，用 fallback（中间值 500）
        bin_values.extend([500] * (3 - len(chunk)))

# 不足 7 个动作，填充
while len(bin_values) < 21:
    bin_values.append(500)

# 反归一化
actions = denormalize_action(bin_values[:21])
```

### 训练 vs 推理一致性

| 方面 | 训练 | 推理 |
|------|------|------|
| Action format | `"025 864 ..."` (字符串) | `"0 2 5 8 6 4 ..."` (数字+空格) |
| Tokenization | 碎片化 (17 tokens) | 碎片化 (27 tokens, 但 LogitsProcessor 约束) |
| Labels | 17 个碎片 token 对应值 | N/A |
| Loss 计算 | 只在 action tokens 上（labels=-100 elsewhere） | N/A |
| 动作提取 | N/A | 从生成文本解析所有数字，按 3 个一组 |

**注意**：训练时 labels 对应的是碎片 token IDs（如 [18, 4229, 223, 26, 24, 22, ...]），推理时通过 LogitsProcessor 约束让模型只生成数字 0-9 和空格，这样碎片 token 格式自然对齐。

## 验证步骤

1. **Tokenizer 测试**：确认数字 0-9 和空格都是单 token
2. **推理测试**：用 LogitsProcessor 生成，检查输出只包含数字和空格
3. **动作提取测试**：从生成文本正确提取 21 个数字字符
4. **端到端测试**：训练 1 epoch，评测成功率是否有提升
5. **对比分析**：对比修改前后的动作值分布

## 预期效果

- LogitsProcessor 强制模型只能生成数字和空格，消除了生成无关内容的风险
- `max_new_tokens=28` 精确匹配动作 token 数量
- 动作提取逻辑更加鲁棒（从"提取 3 位数"改为"提取所有数字，按 3 个一组"）
- 如果仍有碎片 token 问题（因为训练和推理的 token 碎片模式不同），需要考虑进一步方案：训练和推理都用纯数字格式（无空格），`max_new_tokens=21`
