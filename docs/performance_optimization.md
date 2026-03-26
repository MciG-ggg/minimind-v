# VLA Scripts 性能优化清单

> 生成日期：2026-03-26
> 目标脚本：`vla_scripts/prepare_data.py`、`vla_scripts/eval_libero.py`

---

## 一、`prepare_data.py` 优化清单

### 1. `filter` 阶段 — 单进程遍历全量

**位置**：第 226 行

**问题**：`dataset.filter(lambda x: x["task_index"] in task_indices)` 无并行选项，273,465 条全遍历，单进程逐条 Python 过滤，288 examples/s，累计耗时 ~16min。

**优化**：
```python
# 加 num_proc=4 多进程过滤
filtered = dataset.filter(
    lambda x: x["task_index"] in task_indices,
    num_proc=4,
    desc="过滤中",
)
```

**预期收益**：16min → ~4-5min

---

### 2. `map` 阶段 — 最大瓶颈

**位置**：第 232 行

**问题**：单进程逐条处理，无 `batched` 和 `num_proc`，52,042 条样本耗时 **1h25min**（10.2 examples/s）。

**优化 A — 批量化**：
```python
# convert_sample 改为接收 batch（dict of lists），返回 dict of lists
def convert_sample(batch: dict) -> dict:
    results = {"input_text": [], "action_text": [], "images": [], "_orig_task_index": []}
    for i in range(len(batch["task_index"])):
        s = {k: batch[k][i] for k in batch}
        _, language = TASK_MAP.get(s["task_index"], (None, f"Task {s['task_index']}"))
        norm = normalize_action(s["action"])
        action_str = " ".join(f"{int(a):03d}" for a in norm)
        results["input_text"].append(f"Instruction: {language}\nAction:")
        results["action_text"].append(action_str)
        results["images"].append([s["observation.images.image"], s["observation.images.image2"]])
        results["_orig_task_index"].append(s["task_index"])
    return results

# 调用时加 batched=True + batch_size + num_proc
converted = filtered.map(
    convert_sample,
    batched=True,
    batch_size=1000,
    num_proc=4,
    remove_columns=filtered.column_names,
    desc="转换中",
)
```

**预期收益**：1h25min → **5-10min**（提升 ~10x）

---

### 3. `task_counts` 统计 — 冗余遍历

**位置**：第 236-242 行

**问题**：转换后的 52,042 条样本再遍历一遍，纯 Python 循环，耗时约 5min。

**优化**：
```python
# 使用 collections.Counter 替代循环
from collections import Counter
task_counts = Counter(s["_orig_task_index"] for s in converted)
```

**预期收益**：~5min → ~30s

---

### 4. `save_to_disk` 阶段 — 磁盘写入

**位置**：第 245 行

**问题**：52,042 条含图像数据写 Arrow 格式，无并行写入选项。

**优化**：
- 确认输出磁盘为 SSD
- 可选压缩：`converted.save_to_disk(self.output, compression="gzip")`（减少写入量但增加 CPU 开销，总时间可能不变或更长）

**预期收益**：取决于磁盘 I/O

---

### 5. `show_task_distribution` — 小优化

**位置**：第 68-81 行

**问题**：`dataset.take(n)` 再遍历，内存效率略低。

**优化**：当前逻辑可接受，无需修改。

---

### 耗时对比汇总

| 阶段 | 当前耗时 | 优化后预估 | 提升 |
|------|---------|-----------|------|
| filter | ~16min | ~4-5min | ~4x |
| **map** | **~1h25min** | **~8min** | **~10x** |
| task_counts | ~5min | ~30s | ~10x |
| save_to_disk | 待测 | 取决于磁盘 | — |

---

## 二、`eval_libero.py` 优化清单

### 1. 任务间并行 — 最大优化点

**位置**：第 209 行

**问题**：`for task_idx, task in enumerate(tasks)` 逐任务串行评测，N 个任务耗时 = N × 单任务耗时。

**优化方案 A — multiprocessing（推荐）**：

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def init_worker(checkpoint, max_new_tokens, device):
    """每个 worker 进程启动时初始化 VLA 模型（只执行一次）"""
    global _vla_model, _device
    _vla_model = VLAModel(checkpoint, device=device, dual_image=True, max_new_tokens=max_new_tokens)
    _device = device

def eval_single_task(args):
    """单个任务评测（每个 worker 调用一次）"""
    task_idx, bddl_path, instruction, max_steps, save_dir, verbose, seed = args
    global _vla_model
    np.random.seed(seed + task_idx)
    torch.manual_seed(seed + task_idx)

    task_env = OffScreenRenderEnv(bddl_file_name=bddl_path)
    t0 = time.time()
    result = evaluate_task(
        vla=_vla_model, env=task_env,
        instruction=instruction, max_steps=max_steps,
        save_dir=save_dir, verbose=verbose,
    )
    result["task_idx"] = task_idx
    result["instruction"] = instruction
    result["elapsed_s"] = time.time() - t0
    return result

# 在 run_eval 中替换逐任务循环：
tasks_args = []
for task_idx, task in enumerate(tasks[:num_episodes]):
    bddl_path = benchmark.get_task_bddl_file_path(task_idx)
    save_dir = os.path.join(output_dir, f"task_{task_idx:02d}") if True else ""
    tasks_args.append((task_idx, bddl_path, task.language, max_steps, save_dir, verbose, seed))

# ProcessPoolExecutor 并行执行
num_workers = min(num_episodes, 4)  # 根据 GPU 显存调整
with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker,
                         initargs=(checkpoint, max_new_tokens, device)) as executor:
    futures = {executor.submit(eval_single_task, args): args[0] for args in tasks_args}
    for future in as_completed(futures):
        result = future.result()
        results.append(result)
        # 更新进度表...
```

**注意事项**：
- `VLAModel` 必须在 `init_worker` 中创建（每个 worker 进程独立加载模型到 GPU）
- GPU 显存约束：`num_workers` 需要根据模型大小调整（避免 OOM）
- `OffScreenRenderEnv` 在子进程中创建，每个任务独立环境（无状态共享问题）

**预期收益**：N 个任务 → 接近单个最长任务耗时（提升 N 倍）

---

### 2. 环境创建优化 — 减少重复初始化

**位置**：第 212 行

**问题**：每个任务循环内 `OffScreenRenderEnv(bddl_path)`，LIBERO 环境初始化较慢。

**优化**：在任务并行方案中，每个 worker 进程已经各自维护环境，无需额外优化。如果切换到串行评测，可以将同 benchmark 下的环境复用。

**预期收益**：取决于环境初始化耗时

---

### 3. 视频保存优化 — 减少中间 I/O

**位置**：第 146-149 行

**问题**：每 50 步写一张 PNG 到磁盘，小文件 I/O 开销大。

**优化**：
```python
# 去掉中间 PNG 保存，只在最后保存视频
# 当前第 146-149 行可注释掉，保留第 155-162 行的视频合成
if step % 50 == 0:
    p = Path(save_dir) / f"step_{step:04d}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_flip_frame(rgb)).save(p)  # <- 注释掉
```

**预期收益**：减少大量小文件写入

---

### 4. VLA 推理优化 — GPU 利用率

**位置**：第 129 行

**问题**：每个 step 推理 batch=1，GPU 利用率低。

**优化**：当前 `VLAModel.predict` 已在 GPU 上推理，batch=1 是 rollout 场景固有限制。长期可考虑：
- 收集多个 episode 的 observation 批量推理
- 使用 KV-Cache 预取

**预期收益**：GPU 利用率提升，但架构改动较大，短期不推荐。

---

### 耗时对比汇总

| 阶段 | 当前 | 优化后预估 | 提升 |
|------|------|-----------|------|
| **任务并行** | N × 单任务耗时 | ~单任务最长耗时 | **N 倍** |
| 环境复用 | 每任务重建 | 同 benchmark 复用 | 省初始化 |
| 中间 PNG 保存 | 每 50 步写磁盘 | 去掉 | 减少 I/O |

---

## 三、优化优先级

| 优先级 | 脚本 | 阶段 | 优化方式 | 预期收益 | 改动量 |
|--------|------|------|---------|---------|--------|
| P0 | prepare_data | map | +batched +num_proc=4 | ~1h25min→8min | 中（改函数签名）|
| P0 | eval_libero | 任务并行 | ProcessPoolExecutor | N×→1× | 中（封装 worker）|
| P1 | prepare_data | filter | +num_proc=4 | 16min→4min | 小 |
| P1 | prepare_data | task_counts | Counter | ~5min→30s | 小 |
| P2 | eval_libero | 中间 PNG | 注释掉 | 减少 I/O | 微 |
