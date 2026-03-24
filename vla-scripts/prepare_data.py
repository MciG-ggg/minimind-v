from datasets import load_dataset, get_dataset_split_names, get_dataset_config_names
import numpy as np

def normalize_action(action, low=-1, high=1, token_range=(0,999)):
    # 假设原始动作范围已知（例如 [-1,1] 或从数据集统计）
    return ((action - low) / (high - low) * (token_range[1]-token_range[0]) + token_range[0]).astype(int)

def convert_to_vla0_format(sample, chunk_size=10):
    instructions = sample["observation.language_instruction"]
    images = sample["observation.images"]["image"]          # 可能为多帧
    actions = sample["actions"]                              # shape (seq_len, 7)

    # 仅使用前 chunk_size 步作为动作块
    action_chunk = actions[:chunk_size]
    norm_actions = normalize_action(action_chunk)

    # 转为字符串："a1 a2 a3 ... | b1 b2 b3 ..."（这里只有一个块，但后续可以多块）
    action_str = " ".join([f"{int(a)}" for a in norm_actions.flatten()])   # 可选按步或按自由度展平

    # 构建输入文本
    input_text = f"Instruction: {instructions}\nAction:"

    return {"input_text": input_text, "action_text": action_str, "images": images}

configs = get_dataset_config_names("HuggingFaceVLA/libero")
print(configs)   # 应该输出 ['default']

splits = get_dataset_split_names("HuggingFaceVLA/libero")
print("Splits:", splits)

# 加载数据集（只取部分演示）
dataset = load_dataset("HuggingFaceVLA/libero", "default", split="train")

print("Column names:", dataset.column_names)
print("First sample:", dataset[0])
# 假设任务标识在字段 "task" 中，值为 "libero_object"
libero_object_dataset = dataset.filter(lambda x: x["task"] == "libero_object")

# 取前10条作为示例
subset = libero_object_dataset.select(range(10))

converted = libero_object_dataset.map(convert_to_vla0_format, remove_columns=libero_object_dataset.column_names)
print(f"转换完成，共 {len(converted)} 条")
converted.save_to_disk("./vla0_libero_object_train")
print(f"已保存到 ./vla0_libero_object_train")