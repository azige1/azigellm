import json
import os
from PIL import Image
from datasets import Dataset, Sequence
from datasets import Image as ImageData
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import functools

def safe_json_dumps(x):
    """安全地将对象转为 JSON 字符串，失败时转为普通字符串"""
    try:
        return json.dumps(x, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(x)

def is_empty_value(val):
    """判断是否是'空值'，用于决定是否删除字段"""
    if val is None:
        return True
    if isinstance(val, str) and val == "":
        return True
    if isinstance(val, (list, dict)) and len(val) == 0:
        return True
    # 可选：扩展其他空值类型，如 set(), 0, False 等（目前不删）
    return False

def recursive_convert(obj, keys_to_convert=('ground_truth', 'identity'), remove_empty=True):
    """
    递归遍历对象（dict/list）：
      1. 如果是字典，且键在 keys_to_convert 中 → 转为 JSON 字符串
      2. 如果 remove_empty=True，且值是“空值” → 删除该键
      3. 递归处理子结构
    """
    if isinstance(obj, dict):
        # 注意：遍历时不能直接 del，先收集要删的键
        keys_to_delete = []
        for key in list(obj.keys()):
            val = obj[key]

            # 1. 如果是目标键，且非空 → 转 JSON 字符串
            if key in keys_to_convert and not is_empty_value(val):
                obj[key] = safe_json_dumps(val)

            # 2. 递归处理子对象（无论是否目标键）
            recursive_convert(val, keys_to_convert, remove_empty)

            # 3. 如果开启 remove_empty 且当前值为空 → 标记删除
            if remove_empty and is_empty_value(val):
                keys_to_delete.append(key)

        # 统一删除空字段（避免运行时改变 dict 大小）
        for key in keys_to_delete:
            del obj[key]

    elif isinstance(obj, list):
        # 递归处理列表中每个元素
        for i in range(len(obj)):
            recursive_convert(obj[i], keys_to_convert, remove_empty)

    return obj

def process_line(line, to_str, convert_keys, remove_empty):
    """处理单行 JSONL 数据的函数（用于并发）"""
    json_obj = json.loads(line)

    # --- 图片处理逻辑 ---
    if 'image' in json_obj and json_obj['image'] is not None:
        image_value = json_obj['image']
        
        if isinstance(image_value, list):
            processed_images = []
            for img_item in image_value:
                if isinstance(img_item, str):
                    try:
                        with Image.open(img_item, 'r') as img:
                            processed_images.append(img.copy())
                    except Exception as e:
                        # 打印警告而不是中断，增强鲁棒性
                        print(f"⚠️ 警告: 无法加载图片 {img_item}, 已跳过. 错误: {e}")
                else:
                    print(f"⚠️ 警告: image 列表中的项目不是字符串: {img_item}")
            json_obj['image'] = processed_images
        else:
            print(f"⚠️ 警告: 'image' 字段不是一个列表: {image_value}")

    # --- 复用现有转换逻辑 ---
    if to_str or remove_empty:
        json_obj = recursive_convert(json_obj, convert_keys, remove_empty)
    
    return json_obj

def jsonl_to_parquet(jsonl_path, parquet_path, to_str=False, convert_keys=('ground_truth', 'identity'), remove_empty=True):
    """
    将 JSONL 转为 Parquet，兼容多模态数据设计。
    - to_str=True: 递归转换 convert_keys 中的字段为 JSON 字符串
    - remove_empty=True: 递归删除所有“空值”字段（None/""/[]/{}）
    - 使用并发进程池加速处理。
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        print("ℹ️ 提示：JSONL 文件为空，没有数据可写入 Parquet 文件。")
        return

    # 使用 functools.partial 包装任务函数，固定其他参数
    worker_func = functools.partial(
        process_line,
        to_str=to_str,
        convert_keys=convert_keys,
        remove_empty=remove_empty
    )

    data = []
    with ProcessPoolExecutor() as executor:
        # 提交所有任务到进程池
        future_to_line = {executor.submit(worker_func, line): line for line in lines}
        
        # 使用 tqdm 显示进度条并收集结果
        for future in tqdm(as_completed(future_to_line), total=len(lines), desc="🚀 并发处理中"):
            try:
                result = future.result()
                if result:
                    data.append(result)
            except Exception as e:
                line_info = future_to_line[future][:80] # 显示出错行的前80个字符
                print(f"处理行“{line_info}... ”时发生错误: {e}")

    if not data:
        print("ℹ️ 提示：没有数据可写入 Parquet 文件。")
        return

    # 使用 datasets 库进行转换和保存
    dataset = Dataset.from_list(data)

    # 如果数据中存在 'image' 列，则进行类型转换
    if 'image' in dataset.column_names:
        try:
            dataset = dataset.cast_column("image", Sequence(ImageData()))
        except Exception as e:
            print(f"⚠️ 警告：转换 'image' 列时出错: {e}")
            print("将以原始格式保存。")
    
    dataset.to_parquet(parquet_path)
    print(f"✅ 成功将 {jsonl_path} 转换为 {parquet_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--to_str", type=bool, default=True)
    args = parser.parse_args()
    jsonl_path = args.jsonl_path
    parquet_path = jsonl_path.replace(".jsonl", ".parquet")
    jsonl_to_parquet(jsonl_path, parquet_path, to_str=args.to_str)
    print("parquet_path: ", parquet_path)
    # jsonl_to_parquet("./verl/data/verl_oeis_test.jsonl", "./verl/data/verl_oeis_test.parquet")
