#!/usr/bin/env python3
"""用例筛选小工具：只保留「文件名字段出现在另一份清单里」的用例行。

典型用途：用 api_lines.jsonl 里登记过的文件，过滤全量用例集。

用法:
    python tools/filter_cases.py --cases 全量用例.jsonl --keep-with api_lines.jsonl \
        --cases-key file_name --keep-key file -o 过滤后.jsonl
"""

import argparse
import json


def filter_cases(cases_file, keep_file, output_file, cases_key, keep_key):
    """按 keep_file 里 keep_key 字段的取值集合过滤 cases_file。"""
    keep_values = set()
    with open(keep_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if keep_key in obj:
                keep_values.add(obj[keep_key])
    print(f"清单文件共 {len(keep_values)} 个文件条目")

    total = kept = 0
    with open(cases_file, "r", encoding="utf-8") as fin, \
            open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get(cases_key) in keep_values:
                fout.write(line if line.endswith("\n") else line + "\n")
                kept += 1

    print(f"处理 {total} 行，保留 {kept} 行，丢弃 {total - kept} 行")
    print(f"输出: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="按清单过滤用例 jsonl")
    parser.add_argument("--cases", required=True, help="待过滤的用例 jsonl")
    parser.add_argument("--keep-with", required=True, help="清单 jsonl")
    parser.add_argument("--cases-key", default="file_name", help="用例里的文件名字段")
    parser.add_argument("--keep-key", default="file", help="清单里的文件名字段")
    parser.add_argument("-o", "--output", required=True, help="输出路径")
    args = parser.parse_args()
    filter_cases(args.cases, args.keep_with, args.output, args.cases_key, args.keep_key)


if __name__ == "__main__":
    main()
