#!/usr/bin/env python3
"""统计报表：汇总 grade.py 产出的逐文件通过率，含 Bootstrap 置信区间。

用法:
    python stats_report.py --results-dir output/py2node/<model>/grade_results
    python stats_report.py --results-dir ... -n 100   # 少采样几次跑快点
"""

import argparse
import json
import os

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[WARN] 未安装 numpy，跳过 Bootstrap 置信区间计算")


def bootstrap_ci(values, n_bootstrap=1000, confidence=0.95):
    """对有放回重采样求均值的分布，返回均值/标准差/置信区间。"""
    if not HAS_NUMPY or not values:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    values = np.array(values)
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        sample = values[np.random.choice(n, size=n, replace=True)]
        means.append(np.mean(sample))
    means = np.array(means)
    alpha = 1 - confidence
    return {
        "mean": float(np.mean(means)),
        "std": float(np.std(means)),
        "ci_lower": float(np.percentile(means, alpha / 2 * 100)),
        "ci_upper": float(np.percentile(means, (1 - alpha / 2) * 100)),
    }


def collect_stats(results_dir):
    """读取全部 *_file_pass_rates.json，按库聚合 micro/macro 指标。"""
    per_library = []
    case_rates = []   # 每个源文件的用例通过率（做 bootstrap）
    pass_flags = []   # 每个源文件是否全过（0/1）

    for filename in sorted(os.listdir(results_dir)):
        if not filename.endswith("_file_pass_rates.json"):
            continue
        library = filename[: -len("_file_pass_rates.json")]
        with open(os.path.join(results_dir, filename)) as f:
            data = json.load(f)

        total = len(data)
        all_pass = sum(1 for v in data.values() if v.get("pass_rate") == 1.0)
        rates = [v.get("pass_rate", 0) for v in data.values()]
        per_library.append({
            "library": library,
            "total": total,
            "all_pass": all_pass,
            "all_pass_rate": all_pass / total if total else 0,
            "case_pass_rate": sum(rates) / total if total else 0,
        })
        case_rates.extend(rates)
        pass_flags.extend(1 if v.get("pass_rate") == 1.0 else 0 for v in data.values())
    return per_library, case_rates, pass_flags


def print_report(results_dir, n_bootstrap):
    per_library, case_rates, pass_flags = collect_stats(results_dir)
    if not per_library:
        print("没有找到任何 *_file_pass_rates.json，先跑 grade.py。")
        return

    print(f"{'库':<20} {'文件数':<8} {'全过数':<8} {'全过率':<12} {'用例通过率':<12}")
    print("-" * 64)
    for row in per_library:
        print(f"{row['library']:<20} {row['total']:<8} {row['all_pass']:<8} "
              f"{row['all_pass_rate']:<12.4f} {row['case_pass_rate']:<12.4f}")
    print("-" * 64)

    total = sum(r["total"] for r in per_library)
    total_pass = sum(r["all_pass"] for r in per_library)
    micro_all_pass = total_pass / total if total else 0
    macro_all_pass = sum(r["all_pass_rate"] for r in per_library) / len(per_library)
    micro_case = sum(r["case_pass_rate"] * r["total"] for r in per_library) / total if total else 0
    macro_case = sum(r["case_pass_rate"] for r in per_library) / len(per_library)

    print("\n===== Micro / Macro 指标 =====")
    print(f"全过率(All-Pass):  micro={micro_all_pass:.4f}  macro={macro_all_pass:.4f}")
    print(f"用例通过率:        micro={micro_case:.4f}  macro={macro_case:.4f}")
    print(f"库数: {len(per_library)}，文件总数: {total}，全过总数: {total_pass}")

    ci_case = bootstrap_ci(case_rates, n_bootstrap)
    ci_pass = bootstrap_ci(pass_flags, n_bootstrap)
    if HAS_NUMPY:
        print("\n===== Bootstrap 95% 置信区间 =====")
        print(f"用例通过率: mean={ci_case['mean']:.4f} "
              f"CI=[{ci_case['ci_lower']:.4f}, {ci_case['ci_upper']:.4f}] (std={ci_case['std']:.4f})")
        print(f"全过率:     mean={ci_pass['mean']:.4f} "
              f"CI=[{ci_pass['ci_lower']:.4f}, {ci_pass['ci_upper']:.4f}] (std={ci_pass['std']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="判分结果统计（含 Bootstrap CI）")
    parser.add_argument("--results-dir", required=True, help="grade.py 的结果目录")
    parser.add_argument("-n", "--n-bootstrap", type=int, default=1000,
                        help="Bootstrap 重采样次数（默认 1000）")
    args = parser.parse_args()

    print(f"[CONFIG] 结果目录: {args.results_dir}")
    print(f"[CONFIG] Bootstrap 次数: {args.n_bootstrap}\n")
    if not os.path.isdir(args.results_dir):
        print(f"[ERROR] 目录不存在: {args.results_dir}")
        return
    print_report(args.results_dir, args.n_bootstrap)


if __name__ == "__main__":
    main()
