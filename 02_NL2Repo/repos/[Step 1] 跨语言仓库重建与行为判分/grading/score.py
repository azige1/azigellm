"""逐库计分主逻辑。

输入：data/testcases/<task>/*.jsonl（每行一条用例，含文件名字段 + CLI 参数）。
输出：每个用例文件的 file_pass_rates.json，以及控制台汇总指标。

两套任务的判分差异：
- py2node：参考实现是「Python 源文件 / 预编译产物」，生成物是 .mjs 包，
  逐用例跑两端、按行比对 stdout；指标含 API 覆盖率（行级匹配比例）。
- cpp2rust：参考实现是 C++ 预编译二进制，生成物是编译好的 Rust 二进制，
  整段 stdout 比对（允许浮点兜底相等），并额外做难度分桶与错误分类。
"""

import json
import os
from collections import defaultdict

from harness.tasks.base import entry_name, package_dir, strip_ext

from . import compare

# cpp2rust 错误分类标签
ERR_NO_EXEC = "no_executable"
ERR_FIRST4_FAIL = "first4_not_pass"
ERR_RUNTIME = "runtime_error"
ERR_WRONG_OUTPUT = "wrong_output"
ERR_ALL_PASS = "all_pass"


# ---------- 用例读取 ----------

def load_cases(jsonl_path, key_field):
    """读一个用例 jsonl，返回 {文件名: [参数dict/记录, ...]} 与 {文件名: [原始行]}。"""
    groups = defaultdict(list)
    raw_records = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            fname = item[key_field]
            raw_records[fname].append(item)
            params = dict(item)
            params.pop(key_field)
            groups[fname].append(params)
    return groups, raw_records


def load_jsonl_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_jsonl_cache(cache, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=4, ensure_ascii=False)


# ---------- py2node ----------

def find_mjs_entry(pkg_dir):
    """在产物目录里挑入口 .mjs：优先 test 开头，其次任意 .mjs。"""
    if not os.path.isdir(pkg_dir):
        return None
    mjs_files = [f for f in os.listdir(pkg_dir) if f.endswith(".mjs")]
    if not mjs_files:
        return None
    for f in mjs_files:
        if f.startswith("test"):
            return f
    return mjs_files[0]


def score_py2node_cases(groups, dataset_root, output_root, runner, timeout=5, verbose=False):
    """逐文件判分，返回 (file_pass_rates, 汇总指标)。"""
    file_pass_rates = {}
    sample_line_rates = []
    all_pass_files = 0

    for filename, samples in groups.items():
        source_path = os.path.join(dataset_root, filename)
        pkg_dir = package_dir(output_root, filename, ".py")
        entry_file = find_mjs_entry(pkg_dir)

        passed = 0
        for params in samples:
            args = compare.kwargs_style_args(params)
            result = None
            if entry_file:
                ref_out, ref_code = runner.run_source(source_path, args, timeout)
                gen_out, gen_code = runner.run_package(pkg_dir, entry_file, args, timeout)
                if ref_code == 0 and gen_code == 0 and ref_out is not None and gen_out is not None:
                    ref_lines = compare.normalize_lines(ref_out)
                    gen_lines = compare.normalize_lines(gen_out)
                    counted = compare.count_matched_lines(ref_lines, gen_lines)
                    if counted:
                        result = counted
                elif verbose:
                    print(f"[DEBUG {filename}] ref_code={ref_code}, gen_code={gen_code}, args={args}")

            if result:
                matched, total = result
                sample_line_rates.append(matched / total)
                if matched == total:
                    passed += 1
            else:
                sample_line_rates.append(0.0)

        rate = passed / len(samples) if samples else 0
        file_pass_rates[filename] = {
            "pass_rate": rate,
            "passed_samples": passed,
            "total_samples": len(samples),
        }
        if samples and passed == len(samples):
            all_pass_files += 1

    file_count = len(file_pass_rates)
    metrics = {
        "all_pass_rate": all_pass_files / file_count if file_count else 0,
        "test_case_pass_rate": (
            sum(v["pass_rate"] for v in file_pass_rates.values()) / file_count if file_count else 0
        ),
        "api_coverage": sum(sample_line_rates) / len(sample_line_rates) if sample_line_rates else 0,
        "file_count": file_count,
        "all_pass_files": all_pass_files,
    }
    return file_pass_rates, metrics


def run_py2node_grading(cases_dir, dataset_root, output_root, results_dir, runner, timeout=5):
    """对目录下全部用例文件判分并打印汇总。"""
    jsonl_files = sorted(f for f in os.listdir(cases_dir) if f.endswith(".jsonl"))
    print(f"发现 {len(jsonl_files)} 个用例文件\n")
    all_metrics = []
    for file in jsonl_files:
        groups, _ = load_cases(os.path.join(cases_dir, file), key_field="filename")
        file_pass_rates, metrics = score_py2node_cases(
            groups, dataset_root, output_root, runner, timeout=timeout
        )
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(
            results_dir, f"{os.path.splitext(file)[0]}_file_pass_rates.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(file_pass_rates, f, indent=2, ensure_ascii=False)

        print(f"===== {file} =====")
        print(f"文件数: {metrics['file_count']}，全过文件数: {metrics['all_pass_files']}")
        print(f"全过率(All-Pass):     {metrics['all_pass_rate']:.4f}")
        print(f"用例通过率:           {metrics['test_case_pass_rate']:.4f}")
        print(f"API 行覆盖率:         {metrics['api_coverage']:.4f}\n")
        all_metrics.append(metrics)

    def _avg(key):
        return sum(m[key] for m in all_metrics) / len(all_metrics) if all_metrics else 0

    print("=" * 60)
    print("总体平均：")
    print(f"全过率(All-Pass):     {_avg('all_pass_rate'):.4f}")
    print(f"用例通过率:           {_avg('test_case_pass_rate'):.4f}")
    print(f"API 行覆盖率:         {_avg('api_coverage'):.4f}")


# ---------- cpp2rust ----------

def _find_rust_binaries(packages_root, library, test_name):
    """在产物目录里找名为 testN 的可执行文件（cargo 可能编到子目录）。"""
    search_dir = os.path.join(packages_root, library, f"{test_name}_pkg")
    found = []
    if os.path.isdir(search_dir):
        for root, _dirs, files in os.walk(search_dir):
            for file in files:
                if file == test_name:
                    path = os.path.join(root, file)
                    if os.access(path, os.X_OK):
                        found.append(path)
    return found


def score_one_cpp2rust(record, cpp_base, packages_root, runner, timeout=3):
    """单条用例：参考二进制与任一 Rust 二进制输出一致记 1，否则 0。"""
    rel_path = record["file_name"]
    args = compare.positional_args(record)

    ref_exe = os.path.join(cpp_base, rel_path.replace(".cpp", ""))
    if not os.path.exists(ref_exe):
        return 0

    library = rel_path.split("/")[0]
    test_name = os.path.basename(rel_path).replace(".cpp", "")
    rust_exes = _find_rust_binaries(packages_root, library, test_name)
    if not rust_exes:
        return 0

    ref_out, _ = runner.run_binary(ref_exe, args, timeout)
    if ref_out is None:
        return 0
    for rust_exe in rust_exes:
        gen_out, _ = runner.run_binary(rust_exe, args, timeout)
        if gen_out is not None and compare.outputs_equal(ref_out, gen_out):
            return 1
    return 0


def classify_file_errors(case_records, cpp_base, packages_root, runner, timeout=3):
    """按文件归类失败原因：未产出 / 前 4 例未全过 / 运行错 / 输出不一致 / 全过。"""
    rel_path = case_records[0]["file_name"]
    library = rel_path.split("/")[0]
    test_name = os.path.basename(rel_path).replace(".cpp", "")

    rust_exes = _find_rust_binaries(packages_root, library, test_name)
    ref_exe = os.path.join(cpp_base, rel_path.replace(".cpp", ""))
    if not rust_exes or not os.path.exists(ref_exe):
        return ERR_NO_EXEC

    scores = []
    for record in case_records:
        args = compare.positional_args(record)
        ref_out, _ = runner.run_binary(ref_exe, args, timeout)
        if ref_out is None:
            continue
        case_score, case_had_error = 0, False
        for rust_exe in rust_exes:
            gen_out, had_error = runner.run_binary(rust_exe, args, timeout)
            if had_error:
                case_had_error = True
            if gen_out is not None and compare.outputs_equal(ref_out, gen_out):
                case_score, case_had_error = 1, False
                break
        scores.append((case_score, case_had_error))

    if not scores:
        return ERR_NO_EXEC
    if all(s == 1 for s, _ in scores):
        return ERR_ALL_PASS
    if not all(s == 1 for s, _ in scores[:4]):
        return ERR_FIRST4_FAIL
    if any(e for _, e in scores):
        return ERR_RUNTIME
    return ERR_WRONG_OUTPUT


def load_difficulty_buckets(api_lines_path):
    """按库总行数把文件三等分为 easy/medium/hard；文件缺失时返回空表。"""
    if not api_lines_path or not os.path.exists(api_lines_path):
        return {}
    rows = []
    with open(api_lines_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                rows.append((obj["file"], obj["lib_total_lines"]))
    rows.sort(key=lambda item: item[1])
    total = len(rows)
    if total == 0:
        return {}
    base, remainder = divmod(total, 3)
    sizes = [base + (1 if i < remainder else 0) for i in range(3)]
    buckets = {}
    idx = 0
    for bucket, size in zip(["easy", "medium", "hard"], sizes):
        for _ in range(size):
            if idx >= total:
                break
            buckets[rows[idx][0]] = bucket
            idx += 1
    return buckets


def run_cpp2rust_grading(cases_path, cpp_base, packages_root, cache_path, runner,
                         api_lines_path=None, timeout=3):
    """cpp2rust 判分主流程：带缓存逐用例计分，打印分库/分难度/错误分类报表。"""
    cache = load_jsonl_cache(cache_path)
    data_tree = defaultdict(lambda: defaultdict(list))
    file_records = defaultdict(list)
    new_runs = cached_runs = 0

    with open(cases_path, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    for line in lines:
        record = json.loads(line)
        file_records[record["file_name"]].append(record)
        cache_key = json.dumps(record, sort_keys=True)
        if cache_key in cache:
            score = cache[cache_key]
            cached_runs += 1
        else:
            score = score_one_cpp2rust(record, cpp_base, packages_root, runner, timeout)
            cache[cache_key] = score
            new_runs += 1
        if score is not None:
            data_tree[record["file_name"].split("/")[0]][record["file_name"]].append(score)

    save_jsonl_cache(cache, cache_path)
    print(f"\n[Summary] 命中缓存 {cached_runs} 条，新跑 {new_runs} 条。")

    # 难度分桶：api 行数三等分后，再按测试编号修正（<7 easy，<14 medium，其余 hard）
    buckets = load_difficulty_buckets(api_lines_path)
    for key in buckets:
        num = int(key.split("/test")[-1].split(".")[0])
        buckets[key] = "easy" if num < 7 else ("medium" if num < 14 else "hard")
    bucket_stats = {b: {"files": 0, "pass": 0, "case_pass": 0, "case_total": 0}
                    for b in ("easy", "medium", "hard")}

    pkg_stats = {}
    all_scores = []
    total_pass_files = total_files = 0
    for pkg, files in data_tree.items():
        pkg_pass_files = 0
        pkg_scores = []
        for fname, scores in files.items():
            is_all_pass = int(all(s == 1 for s in scores))
            pkg_pass_files += is_all_pass
            pkg_scores.extend(scores)
            all_scores.extend(scores)
            bucket = buckets.get(fname)
            if bucket:
                bucket_stats[bucket]["files"] += 1
                bucket_stats[bucket]["pass"] += is_all_pass
                bucket_stats[bucket]["case_pass"] += sum(scores)
                bucket_stats[bucket]["case_total"] += len(scores)
        pkg_stats[pkg] = {
            "total_files": len(files),
            "pass_files": pkg_pass_files,
            "all_pass_rate": pkg_pass_files / len(files) if files else 0,
            "sample_rate": sum(pkg_scores) / len(pkg_scores) if pkg_scores else 0,
        }
        total_pass_files += pkg_pass_files
        total_files += len(files)

    print("\n" + "=" * 70)
    print(f"{'库':<15} | {'文件(过/总)':<14} | {'文件全过率':<10} | {'用例通过率':<10}")
    print("-" * 70)
    for pkg, s in pkg_stats.items():
        print(f"{pkg:<15} | {s['pass_files']}/{s['total_files']:<12} | "
              f"{s['all_pass_rate']:>10.2%} | {s['sample_rate']:>10.2%}")
    if not pkg_stats:
        print("\n没有任何有效判分记录。")
        return

    macro_all_pass = sum(s["all_pass_rate"] for s in pkg_stats.values()) / len(pkg_stats)
    macro_sample = sum(s["sample_rate"] for s in pkg_stats.values()) / len(pkg_stats)
    micro_all_pass = total_pass_files / total_files if total_files else 0
    micro_sample = sum(all_scores) / len(all_scores) if all_scores else 0

    if any(b["files"] for b in bucket_stats.values()):
        print("\n难度分桶（按库规模三等分）：")
        print(f"{'桶':<8} | {'文件':<6} | {'文件全过率':<10} | {'用例通过率':<10}")
        print("-" * 50)
        for bucket in ("easy", "medium", "hard"):
            s = bucket_stats[bucket]
            file_rate = s["pass"] / s["files"] if s["files"] else 0
            case_rate = s["case_pass"] / s["case_total"] if s["case_total"] else 0
            print(f"{bucket:<8} | {s['files']:<6} | {file_rate:>10.2%} | {case_rate:>10.2%}")

    print("\n汇总指标：")
    print(f"  1. Micro 全过率:  {micro_all_pass:>7.2%}  ({total_pass_files}/{total_files} 个文件)")
    print(f"  2. Macro 全过率:  {macro_all_pass:>7.2%}  （按库平均）")
    print(f"  3. Micro 用例率:  {micro_sample:>7.2%}  ({sum(all_scores)}/{len(all_scores)} 条用例)")
    print(f"  4. Macro 用例率:  {macro_sample:>7.2%}  （按库平均）")

    # 错误分类
    print("\n" + "=" * 70)
    print("失败原因分类（逐文件复查，可能较慢）...")
    error_counts = defaultdict(int)
    pkg_error = defaultdict(lambda: defaultdict(int))
    for fname, records in file_records.items():
        err = classify_file_errors(records, cpp_base, packages_root, runner, timeout)
        error_counts[err] += 1
        pkg_error[fname.split("/")[0]][err] += 1

    label_map = {
        ERR_ALL_PASS: "全部通过",
        ERR_NO_EXEC: "1. 未产出可执行文件",
        ERR_FIRST4_FAIL: "2. 前 4 条用例未全过",
        ERR_RUNTIME: "3. 运行错误 / 超时",
        ERR_WRONG_OUTPUT: "4. 输出与参考不一致",
    }
    total_classified = sum(error_counts.values())
    print(f"\n{'类别':<24} | {'文件数':>6} | {'占比':>8}")
    print("-" * 46)
    for key in [ERR_ALL_PASS, ERR_NO_EXEC, ERR_FIRST4_FAIL, ERR_RUNTIME, ERR_WRONG_OUTPUT]:
        cnt = error_counts[key]
        pct = cnt / total_classified if total_classified else 0
        print(f"{label_map[key]:<24} | {cnt:>6} | {pct:>7.2%}")
    print("-" * 46)
    print(f"{'合计':<24} | {total_classified:>6}")

    col_keys = [ERR_ALL_PASS, ERR_NO_EXEC, ERR_FIRST4_FAIL, ERR_RUNTIME, ERR_WRONG_OUTPUT]
    col_short = ["全过", "未产出", "前4未过", "运行错", "输出错"]
    print("\n按库细分：")
    header = f"{'库':<20}" + "".join(f" | {h:>8}" for h in col_short)
    print(header)
    print("-" * (24 + 11 * len(col_keys)))
    for pkg in sorted(pkg_error):
        row = f"{pkg:<20}"
        for key in col_keys:
            row += f" | {pkg_error[pkg][key]:>8}"
        print(row)
    print("=" * 70 + "\n")
