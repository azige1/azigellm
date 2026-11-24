"""切片深度扫描报告：逐函数统计各切片深度下的缺失符号占比与上下文长度。"""

import ast
import json

import tiktoken
from pydantic import BaseModel, Field

from codeharvest.core import FunctionRecord, MethodRecord
from codeharvest.utils.io import read_callable_records
from codeharvest.workspace import EXTRACTED_DATA_DIR, REPORTS_DIR
from codeharvest.synthesis.context.full_repo import FullRepoContextBuilder
from codeharvest.analysis.slicing import CodeDependencySlicer
from codeharvest.analysis.slicing.missing_symbols import find_missing_symbols


class SliceReportConfig(BaseModel):
    exp_id: str = Field("temp", description="实验 ID，决定抽取结果文件名")
    in_file: str | None = Field(None, description="抽取结果文件名，缺省 {exp_id}_extracted.json")
    depths: str = Field("1,2,3,5,full", description="扫描的切片深度列表，full 表示整仓上下文")


def _parse_depths(depths: str) -> list[int | None]:
    result = []
    for part in depths.split(","):
        part = part.strip()
        if part == "full":
            result.append(None)
        else:
            result.append(int(part))
    return result


def _slice_context_and_missing(func, depth: int | None) -> tuple[str, list[str]]:
    """按深度切片，返回拼接代码与缺失符号列表；切片失败时返回空串与全部名字。"""
    try:
        if isinstance(func, MethodRecord):
            slicer = CodeDependencySlicer.from_class_models(func.parent_class, depth=depth)
            entry_names = [func.name]
        else:
            slicer = CodeDependencySlicer.from_function_models(func, depth=depth)
            entry_names = [func.name]
        slicer.run()
        file_code = slicer.dependency_graph.unparse_by_file()
        code = "\n\n".join(file_code.values())
    except Exception:
        return "", [func.name]

    missing = find_missing_symbols(code, entry_names)
    return code, missing


def _full_context_tokens(func, tokenizer) -> int | None:
    try:
        context = FullRepoContextBuilder(func, None).get_context().context
        return len(tokenizer.encode(context, disallowed_special=()))
    except Exception:
        return None


def run_slice_report(args: SliceReportConfig) -> dict:
    """对抽取结果逐函数按各深度切片并输出报告 JSON 与表格。"""
    in_file = args.in_file or f"{args.exp_id}_extracted.json"
    functions = read_callable_records(EXTRACTED_DATA_DIR / in_file)

    depths = _parse_depths(args.depths)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    report = {
        "exp_id": args.exp_id,
        "num_functions": len(functions),
        "depths": {},
    }
    for depth in depths:
        key = "full" if depth is None else str(depth)
        report["depths"][key] = {
            "functions": 0,
            "no_missing": 0,
            "tokens": [],
            "over_32k": 0,
        }

    full_tokens = []

    for func in functions:
        for depth in depths:
            key = "full" if depth is None else str(depth)
            stats = report["depths"][key]
            if depth is None:
                tokens = _full_context_tokens(func, tokenizer)
                if tokens is None:
                    continue
                stats["functions"] += 1
                stats["tokens"].append(tokens)
                if tokens > 32000:
                    stats["over_32k"] += 1
                full_tokens.append(tokens)
                continue

            code, missing = _slice_context_and_missing(func, depth)
            if not code:
                continue
            stats["functions"] += 1
            if not missing:
                stats["no_missing"] += 1
            tokens = len(tokenizer.encode(code, disallowed_special=()))
            stats["tokens"].append(tokens)
            if tokens > 32000:
                stats["over_32k"] += 1

    for depth in depths:
        key = "full" if depth is None else str(depth)
        stats = report["depths"][key]
        token_list = stats.pop("tokens")
        stats["mean_tokens"] = sum(token_list) / len(token_list) if token_list else 0
        if key != "full" and stats["functions"]:
            stats["no_missing_ratio"] = stats["no_missing"] / stats["functions"]
        stats["over_32k_ratio"] = (
            stats["over_32k"] / stats["functions"] if stats["functions"] else 0
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{args.exp_id}_slice_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"{'深度':<6} {'函数数':<8} {'无缺失符号占比':<14} {'token 均值':<12} {'>32K 占比':<10}")
    for depth in depths:
        key = "full" if depth is None else str(depth)
        stats = report["depths"][key]
        no_missing = f"{stats.get('no_missing_ratio', 0):.2%}" if key != "full" else "-"
        print(
            f"{key:<6} {stats['functions']:<8} {no_missing:<14} "
            f"{stats['mean_tokens']:<12.0f} {stats['over_32k_ratio']:.2%}"
        )
    print(f"\nResult: {report_path}")
    return report
