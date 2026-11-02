"""任务导出：把通过可解性检查的题目打包为训练 / 评测 JSONL。"""

import ast
import hashlib
import json

from pydantic import BaseModel, Field

from codeharvest.utils.io import read_json_data
from codeharvest.core.repository import RepoRecord
from codeharvest.workspace import EXECUTION_DIR, TASKS_DIR, REPOS_DIR

DEFAULT_BLOCKLIST_PATH = "data/benchmark_repo_blocklist.txt"


class ExportConfig(BaseModel):
    exp_id: str = Field("temp", description="实验 ID")
    in_file: str | None = Field(
        None, description="可解性检查结果文件，缺省 {exp_id}_solvability.json"
    )
    tasks_file: str | None = Field(
        None, description="规格任务文件，缺省 {exp_id}_specgen.json"
    )
    eval_ratio: float = Field(0.1, description="按仓库维度切分的评测集比例")
    blocklist: str = Field(
        DEFAULT_BLOCKLIST_PATH, description="公开基准仓库黑名单文件"
    )


def load_blocklist(blocklist_path: str) -> set[str]:
    """读黑名单并按「不区分大小写、去掉 org」规整。"""
    names = set()
    with open(blocklist_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            short = line.rsplit("/", 1)[-1].lower()
            names.add(short)
    return names


def is_blocked(repo_id: str, blocklist: set[str]) -> bool:
    """仓库短名（不区分大小写，去掉 org）命中黑名单即过滤。"""
    short = repo_id.rsplit("___", 1)[-1].lower()
    return short in blocklist


def assign_split(repo_id: str, eval_ratio: float) -> str:
    """按仓库维度切分：同一仓库只落在训练或评测一侧。"""
    bucket = int(hashlib.md5(repo_id.encode("utf-8")).hexdigest(), 16) % 100
    return "eval" if bucket < eval_ratio * 100 else "train"


def _signature_from_code(code: str) -> str:
    """从函数 / 方法源码提取带完整类型注解的签名。"""
    tree = ast.parse(code)
    node = tree.body[0]
    if isinstance(node, ast.ClassDef):
        for sub in node.body:
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return ast.unparse(sub).split('"""')[0].split("'''")[0].strip()
    return ast.unparse(node).split('"""')[0].split("'''")[0].strip()


def _repo_commit(repo_id: str) -> str | None:
    try:
        from git import Repo

        repo = Repo(str(REPOS_DIR / repo_id))
        return repo.head.commit.hexsha
    except Exception:
        return None


def _relative_file_path(file_module: dict) -> str:
    """由序列化后的 ModuleRecord 还原相对文件路径（relative_module_path 是属性，不在 dump 结果中）。"""
    identifier = file_module["module_id"]["identifier"]
    module_type = str(file_module.get("module_type", "")).lower()
    if "package" in module_type:
        return identifier.replace(".", "/")
    return f"{identifier.replace('.', '/')}.py"


def _make_task_id(fut, index: int) -> str:
    repo_short = fut.repo_id.rsplit("___", 1)[-1]
    return f"{repo_short}-{fut.name}-{index}"


def export_tasks(args: ExportConfig) -> None:
    solvability_path = EXECUTION_DIR / (args.in_file or f"{args.exp_id}_solvability.json")
    tasks_path = EXECUTION_DIR / (args.tasks_file or f"{args.exp_id}_specgen.json")

    solvability = read_json_data(solvability_path)
    task_records = read_json_data(tasks_path)

    blocklist = load_blocklist(args.blocklist)
    solve_rates = {record["id"]: record for record in solvability["records"]}

    # 原始任务记录按 id 索引，取签名 / 路径等信息
    raw_by_id = {}
    for record in task_records:
        raw = record.get("function_id") or record.get("method_id")
        record_id = raw.get("identifier") if isinstance(raw, dict) else raw
        raw_by_id[record_id] = record

    train_tasks = []
    eval_tasks = []
    counters = {
        "candidates": len(task_records),
        "solvable": 0,
        "blocked": 0,
        "exported": 0,
    }

    index_by_repo_name: dict[str, int] = {}
    for record_id, solv_record in solve_rates.items():
        if solv_record.get("status") == "dropped_unsolvable":
            continue
        counters["solvable"] += 1

        raw = raw_by_id.get(record_id)
        if raw is None:
            continue

        repo_id = raw["file"]["file_module"]["repo"]["repo_id"]
        if is_blocked(repo_id, blocklist):
            counters["blocked"] += 1
            continue

        code = raw.get("function_code") or raw.get("method_code") or ""
        repo_short = repo_id.rsplit("___", 1)[-1]
        index_by_repo_name[repo_short] = index_by_repo_name.get(repo_short, 0) + 1

        tests = raw.get("test_history", {}).get("history", [])
        test_source = ""
        coverage = 0.0
        verdict = ""
        if tests:
            latest = tests[-1]
            test_source = "\n\n".join(latest.get("tests", {}).values())
            exec_stats = latest.get("exec_stats") or {}
            verdict = exec_stats.get("verdict", "")
            coverage_logs = exec_stats.get("coverage_logs") or []
            if coverage_logs:
                coverage = coverage_logs[-1].get("branch_coverage_percentage", 0) / 100

        task = {
            "task_id": f"{repo_short}-{raw.get('function_name') or raw.get('method_name')}-{index_by_repo_name[repo_short]}",
            "signature": _signature_from_code(code),
            "spec": solv_record.get("spec", ""),
            "source_ref": {
                "repo_id": repo_id,
                "file_path": _relative_file_path(raw["file"]["file_module"]),
                "commit": _repo_commit(repo_id),
            },
            "test_source": test_source,
            "coverage": coverage,
            "verdict": verdict,
            "solve_rate": solv_record.get("solve_rate", 0.0),
            "split": assign_split(repo_id, args.eval_ratio),
        }

        counters["exported"] += 1
        if task["split"] == "eval":
            eval_tasks.append(task)
        else:
            train_tasks.append(task)

    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    train_path = TASKS_DIR / f"{args.exp_id}_train.jsonl"
    eval_path = TASKS_DIR / f"{args.exp_id}_eval.jsonl"

    for tasks, path in ((train_tasks, train_path), (eval_tasks, eval_path)):
        with open(path, "w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(json.dumps(counters, indent=2))
    print(f"训练集: {train_path} ({len(train_tasks)} 条)")
    print(f"评测集: {eval_path} ({len(eval_tasks)} 条)")
