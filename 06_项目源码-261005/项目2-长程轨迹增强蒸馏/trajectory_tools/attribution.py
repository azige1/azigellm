"""失败归因：判分层（失败断言与 file:line）→ 轨迹层（最后一次改动位置）→ 九类模式。

规则以声明式列表 ``RULES`` 存放，每条含 id、名称、严重度与判定函数；
统计只在全量轨迹上进行，子集统计必须显式给出条件描述。
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from rebuild_spec.core.report_parser import load_report, report_call_outcomes, collection_errors
from trajectory_tools.behavior_metrics import (
    compute_blind_metrics,
    is_verification_command,
    iter_tool_calls,
)


@dataclass
class Failure:
    """一条失败用例的判分信息。"""

    nodeid: str
    exception_type: str
    file_line: Optional[str]  # 属于 src_dir 的最深一层 file:line
    longrepr: str


@dataclass
class AttributionContext:
    """归因判定上下文。"""

    traj: dict
    task: dict
    failures: list[Failure]
    collection_error: bool
    bash_commands: list[str]  # 实际执行的 bash 命令，按顺序
    tool_outputs: list[str]  # 工具结果输出，按顺序
    tool_error_flags: list[bool]  # 与 bash_commands 对齐到工具结果级
    pass_rate: float
    max_blind: int
    blind_segments: int


FailureRule = tuple[str, str, str, Callable[[AttributionContext], bool]]  # (id, 名称, 严重度, 判定)


def extract_failures(report_path: Path, test_output: str, src_dir: str) -> tuple[list[Failure], bool]:
    """解析 report.json：取失败用例 nodeid、异常类型与最深一层 src_dir 内 file:line。"""
    report = load_report(report_path)
    if report is None:
        return [], True
    outcomes = report_call_outcomes(report)
    failures = []
    for nodeid, call in outcomes.items():
        if call["outcome"] in ("passed", "xfail"):
            continue
        longrepr = str(call.get("longrepr", ""))
        exc_type = ""
        match = re.search(r"^(\w+(?:Error|Exception|Failed|Interrupt))\b", longrepr, re.M)
        if match:
            exc_type = match.group(1)
        file_line = deepest_src_line(longrepr, src_dir)
        failures.append(Failure(nodeid, exc_type, file_line, longrepr))
    return failures, bool(collection_errors(report))


def deepest_src_line(longrepr: str, src_dir: str) -> Optional[str]:
    """在 longrepr 里找属于 src_dir 的最后一处 file:line。"""
    pattern = re.compile(rf"({re.escape(src_dir)}/[\w./-]+\.py):(\d+)")
    matches = pattern.findall(longrepr)
    if not matches:
        return None
    file, line = matches[-1]
    return f"{file}:{line}"


def _bash_history(traj: dict) -> list[tuple[int, str]]:
    """(assistant 下标, bash 命令) 列表。"""
    out = []
    for idx, call in iter_tool_calls(traj["messages"]):
        if call.get("function", {}).get("name") == "bash":
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            out.append((idx, args.get("command", "")))
    return out


def _write_edit_calls(traj: dict) -> list[tuple[int, str]]:
    """(assistant 下标, 文件路径) 列表，来自 write_file/edit_file。"""
    out = []
    for idx, call in iter_tool_calls(traj["messages"]):
        if call.get("function", {}).get("name") in ("write_file", "edit_file"):
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            out.append((idx, args.get("path", "")))
    return out


def last_writer_g(messages: list[dict[str, Any]], file_path: str) -> Optional[int]:
    """倒序查找最后一次写某文件（按 file_path 规范化匹配）的 assistant 的 g。"""
    target = _normalize_path(file_path)
    for idx, message in reversed(list(enumerate(messages))):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if call.get("function", {}).get("name") not in ("write_file", "edit_file"):
                continue
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            if _normalize_path(args.get("path", "")) == target:
                return message.get("g")
    return None


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def build_context(traj: dict, task: dict, report_path: Path, test_output_path: Path) -> AttributionContext:
    """组装归因上下文。"""
    messages = traj["messages"]
    test_output = (
        Path(test_output_path).read_text(encoding="utf-8", errors="replace")
        if Path(test_output_path).exists()
        else ""
    )
    failures, collection_error = extract_failures(report_path, test_output, task.get("src_dir", ""))
    blind = compute_blind_metrics(messages)
    return AttributionContext(
        traj=traj,
        task=task,
        failures=failures,
        collection_error=collection_error,
        bash_commands=[cmd for _, cmd in _bash_history(traj)],
        tool_outputs=[m.get("content", "") for m in messages if m.get("role") == "tool"],
        tool_error_flags=[bool(m.get("is_error")) for m in messages if m.get("role") == "tool"],
        pass_rate=float(traj.get("meta", {}).get("hidden_pass_rate") or 0.0),  # 无判分结果按 0 计
        max_blind=blind["max_blind"],
        blind_segments=blind["blind_segments"],
    )


# ---------------------------------------------------------------------------
# 九类失败模式
# ---------------------------------------------------------------------------

def _has_completion_claim(ctx: AttributionContext) -> bool:
    return any(
        m.get("role") == "assistant" and re.search(r"TASK COMPLETE|all tests pass|task is complete", m.get("content") or "", re.I)
        for m in ctx.traj["messages"]
    )


def _last_verification_idx(ctx: AttributionContext) -> int:
    last = -1
    for idx, cmd in _bash_history(ctx.traj):
        if is_verification_command(cmd):
            last = idx
    return last


def _last_write_idx(ctx: AttributionContext) -> int:
    writes = _write_edit_calls(ctx.traj)
    return max((idx for idx, _ in writes), default=-1)


def _rule_premature_completion(ctx: AttributionContext) -> bool:
    """完成宣告且最后一次改动之后没有执行验证。"""
    if not _has_completion_claim(ctx):
        return False
    return _last_verification_idx(ctx) < _last_write_idx(ctx)


def _rule_incomplete(ctx: AttributionContext) -> bool:
    return ctx.traj.get("meta", {}).get("result_subtype") in ("error_max_turns", "timeout")


def _rule_context_truncation(ctx: AttributionContext) -> bool:
    """出现压缩边界，且边界后第一个 Edit 失败。"""
    if not ctx.traj.get("meta", {}).get("has_compact_boundary"):
        return False
    boundary_g = None
    for m in ctx.traj["messages"]:
        if m.get("role") == "user" and "continued from a previous conversation" in (m.get("content") or ""):
            boundary_g = m.get("g")
    if boundary_g is None:
        return False
    for m in ctx.traj["messages"]:
        if m.get("role") != "tool" or not m.get("is_error"):
            continue
        # 找该工具结果对应的 edit_file 调用
        for call in (c for mm in ctx.traj["messages"] if mm.get("role") == "assistant" for c in mm.get("tool_calls") or []):
            if call.get("id") == m.get("tool_call_id") and call.get("function", {}).get("name") == "edit_file":
                if last_writer_before(ctx.traj["messages"], boundary_g, call) is not None:
                    return True
    return False


def last_writer_before(messages: list[dict], boundary_g: int, call: dict) -> Optional[int]:
    """调用是否在边界之后。"""
    for m in messages:
        if m.get("role") == "assistant" and any(c.get("id") == call.get("id") for c in m.get("tool_calls") or []):
            return m.get("g") if m.get("g", -1) > boundary_g else None
    return None


def _rule_navigation_loop(ctx: AttributionContext) -> bool:
    """连续 >=8 次导航工具且无写操作，或同一文件被 Read >=4 次且期间无改动。"""
    nav_run = 0
    read_counts: dict[str, int] = {}
    written_since_read: set[str] = set()
    for idx, call in iter_tool_calls(ctx.traj["messages"]):
        name = call.get("function", {}).get("name")
        try:
            args = json.loads(call["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        if name in ("read_file", "glob_files", "grep_text"):
            nav_run += 1
            if nav_run >= 8:
                return True
            if name == "read_file":
                path = _normalize_path(args.get("path", ""))
                if path in read_counts and path not in written_since_read:
                    read_counts[path] += 1
                    if read_counts[path] >= 4:
                        return True
                else:
                    read_counts[path] = 1
                written_since_read.discard(path)
        elif name in ("write_file", "edit_file"):
            nav_run = 0
            written_since_read.add(_normalize_path(args.get("path", "")))
        else:
            nav_run = 0
    return False


def _rule_blind_editing(ctx: AttributionContext) -> bool:
    return ctx.max_blind >= 5


def _rule_packaging_failure(ctx: AttributionContext) -> bool:
    """collection 错误、ImportError/ModuleNotFoundError 或 pip install -e . 失败。"""
    if ctx.collection_error:
        return True
    if any(f.exception_type in ("ImportError", "ModuleNotFoundError") for f in ctx.failures):
        return True
    return any("pip install -e ." in out and ("ERROR" in out or "error" in out) for out in ctx.tool_outputs)


def _rule_verification_mismatch(ctx: AttributionContext) -> bool:
    """自测命令最后退出码 0，但隐藏测试通过率 < 0.5。"""
    if ctx.pass_rate >= 0.5:
        return False
    verified_ok = False
    for out in ctx.tool_outputs:
        if "[退出码 0]" in (out or "") and re.search(r"pytest|unittest|tox", out):
            verified_ok = True
    return verified_ok


def _rule_resource_exhaustion(ctx: AttributionContext) -> bool:
    return any(
        re.search(r"MemoryError|Killed|超时（\d+ 秒）", out or "")
        for out in ctx.tool_outputs
    )


def _normalize_package(name: str) -> str:
    """包名规整：小写、- 与 . 统一为 _，去掉版本约束与 extras。"""
    name = re.split(r"[<>=!~\[;@ ]", name.strip().lower(), maxsplit=1)[0]
    return re.sub(r"[-.]+", "_", name)


def _target_package_names(task: dict) -> set[str]:
    """目标库可能的包名：仓库名、任务名与源码目录名。"""
    names = set()
    for raw in (
        task.get("repo", "").split("/")[-1],
        task.get("instance_id", ""),
        Path(task.get("src_dir", "") or "").name,
    ):
        if raw:
            names.add(_normalize_package(raw))
    return names


def _pip_targets(command: str, verb: str) -> list[str]:
    """取 pip install / pip download 后的包名参数（跳过选项及其取值）。"""
    targets = []
    for match in re.finditer(rf"pip3?\s+{verb}\s+([^;&|\n]+)", command):
        tokens = match.group(1).split()
        skip_next = False
        for token in tokens:
            if skip_next:
                skip_next = False
                continue
            if token in ("-r", "-c", "-e", "--requirement", "--constraint", "--editable", "-i", "--index-url", "--extra-index-url", "-d", "--dest"):
                skip_next = True
                continue
            if token.startswith("-") or token in (".", "./"):
                continue
            targets.append(token)
    return targets


def _rule_answer_access(ctx: AttributionContext) -> bool:
    """只看实际执行的 bash 命令与 URL 参数：安装或下载目标库本身、克隆或下载目标仓库、读取隐藏测试或已安装的同名包源码。

    安装其他依赖（如 pip install attrs）与以可编辑方式安装工作区（pip install -e .）不算。
    """
    target_names = _target_package_names(ctx.task)
    repo_name = ctx.task.get("repo", "").split("/")[-1].lower()
    for cmd in ctx.bash_commands:
        lowered = cmd.lower()
        for verb in ("install", "download"):
            if any(_normalize_package(t) in target_names for t in _pip_targets(lowered, verb)):
                return True
        if repo_name and re.search(
            r"(git\s+clone|curl|wget)\s+[^\n]*" + re.escape(repo_name), lowered
        ):
            return True
        if "hidden_tests" in lowered:
            return True
        for name in target_names:
            if re.search(r"site-packages/[^\s]*" + re.escape(name), lowered):
                return True
    return False


RULES: list[FailureRule] = [
    ("premature_completion", "过早终止", "high", _rule_premature_completion),
    ("incomplete", "未完成", "high", _rule_incomplete),
    ("context_truncation", "上下文截断", "medium", _rule_context_truncation),
    ("navigation_loop", "导航空转", "medium", _rule_navigation_loop),
    ("blind_editing", "盲改", "medium", _rule_blind_editing),
    ("packaging_failure", "依赖与包结构失败", "high", _rule_packaging_failure),
    ("verification_mismatch", "自测与判分不一致", "high", _rule_verification_mismatch),
    ("resource_exhaustion", "资源耗尽", "medium", _rule_resource_exhaustion),
    ("answer_access", "答案获取", "high", _rule_answer_access),
]


def attribute(traj: dict, task: dict, report_path: Path, test_output_path: Path) -> set[str]:
    """对单条轨迹执行全部规则，返回命中的模式 id 集合。"""
    ctx = build_context(traj, task, report_path, test_output_path)
    return {rule_id for rule_id, _, _, match in RULES if match(ctx)}


def attribute_all(
    exported_dir: Path,
    tasks_file: Path,
    reports_root: Path,
    out_path: Path,
) -> dict[str, list[str]]:
    """对全部导出轨迹做归因，写 attribution_report.json，返回 {id: [模式]}。"""
    tasks = {}
    for line in Path(tasks_file).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            tasks[record["instance_id"]] = record
    results: dict[str, list[str]] = {}
    for traj_file in sorted(Path(exported_dir).glob("*.json")):
        traj = json.loads(traj_file.read_text(encoding="utf-8"))
        task = tasks.get(traj.get("instance_id"), {})
        # 判分结果由 judge_run 复制到轨迹目录 runs/{instance_id}/{run_id}/
        instance_id, run_id = traj["id"].split("__")
        report_path = Path(reports_root) / instance_id / run_id / "report.json"
        output_path = Path(reports_root) / instance_id / run_id / "test_output.txt"
        results[traj["id"]] = sorted(attribute(traj, task, report_path, output_path))
    report = [
        {"id": tid, "modes": modes, "num_modes": len(modes)} for tid, modes in sorted(results.items())
    ]
    Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def summarize(
    results: dict[str, list[str]],
    index_entries: list[dict],
    *,
    condition: Optional[str] = None,
    excluded: int = 0,
) -> list[dict]:
    """全量统计：命中轨迹数与命中轨迹平均 hidden_pass_rate 两列。

    传入 condition 即表示显式声明的子集条件，输出中写明被排除样本数。
    """
    pass_rates = {e["id"]: e.get("hidden_pass_rate") for e in index_entries}
    if excluded and not condition:
        raise ValueError("统计排除了样本时必须显式传入 condition 描述")
    stats: dict[str, dict] = {}
    for tid, modes in results.items():
        for mode in modes:
            bucket = stats.setdefault(mode, {"hits": 0, "pass_sum": 0.0, "pass_n": 0})
            bucket["hits"] += 1
            rate = pass_rates.get(tid)
            if rate is not None:
                bucket["pass_sum"] += rate
                bucket["pass_n"] += 1
    table = [
        {
            "mode": mode,
            "num_trajs": b["hits"],
            "avg_hidden_pass_rate": round(b["pass_sum"] / b["pass_n"], 4) if b["pass_n"] else None,
            "condition": condition,
            "excluded": excluded if condition else 0,
        }
        for mode, b in sorted(stats.items(), key=lambda kv: -kv[1]["hits"])
    ]
    return table


__all__ = []
