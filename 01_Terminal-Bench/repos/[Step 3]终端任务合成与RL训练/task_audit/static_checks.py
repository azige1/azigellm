"""静态审计：不启动容器、不联网，纯文件与 AST 级别的质量检查。

检查项：

- ``structural_score``：终态测试中断言「任务结果」（进程退出码 / 标准输出 /
  文件内容）的比例。比例为 0 说明 verifier 只检查被动状态（文件存在、
  权限位等），属于结构性缺陷。
- ``desc_overlap``：终态测试断言中引用题面关键词的比例，衡量 verifier
  是否真的在检查该任务专属的行为。
- ``assert_diversity``：断言覆盖的类别数。
- 模式完整性：Harbor 目录必需文件是否齐全。
- 语法有效性：两个测试文件必须能被 ``ast.parse`` 解析。
- 元数据有效性：task.toml 的 difficulty / category 字段。
"""
from __future__ import annotations

import ast
import re
import statistics
from pathlib import Path
from typing import Any

try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from task_audit import Decision, TaskAuditResult


# Harbor 任务目录的必需文件
REQUIRED_FILES = [
    "instruction.md",
    "task.toml",
    "environment/Dockerfile",
    "environment/task.json",
    "environment/test_initial_state.py",
    "tests/test_final_state.py",
    "tests/test.sh",
]

VALID_DIFFICULTIES = {"easy", "medium", "hard", "intricate"}

# 断言分类：按 ast.unparse 后的文本做关键词匹配
ASSERT_CATEGORIES: dict[str, tuple[str, ...]] = {
    "file_exists": ("os.path.exists", "os.path.isfile", "os.path.isdir", "os.path.islink", ".exists()"),
    "file_content": ("open(", ".read(", ".readlines(", "content"),
    "permissions": ("os.stat", "st_mode", "oct(", "os.access"),
    "process_exit": ("subprocess", "returncode", "Popen", "check_call", "check_output"),
    "stdout_output": ("stdout", "stderr", "capture_output"),
}

# 检查「任务结果」而非「被动状态」的类别
OUTCOME_CATEGORIES = {"process_exit", "stdout_output", "file_content"}

# 结构性阈值
STRUCTURAL_REJECT = 0.0   # 等于 0：没有任何结果性断言，直接拒绝
STRUCTURAL_FLAG = 0.2     # 低于 0.2：结果性断言占比过低，标记复查

# 题面关键词提取时的停用词
STOPWORDS = {
    "the", "and", "for", "you", "your", "yours", "with", "that", "this", "from",
    "have", "has", "had", "are", "was", "were", "will", "would", "should", "could",
    "can", "not", "but", "all", "any", "each", "into", "out", "off", "over",
    "under", "again", "then", "than", "too", "very", "just", "about", "there",
    "here", "when", "where", "which", "while", "what", "how", "why", "who",
    "them", "they", "their", "its", "it", "is", "in", "on", "at", "to", "of",
    "a", "an", "or", "if", "else", "elif", "i", "me", "my", "mine", "we", "our",
    "us", "he", "she", "his", "her", "him", "be", "been", "being", "do", "does",
    "did", "done", "get", "got", "make", "made", "need", "needs", "want", "wants",
    "like", "know", "think", "see", "say", "said", "use", "using", "used",
    "run", "running", "file", "files", "also", "still", "now", "way", "thing",
    "things", "something", "anything", "everything", "nothing",
}

_TOKEN_RE = re.compile(r"[a-z]{3,}")


def _classify_assert(node: ast.Assert) -> str:
    """按关键词把一条断言谈进某个类别。"""
    text = ast.unparse(node)
    for category, keywords in ASSERT_CATEGORIES.items():
        if any(kw in text for kw in keywords):
            return category
    return "other"


def _iter_asserts(source: str) -> list[ast.Assert]:
    """取出源码里的全部断言语句。"""
    tree = ast.parse(source)
    return [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]


def _extract_keywords(instruction: str) -> set[str]:
    """从题面提取关键词：小写字母 token、长度 ≥ 3、去停用词。"""
    return {
        tok for tok in _TOKEN_RE.findall(instruction.lower())
        if tok not in STOPWORDS
    }


def _load_metadata(task_dir: Path) -> dict[str, Any]:
    """读取 task.toml 的 [metadata] 段；读不到时返回空 dict。"""
    toml_path = task_dir / "task.toml"
    if not toml_path.exists():
        return {}
    try:
        return tomllib.loads(toml_path.read_text(encoding="utf-8")).get("metadata", {}) or {}
    except Exception:
        return {}


def check_task(task_dir: Path) -> TaskAuditResult:
    """对单个任务目录执行全部静态检查。"""
    task_dir = Path(task_dir)
    flags: list[str] = []
    metrics: dict[str, Any] = {}
    decision = Decision.PASS

    def _reject(reason: str) -> None:
        nonlocal decision
        decision = Decision.REJECT
        flags.append(reason)

    def _flag(reason: str) -> None:
        nonlocal decision
        if decision != Decision.REJECT:
            decision = Decision.FLAG
        flags.append(reason)

    # ---- 模式完整性 ----
    missing = [rel for rel in REQUIRED_FILES if not (task_dir / rel).exists()]
    if missing:
        _reject("missing_required_files")
        metrics["missing_files"] = missing
        return TaskAuditResult(task_dir.name, task_dir, decision, metrics, flags)

    # ---- 语法有效性 ----
    init_path = task_dir / "environment" / "test_initial_state.py"
    final_path = task_dir / "tests" / "test_final_state.py"
    init_src = init_path.read_text(encoding="utf-8", errors="replace")
    final_src = final_path.read_text(encoding="utf-8", errors="replace")

    try:
        ast.parse(init_src)
        ast.parse(final_src)
    except SyntaxError:
        _reject("test_syntax_error")
        return TaskAuditResult(task_dir.name, task_dir, decision, metrics, flags)

    # ---- 断言结构与重合度 ----
    asserts = _iter_asserts(final_src)
    categories = [_classify_assert(a) for a in asserts]
    total = len(categories)
    outcome = sum(1 for c in categories if c in OUTCOME_CATEGORIES)
    structural_score = (outcome / total) if total else 0.0

    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8", errors="replace")
    keywords = _extract_keywords(instruction)
    if total and keywords:
        hits = sum(
            1 for a in asserts
            if any(kw in ast.unparse(a).lower() for kw in keywords)
        )
        desc_overlap = hits / total
    else:
        desc_overlap = 0.0

    diversity = len(set(categories))

    metrics.update(
        structural_score=round(structural_score, 4),
        desc_overlap=round(desc_overlap, 4),
        assert_diversity=diversity,
        assert_count=total,
        assert_categories={c: categories.count(c) for c in sorted(set(categories))},
    )

    if structural_score == STRUCTURAL_REJECT:
        _reject("no_outcome_asserts")
    elif structural_score < STRUCTURAL_FLAG:
        _flag("structural_score_low")

    if desc_overlap == 0.0:
        _reject("verifier_disconnected_from_task")

    if diversity < 2:
        _flag("low_assert_diversity")

    # ---- 元数据有效性 ----
    meta = _load_metadata(task_dir)
    difficulty = meta.get("difficulty")
    category = meta.get("category")
    if not difficulty or not category:
        _flag("metadata_missing_fields")
    elif difficulty not in VALID_DIFFICULTIES:
        _flag("metadata_unknown_difficulty")
    metrics["difficulty"] = difficulty
    metrics["category"] = category

    return TaskAuditResult(task_dir.name, task_dir, decision, metrics, flags)


def check_batch(task_dirs: list[Path]) -> list[TaskAuditResult]:
    """对一批任务目录执行静态检查，返回与输入对齐的结果列表。"""
    return [check_task(d) for d in task_dirs]


def dataset_summary(results: list[TaskAuditResult]) -> dict[str, Any]:
    """把单任务结果聚合为数据集级统计。"""
    total = len(results)
    n_pass = sum(1 for r in results if r.decision == Decision.PASS)
    n_flag = sum(1 for r in results if r.decision == Decision.FLAG)
    n_reject = sum(1 for r in results if r.decision == Decision.REJECT)

    scores = [r.metrics.get("structural_score", 0.0) for r in results]
    overlaps = [r.metrics.get("desc_overlap", 0.0) for r in results]

    category_dist: dict[str, int] = {}
    difficulty_dist: dict[str, int] = {}
    for r in results:
        cat = r.metrics.get("category")
        diff = r.metrics.get("difficulty")
        if cat:
            category_dist[cat] = category_dist.get(cat, 0) + 1
        if diff:
            difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1

    return {
        "total": total,
        "pass": n_pass,
        "flag": n_flag,
        "reject": n_reject,
        "mean_structural_score": round(statistics.fmean(scores), 4) if scores else 0.0,
        "median_structural_score": round(statistics.median(scores), 4) if scores else 0.0,
        "mean_desc_overlap": round(statistics.fmean(overlaps), 4) if overlaps else 0.0,
        "category_distribution": dict(sorted(category_dist.items(), key=lambda kv: -kv[1])),
        "difficulty_distribution": dict(sorted(difficulty_dist.items())),
    }
