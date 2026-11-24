"""Data preparation: open-source conversion, type filtering, exact & n-gram Jaccard dedup, and parquet export."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq

FILTERED_PROBLEM_TYPES = {"选择", "选择题", "判断", "判断题", "证明", "证明题", "choice", "true_false", "proof"}

_CHOICE_OPTION_RE = re.compile(r"(?:^|\n|\s)A[\.、．]\s*.+?(?:\n|\s)B[\.、．]\s*", re.DOTALL)
_JUDGE_TYPE_RE = re.compile(r"(?:判断对错|下列说法(?:正确|错误)的是|是否正确[？?]?$)")
_PROOF_RE = re.compile(r"(?:求证|证明\s*[:：]|试证|请证明)")
_NORM_TEXT_RE = re.compile(r"\s+")
_CHAR_OR_WORD_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")

SYSTEM_PROMPT_THINKING = (
    "你是一个数学推理助手。请先在 <think>...</think> 中逐步推导，"
    "再在 </think> 后给出清晰完整的最终解答；若答案为单一数学表达式，请用 \\boxed{...} 标出。"
)


@dataclass
class DataPrepStats:
    """Accounting statistics produced during data preparation."""

    total_input: int = 0
    filtered_by_type: int = 0
    exact_duplicates: int = 0
    fuzzy_duplicates: int = 0
    kept: int = 0
    rule_routed: int = 0
    judge_routed: int = 0


def normalize_question_text(text: str) -> str:
    """Normalize question text for exact deduplication and n-gram shingling."""
    cleaned = text.strip().lower()
    cleaned = _NORM_TEXT_RE.sub("", cleaned)
    for ch in ("，", "。", "？", "！", "；", "：", "、", ",", ".", "?", "!", ";", ":"):
        cleaned = cleaned.replace(ch, "")
    return cleaned


def is_filtered_problem_type(record: dict[str, Any]) -> bool:
    """Return True if the record is multiple-choice, true/false, or proof."""
    raw_type = str(record.get("type", "")).strip().lower()
    if raw_type in FILTERED_PROBLEM_TYPES:
        return True

    question = str(record.get("question", ""))
    if _CHOICE_OPTION_RE.search(question):
        return True
    if _JUDGE_TYPE_RE.search(question):
        return True
    if _PROOF_RE.search(question):
        return True
    return False


def build_shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    """Build token/character n-gram shingles for fuzzy Jaccard deduplication."""
    tokens = _CHAR_OR_WORD_RE.findall(text.lower())
    if not tokens:
        return set()
    if len(tokens) < n:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def ngram_jaccard_similarity(shingles_a: set[tuple[str, ...]], shingles_b: set[tuple[str, ...]]) -> float:
    """Compute Jaccard similarity between two n-gram shingle sets."""
    if not shingles_a or not shingles_b:
        return 0.0
    union_size = len(shingles_a | shingles_b)
    if union_size == 0:
        return 0.0
    return len(shingles_a & shingles_b) / union_size


def convert_open_source_record(raw: dict[str, Any], idx: int, default_source: str = "opensource_math") -> dict[str, Any]:
    """Normalize open-source math JSON/JSONL records into unified project schema."""
    if "subq" in raw and "question" in raw:
        return {
            "qid": str(raw.get("qid") or f"{default_source}-{idx:06d}"),
            "type": str(raw.get("type") or ("解答" if len(raw["subq"]) > 1 else "计算")),
            "question": str(raw["question"]),
            "subq": raw["subq"],
            "source": str(raw.get("source") or default_source),
            "route": raw.get("route"),
        }

    question = str(raw.get("question") or raw.get("problem") or raw.get("prompt") or "")
    answer = str(raw.get("answer") or raw.get("solution") or raw.get("ground_truth") or "")
    problem_type = str(raw.get("type") or raw.get("problem_type") or "计算")
    qid = str(raw.get("qid") or raw.get("id") or f"{default_source}-{idx:06d}")
    return {
        "qid": qid,
        "type": problem_type,
        "question": question,
        "subq": [{"id": 1, "answer": answer}],
        "source": str(raw.get("source") or default_source),
        "route": raw.get("route"),
    }


def resolve_route(record: dict[str, Any]) -> str:
    """Determine ``rule`` vs ``judge`` route for a unified math problem record."""
    if record.get("route") in ("rule", "judge"):
        return str(record["route"])
    subq = record.get("subq") or []
    if len(subq) > 1 or str(record.get("type", "")) == "解答":
        return "judge"
    return "rule"


def deduplicate_and_filter(
    records: list[dict[str, Any]],
    jaccard_threshold: float = 0.85,
    ngram_n: int = 5,
) -> tuple[list[dict[str, Any]], DataPrepStats]:
    """Filter disallowed types and apply exact + n-gram Jaccard fuzzy deduplication."""
    stats = DataPrepStats(total_input=len(records))
    seen_hashes: set[str] = set()
    kept_shingles: list[set[tuple[str, ...]]] = []
    # Inverted index from shingle -> candidate record indices for fast fuzzy lookup.
    shingle_index: dict[tuple[str, ...], list[int]] = {}
    cleaned: list[dict[str, Any]] = []

    for rec in records:
        if is_filtered_problem_type(rec):
            stats.filtered_by_type += 1
            continue

        question = str(rec.get("question", "")).strip()
        if not question:
            stats.filtered_by_type += 1
            continue

        norm = normalize_question_text(question)
        q_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if q_hash in seen_hashes:
            stats.exact_duplicates += 1
            continue

        shingles = build_shingles(question, n=ngram_n)
        candidate_ids: set[int] = set()
        for sh in shingles:
            for cid in shingle_index.get(sh, ()):
                candidate_ids.add(cid)

        is_fuzzy_dup = False
        for cid in candidate_ids:
            if ngram_jaccard_similarity(shingles, kept_shingles[cid]) >= jaccard_threshold:
                is_fuzzy_dup = True
                break

        if is_fuzzy_dup:
            stats.fuzzy_duplicates += 1
            continue

        seen_hashes.add(q_hash)
        new_idx = len(kept_shingles)
        kept_shingles.append(shingles)
        for sh in shingles:
            shingle_index.setdefault(sh, []).append(new_idx)

        route = resolve_route(rec)
        rec_out = dict(rec)
        rec_out["route"] = route
        cleaned.append(rec_out)
        stats.kept += 1
        if route == "rule":
            stats.rule_routed += 1
        else:
            stats.judge_routed += 1

    return cleaned, stats


def to_verl_parquet_row(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Convert a unified record into verl RLHFDataset parquet row schema.

    Heterogeneous nested structures (``subq``) are serialized as JSON strings in ``extra_info``.
    """
    route = resolve_route(record)
    subq = record.get("subq") or []
    first_answer = str(subq[0].get("answer", "")) if subq else ""

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT_THINKING},
        {"role": "user", "content": str(record["question"])},
    ]

    return {
        "prompt": prompt,
        "data_source": str(record.get("source", "k12_math")),
        "ability": "math",
        "reward_model": {
            "style": "rule_or_judge",
            "ground_truth": first_answer if route == "rule" else json.dumps(subq, ensure_ascii=False),
        },
        "extra_info": {
            "index": index,
            "qid": str(record["qid"]),
            "type": str(record.get("type", "计算")),
            "route": route,
            "question": str(record["question"]),
            "subq_json": json.dumps(subq, ensure_ascii=False),
        },
    }


def write_parquet(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write verl-ready rows to a Parquet file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(output_path))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare math reasoning RL dataset in verl parquet format.")
    parser.add_argument("--input", nargs="+", required=True, help="Input JSONL files.")
    parser.add_argument("--output-parquet", type=str, required=True, help="Output parquet path.")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Optional cleaned JSONL output.")
    parser.add_argument("--jaccard-threshold", type=float, default=0.85)
    parser.add_argument("--ngram-n", type=int, default=5)
    parser.add_argument("--default-source", type=str, default="k12_math")
    args = parser.parse_args(argv)

    raw_records: list[dict[str, Any]] = []
    for path in args.input:
        for idx, rec in enumerate(load_jsonl(path)):
            raw_records.append(convert_open_source_record(rec, idx=len(raw_records), default_source=args.default_source))

    cleaned, stats = deduplicate_and_filter(
        raw_records,
        jaccard_threshold=args.jaccard_threshold,
        ngram_n=args.ngram_n,
    )
    parquet_rows = [to_verl_parquet_row(rec, index=i) for i, rec in enumerate(cleaned)]
    write_parquet(parquet_rows, args.output_parquet)

    if args.output_jsonl:
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for rec in cleaned:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "total_input": stats.total_input,
                "filtered_by_type": stats.filtered_by_type,
                "exact_duplicates": stats.exact_duplicates,
                "fuzzy_duplicates": stats.fuzzy_duplicates,
                "kept": stats.kept,
                "rule_routed": stats.rule_routed,
                "judge_routed": stats.judge_routed,
                "output_parquet": args.output_parquet,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
