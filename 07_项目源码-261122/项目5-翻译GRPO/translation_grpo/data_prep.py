"""Data preparation script for balanced Zh-En / En-Zh translation RL datasets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import datasets

from .reward_scoring import count_text_units

VALID_DIRECTIONS = {"zh2en", "en2zh"}


def build_translation_messages(source: str, direction: str) -> list[dict[str, str]]:
    """Build chat messages for bidirectional Zh-En translation."""
    if direction == "zh2en":
        instruction = (
            "Translate the following Chinese text into accurate, fluent, and idiomatic English. "
            "Preserve all domain terminology and output only the translation."
        )
    elif direction == "en2zh":
        instruction = (
            "将下面的英文准确、流畅、地道地翻译为中文，保持专业术语一致，只输出译文。"
        )
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": source.strip()},
    ]


def load_parallel_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    """Load and validate raw parallel jsonl records keyed by sample_id."""
    records_by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                item = json.loads(line)
                sample_id = str(item["sample_id"]).strip()
                direction = str(item["direction"]).strip()
                source = str(item["source"]).strip()
                reference = str(item["reference"]).strip()
                if direction not in VALID_DIRECTIONS:
                    raise ValueError(f"{path}:{line_no}: invalid direction '{direction}'")
                if not sample_id or not source or not reference:
                    raise ValueError(f"{path}:{line_no}: empty sample_id/source/reference")

                source_len = int(item.get("source_len") or count_text_units(source))
                ref_len = int(item.get("reference_len") or count_text_units(reference))
                records_by_id[sample_id] = {
                    "sample_id": sample_id,
                    "direction": direction,
                    "source": source,
                    "reference": reference,
                    "source_len": source_len,
                    "reference_len": ref_len,
                    "domain": str(item.get("domain", "general")),
                }
    return list(records_by_id.values())


def balance_directions(
    records: list[dict[str, Any]],
    *,
    max_per_direction: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Balance zh2en and en2zh records 1:1 and interleave deterministically."""
    zh2en = sorted([r for r in records if r["direction"] == "zh2en"], key=lambda x: x["sample_id"])
    en2zh = sorted([r for r in records if r["direction"] == "en2zh"], key=lambda x: x["sample_id"])

    rng = random.Random(seed)
    rng.shuffle(zh2en)
    rng.shuffle(en2zh)

    count = min(len(zh2en), len(en2zh))
    if max_per_direction is not None and max_per_direction > 0:
        count = min(count, max_per_direction)

    balanced: list[dict[str, Any]] = []
    for i in range(count):
        balanced.append(zh2en[i])
        balanced.append(en2zh[i])
    return balanced


def format_verl_record(record: dict[str, Any], index: int, split: str) -> dict[str, Any]:
    """Convert a parallel record into verl RLHFDataset schema."""
    sample_id = record["sample_id"]
    direction = record["direction"]
    source = record["source"]
    reference = record["reference"]
    return {
        "data_source": f"translation_{direction}",
        "prompt": build_translation_messages(source, direction),
        "ability": "translation",
        "reward_model": {"style": "rule", "ground_truth": reference},
        "extra_info": {
            "index": index,
            "split": split,
            "sample_id": sample_id,
            "direction": direction,
            "source": source,
            "reference": reference,
            "source_len": int(record["source_len"]),
            "reference_len": int(record["reference_len"]),
            "domain": record.get("domain", "general"),
        },
    }


def prepare_rl_datasets(
    input_paths: list[Path],
    output_dir: Path,
    *,
    val_ratio: float = 0.1,
    max_per_direction: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Build balanced train/val parquet and jsonl datasets aligned by sample_id."""
    raw_records = load_parallel_jsonl(input_paths)
    balanced = balance_directions(raw_records, max_per_direction=max_per_direction, seed=seed)
    if not balanced:
        raise ValueError("No balanced records produced from inputs.")

    n_total = len(balanced)
    n_val = max(2, int(round(n_total * val_ratio))) if n_total >= 4 else 1
    n_val = min(n_val, n_total - 1) if n_total > 1 else 0
    val_raw = balanced[:n_val]
    train_raw = balanced[n_val:] if n_val < n_total else balanced

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"total_balanced": n_total, "splits": {}}

    for split_name, split_records in (("train", train_raw), ("val", val_raw)):
        formatted = [format_verl_record(r, idx, split_name) for idx, r in enumerate(split_records)]
        parquet_path = output_dir / f"{split_name}.parquet"
        jsonl_path = output_dir / f"{split_name}.jsonl"

        ds = datasets.Dataset.from_list(formatted)
        ds.to_parquet(str(parquet_path))
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in formatted:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        zh_cnt = sum(1 for r in split_records if r["direction"] == "zh2en")
        en_cnt = sum(1 for r in split_records if r["direction"] == "en2zh")
        summary["splits"][split_name] = {
            "count": len(formatted),
            "zh2en": zh_cnt,
            "en2zh": en_cnt,
            "parquet": str(parquet_path),
            "jsonl": str(jsonl_path),
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare balanced Zh-En RL translation datasets for verl.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input jsonl files containing sample_id, direction, source, reference.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for train/val parquet and jsonl.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--max-per-direction", type=int, default=None, help="Optional cap per direction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    summary = prepare_rl_datasets(
        [Path(p) for p in args.inputs],
        Path(args.output_dir),
        val_ratio=args.val_ratio,
        max_per_direction=args.max_per_direction,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
