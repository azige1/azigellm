"""Calibration script for 300-item subquestion-level judge false positive rate (6.1)."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from math_reasoning_rl.config import RewardRoutingConfig
from math_reasoning_rl.judge_client import (
    OpenAICompatJudgeClient,
    build_whole_problem_judge_prompt,
    parse_judge_response,
)


@dataclass
class CalibrationMetrics:
    """Summary metrics for subquestion-level judge calibration."""

    strategy: str
    total_subquestions: int
    tp: int
    fp: int
    tn: int
    fn: int
    fpr_over_total: float
    fpr_conditional: float
    accuracy: float
    precision: float
    recall: float


def compute_calibration_metrics(
    records: list[dict[str, Any]],
    pred_verdicts: list[int],
    strategy: str,
) -> CalibrationMetrics:
    """Compute FPR (both FP/total per 6.1 and FP/(FP+TN)) plus accuracy/precision/recall."""
    if len(records) != len(pred_verdicts):
        raise ValueError(f"Length mismatch: {len(records)} records vs {len(pred_verdicts)} predictions")
    total = len(records)
    tp = fp = tn = fn = 0
    for rec, pred in zip(records, pred_verdicts, strict=True):
        gold = 1 if int(rec["gold_verdict"]) == 1 else 0
        pred_bit = 1 if int(pred) == 1 else 0
        if gold == 1 and pred_bit == 1:
            tp += 1
        elif gold == 0 and pred_bit == 1:
            fp += 1
        elif gold == 0 and pred_bit == 0:
            tn += 1
        else:
            fn += 1

    fpr_total = fp / total if total > 0 else 0.0
    neg = fp + tn
    fpr_cond = fp / neg if neg > 0 else 0.0
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return CalibrationMetrics(
        strategy=strategy,
        total_subquestions=total,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        fpr_over_total=fpr_total,
        fpr_conditional=fpr_cond,
        accuracy=acc,
        precision=prec,
        recall=rec_rate,
    )


def load_gold_records(path: str | Path) -> list[dict[str, Any]]:
    """Load subquestion-level gold annotations from a JSONL file."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


async def evaluate_strategy_async(
    records: list[dict[str, Any]],
    client: OpenAICompatJudgeClient,
    strategy: str,
    concurrency: int = 16,
) -> tuple[CalibrationMetrics, list[dict[str, Any]]]:
    """Run one judge strategy across all subquestion records and return metrics + details."""
    semaphore = asyncio.Semaphore(concurrency)
    details: list[dict[str, Any]] = [{} for _ in range(len(records))]
    preds = [0] * len(records)

    async def _eval_record(idx: int, rec: dict[str, Any]) -> None:
        async with semaphore:
            question = str(rec["question"])
            student_response = str(rec["student_response"])
            subq_item = {
                "id": rec.get("subq_id", 1),
                "answer": rec["std_answer"],
                "key_step": rec.get("key_step"),
            }
            if strategy == "whole_problem_once":
                all_subq = rec.get("all_subq") or [subq_item]
                prompt = build_whole_problem_judge_prompt(question, all_subq, student_response)
                raw = await client.judge_prompt_async(prompt)
                verdict_obj = parse_judge_response(raw, subq_id=subq_item["id"])
            elif strategy in ("subq_direct", "subq_extract_then_judge"):
                prompt_mode = "direct" if strategy == "subq_direct" else "extract_then_judge"
                verdicts = await client.judge_subquestions_async(
                    question=question,
                    subq_items=[subq_item],
                    student_response=student_response,
                    strategy=prompt_mode,
                )
                verdict_obj = verdicts[0]
            else:
                raise ValueError(f"Unsupported judge strategy: {strategy}")

            preds[idx] = verdict_obj.verdict
            details[idx] = {
                "qid": rec.get("qid"),
                "subq_id": rec.get("subq_id", 1),
                "gold_verdict": rec["gold_verdict"],
                "pred_verdict": verdict_obj.verdict,
                "extracted_answer": verdict_obj.extracted_answer,
                "reason": verdict_obj.reason,
            }

    await asyncio.gather(*(_eval_record(i, rec) for i, rec in enumerate(records)))
    metrics = compute_calibration_metrics(records, preds, strategy=strategy)
    return metrics, details


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate subquestion-level judge FPR on golden set.")
    parser.add_argument("--gold-file", type=str, required=True, help="Path to subquestion gold JSONL.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["whole_problem_once", "subq_direct", "subq_extract_then_judge"],
        choices=["whole_problem_once", "subq_direct", "subq_extract_then_judge"],
    )
    parser.add_argument("--judge-base-url", type=str, default="http://127.0.0.1:8001/v1")
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--judge-api-key", type=str, default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args(argv)

    records = load_gold_records(args.gold_file)
    cfg = RewardRoutingConfig(
        judge_base_url=args.judge_base_url,
        judge_model=args.judge_model,
        judge_api_key=args.judge_api_key,
    )
    client = OpenAICompatJudgeClient(cfg)

    async def _run_all() -> dict[str, Any]:
        summary: dict[str, Any] = {"total_records": len(records), "strategies": {}}
        for strategy in args.strategies:
            metrics, details = await evaluate_strategy_async(
                records=records,
                client=client,
                strategy=strategy,
                concurrency=args.concurrency,
            )
            summary["strategies"][strategy] = {
                "tp": metrics.tp,
                "fp": metrics.fp,
                "tn": metrics.tn,
                "fn": metrics.fn,
                "fpr_over_total": round(metrics.fpr_over_total, 4),
                "fpr_conditional": round(metrics.fpr_conditional, 4),
                "accuracy": round(metrics.accuracy, 4),
                "precision": round(metrics.precision, 4),
                "recall": round(metrics.recall, 4),
                "details": details,
            }
        return summary

    results = asyncio.run(_run_all())
    print(json.dumps({k: v for k, v in results.items() if k != "details"}, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
