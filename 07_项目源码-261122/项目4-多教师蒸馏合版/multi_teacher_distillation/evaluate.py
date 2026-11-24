import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from . import EVAL_JUDGE_MODEL, DISTILL_TEACHER_DOMAINS
from .domains.verifiers import verifier_for
from .metrics import summarize_recovery


def records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def task(record: dict[str, Any]) -> dict[str, Any]:
    extra = record.get("extra_info", {})
    value = extra.get("task_payload_json", record.get("task", record))
    return json.loads(value) if isinstance(value, str) else dict(value)


async def chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> str:
    response = await client.post(
        base_url.rstrip("/") + "/chat/completions",
        json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"] or ""


async def run_domain_evaluation(args) -> dict[str, Any]:
    selected = records(args.input)[: args.limit]
    semaphore = asyncio.Semaphore(args.concurrency)
    config = SimpleNamespace(
        judge=SimpleNamespace(base_url=args.judge_base_url, model=args.judge_model, timeout=args.timeout)
    )
    verifier = verifier_for(args.domain, config)
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        async def one(record: dict[str, Any]):
            async with semaphore:
                if args.domain in {"repo_impl", "issue_fix"}:
                    raise RuntimeError(f"{args.domain} 评测需在沙箱容器执行器中运行")
                text = await chat(client, args.base_url, args.model, list(record["prompt"]), args.max_tokens, args.temperature)
                return await verifier(task(record), text)

        results = await asyncio.gather(*(one(record) for record in selected for _ in range(args.samples)))

    item_scores = [
        sum(float(result.score) for result in results[index * args.samples : (index + 1) * args.samples]) / args.samples
        for index in range(len(selected))
    ]
    return {
        "domain": args.domain,
        "task_count": len(selected),
        "samples_per_prompt": args.samples,
        "count": len(results),
        "score": 100.0 * sum(result.score for result in results) / len(results) if results else 0.0,
        "solved_share": sum(result.passed for result in results) / len(results) if results else 0.0,
        "item_scores": item_scores,
    }


def result_table(result: dict[str, Any]) -> str:
    lines = ["| domain | start | teacher | mixed_rl | student | recovery |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in result["domains"].items():
        mixed = result.get("mixed_rl", {}).get(name)
        lines.append(
            f"| {name} | {row['start']:.2f} | {row['teacher']:.2f} | "
            f"{mixed:.2f} | {row['student']:.2f} | {100 * row['recovery']:.1f}% |"
            if mixed is not None
            else f"| {name} | {row['start']:.2f} | {row['teacher']:.2f} | — | {row['student']:.2f} | {100 * row['recovery']:.1f}% |"
        )
    return "\n".join(lines)


def aggregate_results(
    inputs: list[Path],
    output: Path,
    start_name: str,
    student_name: str,
    mixed_rl_name: str | None = None,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, float]] = {}
    for path in inputs:
        value = json.loads(path.read_text(encoding="utf-8"))
        model = value["model_name"]
        domain = value.get("domain", value.get("task_type"))
        grouped.setdefault(model, {})[domain] = float(value["score"])
    teachers = {name: grouped[f"teacher_{name}"][name] for name in grouped[start_name]}
    mixed_rl = grouped.get(mixed_rl_name) if mixed_rl_name else None
    result = summarize_recovery(grouped[start_name], teachers, grouped[student_name], mixed_rl)
    result["table_markdown"] = result_table(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    text = subparsers.add_parser("text")
    text.add_argument("--domain", choices=list(DISTILL_TEACHER_DOMAINS), required=True)
    text.add_argument("--input", type=Path, required=True)
    text.add_argument("--output", type=Path, required=True)
    text.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    text.add_argument("--model", default="student")
    text.add_argument("--limit", type=int, required=True)
    text.add_argument("--max-tokens", type=int, default=16384)
    text.add_argument("--samples", type=int, default=1)
    text.add_argument("--temperature", type=float, default=0.6)
    text.add_argument("--concurrency", type=int, default=8)
    text.add_argument("--timeout", type=float, default=600)
    text.add_argument("--judge-base-url", default="")
    text.add_argument("--judge-model", default=EVAL_JUDGE_MODEL)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--inputs", type=Path, nargs="+", required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--start-name", default="start")
    summary.add_argument("--student-name", default="student")
    summary.add_argument("--mixed-rl-name", default=None)

    args = parser.parse_args()
    if args.command == "text":
        if args.samples < 1:
            raise ValueError("每题采样次数必须至少为 1")
        result = asyncio.run(run_domain_evaluation(args))
        result["model_name"] = args.model
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        aggregate_results(args.inputs, args.output, args.start_name, args.student_name, args.mixed_rl_name)


if __name__ == "__main__":
    main()
