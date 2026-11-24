import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .. import AGENT_TRAJECTORY_DOMAINS, DEFAULT_DOMAIN_TO_TEACHER, DEFAULT_POOL_SIZES, ROUTED_DOMAINS, SELF_ANCHOR_DOMAIN
from ..routing import validate_routing_stage
from ..teacher_merging import load_domain_to_teacher

CONVERSATION_INSTRUCTIONS = "严格遵循任务指令和交互协议；需要工具时调用对应工具，最终给出要求的动作或答案。"


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def standardize_row(
    record: dict[str, Any],
    source_domain: str,
    index: int,
    domain_to_teacher: Mapping[str, str] | None = None,
    teacher_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    mapping = dict(domain_to_teacher or DEFAULT_DOMAIN_TO_TEACHER)
    raw_domain = str(record.get("domain", source_domain))
    if raw_domain not in mapping and raw_domain not in ROUTED_DOMAINS:
        raise ValueError(f"未知领域: {raw_domain}")

    routed_domain = mapping.get(raw_domain, raw_domain)
    if routed_domain not in ROUTED_DOMAINS:
        raise ValueError(f"领域 {raw_domain} 映射到的目标 {routed_domain} 不在合法路由域中")

    teacher = routed_domain
    teacher_version = str(record.get("teacher_version", (teacher_versions or {}).get(teacher, "")))
    task = dict(record.get("task", record))

    if "prompt" in record or "messages" in record:
        prompt = record.get("prompt", record.get("messages"))
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        prompt = [dict(message) for message in prompt]
    elif "instruction" in record or "instruction" in task:
        prompt = [{"role": "user", "content": str(record.get("instruction", task["instruction"]))}]
    else:
        raise ValueError("样本必须包含 prompt、messages 或 instruction")

    if not isinstance(prompt, list) or not prompt or not any(msg.get("role") == "user" for msg in prompt):
        raise ValueError("prompt 必须包含 user 消息")
    if prompt[0].get("role") != "system":
        prompt = [{"role": "system", "content": CONVERSATION_INSTRUCTIONS}, *prompt]

    task_id = str(record.get("prompt_id", record.get("task_id", f"{routed_domain}_{index}")))
    return {
        "domain": routed_domain,
        "sampling_domain": routed_domain,
        "source_domain": raw_domain,
        "task_type": routed_domain,
        "teacher": teacher,
        "teacher_version": teacher_version,
        "segment_index": int(record.get("segment_index", 0)),
        "data_source": routed_domain,
        "agent_name": routed_domain,
        "verifier_group": "container_sandbox" if routed_domain in AGENT_TRAJECTORY_DOMAINS else "text",
        "__rollout_n__": int(record.get("samples", 1)),
        "prompt": prompt,
        "reward_model": {"style": "rule", "ground_truth": str(record.get("answer", ""))},
        "extra_info": {
            "task_id": task_id,
            "prompt_id": str(record.get("prompt_id", task_id)),
            "domain": routed_domain,
            "source_domain": raw_domain,
            "teacher": teacher,
            "teacher_version": teacher_version,
            "segment_index": int(record.get("segment_index", 0)),
            "task_payload_json": json.dumps(task, ensure_ascii=False),
        },
    }


def build_prompt_pool(
    inputs: Mapping[str, list[dict[str, Any]]],
    seed: int = 17,
    domain_to_teacher: Mapping[str, str] | None = None,
    teacher_versions: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping = dict(domain_to_teacher or DEFAULT_DOMAIN_TO_TEACHER)
    rows: list[dict[str, Any]] = []
    for source_domain, domain_records in inputs.items():
        for index, row in enumerate(domain_records):
            rows.append(
                standardize_row(
                    row,
                    source_domain=source_domain,
                    index=index,
                    domain_to_teacher=mapping,
                    teacher_versions=teacher_versions,
                )
            )
    if not rows:
        raise ValueError("蒸馏 prompt 池不能为空")
    validate_routing_stage("data_build", rows, domain_to_teacher=mapping, strict_mode="raise")
    np.random.default_rng(seed).shuffle(rows)
    counts = {name: sum(row["domain"] == name for row in rows) for name in ROUTED_DOMAINS}
    return rows, {"counts": counts, "total": len(rows)}


def require_default_pool_size(stats: dict[str, Any]) -> None:
    if stats["counts"] != DEFAULT_POOL_SIZES:
        raise ValueError(f"默认 prompt 池必须为 96000 条，分域数量为 {DEFAULT_POOL_SIZES}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True, help="JSON mapping domain -> jsonl/parquet path")
    parser.add_argument("--domain-map", type=Path, default=None, help="4.1 输出的 domain_to_teacher JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--require-default-size", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    domain_to_teacher = load_domain_to_teacher(args.domain_map)
    inputs = {str(name): read_records(Path(path)) for name, path in manifest.items()}
    rows, stats = build_prompt_pool(inputs, seed=args.seed, domain_to_teacher=domain_to_teacher)
    if args.require_default_size:
        require_default_pool_size(stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
