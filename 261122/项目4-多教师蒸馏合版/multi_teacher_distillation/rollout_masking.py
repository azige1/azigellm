from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def is_truncated(response_ids: list[int], generation_limit: int, terminal_token_ids: set[int]) -> bool:
    return len(response_ids) >= generation_limit and (not response_ids or response_ids[-1] not in terminal_token_ids)


def rollout_loss_mask(response_ids: list[int], generation_limit: int, terminal_token_ids: set[int]) -> list[int]:
    truncated = is_truncated(response_ids, generation_limit, terminal_token_ids)
    return [0 if truncated else 1] * len(response_ids)


def repeated_ngram_fraction(token_ids: list[int], n: int = 4) -> float:
    if n <= 0:
        raise ValueError("n-gram 的 n 必须为正")
    if len(token_ids) < n:
        return 0.0
    grams = [tuple(token_ids[index : index + n]) for index in range(len(token_ids) - n + 1)]
    unique = len(set(grams))
    return float((len(grams) - unique) / len(grams))


def summarize_rollouts(rows: list[dict[str, Any]], warning_threshold: float = 0.05) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["domain"])].append(row)
    domains = {}
    for domain, values in grouped.items():
        count = len(values)
        truncation_rate = sum(bool(row["truncated"]) for row in values) / count
        domains[domain] = {
            "count": count,
            "truncation_rate": truncation_rate,
            "mean_length": float(np.mean([len(row["response_ids"]) for row in values])),
            "repeated_ngram_fraction": float(np.mean([repeated_ngram_fraction(row["response_ids"]) for row in values])),
            "warning": truncation_rate > warning_threshold,
        }
    return {"domains": domains, "warning_domains": [name for name, value in domains.items() if value["warning"]]}


def advantage_deciles(rows: list[dict[str, Any]], groups: int = 10) -> list[dict[str, Any]]:
    if groups != 10:
        raise ValueError("分组诊断固定为 10 组")
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: float(row["mean_advantage"]))
    result = []
    for group, indices in enumerate(np.array_split(np.arange(len(ordered)), groups)):
        selected = [ordered[int(index)] for index in indices]
        result.append({
            "group": group,
            "count": len(selected),
            "mean_advantage": float(np.mean([row["mean_advantage"] for row in selected])),
            "accuracy": float(np.mean([float(row["correct"]) for row in selected])),
            "mean_length": float(np.mean([row["length"] for row in selected])),
            "length_limit_rate": float(np.mean([float(row["at_limit"]) for row in selected])),
            "repetition_rate": float(np.mean([row["repetition_rate"] for row in selected])),
        })
    return result


def clipped_advantage(
    teacher: Iterable[float],
    student: Iterable[float],
    clip_min: float = -5.0,
    clip_max: float = 5.0,
) -> np.ndarray:
    teacher_values = np.asarray(list(teacher), dtype=np.float32)
    student_values = np.asarray(list(student), dtype=np.float32)
    if teacher_values.shape != student_values.shape:
        raise ValueError("教师与学生 log-prob 必须等长")
    return np.clip(teacher_values - student_values, clip_min, clip_max).astype(np.float32)
