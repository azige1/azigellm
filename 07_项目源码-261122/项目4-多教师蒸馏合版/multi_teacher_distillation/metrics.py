from typing import Any

import numpy as np


def recovery_rate(current: float, teacher: float, start: float) -> float:
    if teacher <= start:
        raise ValueError("恢复率要求教师相对起点有正增益")
    return (current - start) / (teacher - start)


def token_gap(student_logprobs: list[float], teacher_logprobs: list[float], mask: list[int]) -> float:
    student = np.asarray(student_logprobs, dtype=np.float64)
    teacher = np.asarray(teacher_logprobs, dtype=np.float64)
    weights = np.asarray(mask, dtype=np.float64)
    if student.shape != teacher.shape or student.shape != weights.shape:
        raise ValueError("log-prob 与 mask 必须等长")
    if weights.sum() == 0:
        raise ValueError("mask 没有有效 token")
    return float(((student - teacher) * weights).sum() / weights.sum())


def domain_mean(scores: dict[str, float]) -> float:
    if not scores:
        raise ValueError("至少需要一个领域分数")
    return float(np.mean(list(scores.values())))


def summarize_recovery(
    start: dict[str, float],
    teacher: dict[str, float],
    student: dict[str, float],
    mixed_rl: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not set(start) == set(teacher) == set(student) or not start:
        raise ValueError("三组模型必须覆盖相同且非空的领域")
    rates = {name: recovery_rate(student[name], teacher[name], start[name]) for name in start}
    result: dict[str, Any] = {
        "domains": {
            name: {
                "start": start[name],
                "teacher": teacher[name],
                "student": student[name],
                "recovery": rates[name],
            }
            for name in start
        },
        "start_mean": domain_mean(start),
        "teacher_mean": domain_mean(teacher),
        "student_mean": domain_mean(student),
        "mean_recovery": domain_mean(rates),
        "weakest_domain": min(rates, key=rates.get),
        "weakest_recovery": min(rates.values()),
    }
    if mixed_rl is not None:
        if set(mixed_rl) != set(start):
            raise ValueError("数据混合统一 RL 对照必须覆盖相同领域")
        result["mixed_rl"] = mixed_rl
        result["mixed_rl_mean"] = domain_mean(mixed_rl)
        result["improvement_over_mixed_rl"] = result["student_mean"] - result["mixed_rl_mean"]
    return result
