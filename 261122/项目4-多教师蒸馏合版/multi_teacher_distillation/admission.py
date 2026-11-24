import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import DISTILL_TEACHER_DOMAINS
from .reliability_weighting import compute_admission_multiplier

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


def tokenizer_signature(path: Path) -> dict[str, str]:
    import hashlib

    if not path.is_dir():
        raise ValueError(f"模型目录不存在: {path}")
    signatures = {}
    for name in TOKENIZER_FILES:
        candidate = path / name
        if candidate.exists():
            signatures[name] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if "tokenizer.json" not in signatures and "vocab.json" not in signatures:
        raise ValueError(f"{path} 缺少可比对的 tokenizer 词表文件")
    return signatures


def same_model_source(student: Path, teacher: Path) -> bool:
    student_config = json.loads((student / "config.json").read_text(encoding="utf-8"))
    teacher_config = json.loads((teacher / "config.json").read_text(encoding="utf-8"))
    keys = ("architectures", "model_type", "vocab_size")
    for key in keys:
        student_value = student_config.get(key, student_config.get("text_config", {}).get(key))
        teacher_value = teacher_config.get(key, teacher_config.get("text_config", {}).get(key))
        if student_value != teacher_value or student_value is None:
            return False
    return tokenizer_signature(student) == tokenizer_signature(teacher)


def paired_bootstrap_score_diff_lower_bound(
    start_item_scores: Sequence[float],
    teacher_item_scores: Sequence[float],
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 17,
) -> float:
    """估计教师相对学生起点分差的配对 bootstrap 95% 单侧下界（百分制分值）。"""
    s_arr = np.asarray(start_item_scores, dtype=np.float64)
    t_arr = np.asarray(teacher_item_scores, dtype=np.float64)
    if s_arr.shape != t_arr.shape or s_arr.size == 0:
        raise ValueError("配对 bootstrap 要求起点与教师逐题分数等长且非空")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap 次数必须为正整数")
    scale = 100.0 if max(float(np.max(np.abs(s_arr))), float(np.max(np.abs(t_arr)))) <= 1.0 else 1.0
    diffs = (t_arr - s_arr) * scale
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diffs), size=(bootstrap_samples, len(diffs)))
    boot_means = diffs[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) * 100.0
    return float(np.percentile(boot_means, alpha))


@dataclass(frozen=True)
class AdmissionResult:
    domain: str
    domain_gain: float
    general_drop: float
    same_source_and_tokenizer: bool
    required_gain: float
    maximum_general_drop: float
    score_diff_lower_bound: float
    teacher_reliability_multiplier: float
    admitted: bool


def evaluate_admission(
    domain: str,
    start_score: float,
    teacher_score: float,
    start_general_score: float,
    teacher_general_score: float,
    student_path: Path,
    teacher_model_dir: Path,
    required_gain: float = 3.0,
    maximum_general_drop: float = 1.0,
    score_diff_lower_bound: float | None = None,
    start_item_scores: Sequence[float] | None = None,
    teacher_item_scores: Sequence[float] | None = None,
) -> AdmissionResult:
    values = (start_score, teacher_score, start_general_score, teacher_general_score)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("准入分数必须为有限值")
    gain = float(teacher_score - start_score)
    drop = float(start_general_score - teacher_general_score)
    same_source = same_model_source(student_path, teacher_model_dir)

    if score_diff_lower_bound is not None:
        lb = float(score_diff_lower_bound)
    elif start_item_scores is not None and teacher_item_scores is not None:
        lb = paired_bootstrap_score_diff_lower_bound(start_item_scores, teacher_item_scores)
    else:
        lb = gain

    multiplier = compute_admission_multiplier(lb, lb_scale=2.0)
    admitted = same_source and gain >= required_gain and drop <= maximum_general_drop and lb > 0.0
    return AdmissionResult(
        domain=domain,
        domain_gain=gain,
        general_drop=drop,
        same_source_and_tokenizer=same_source,
        required_gain=required_gain,
        maximum_general_drop=maximum_general_drop,
        score_diff_lower_bound=lb,
        teacher_reliability_multiplier=multiplier,
        admitted=admitted,
    )


def require_admitted(result: AdmissionResult) -> None:
    if not result.admitted:
        raise RuntimeError(f"教师 {result.domain} 未通过同源 tokenizer、领域增益/分差下界或通用回归准入检查")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=list(DISTILL_TEACHER_DOMAINS), required=True)
    parser.add_argument("--start-score", type=float, required=True)
    parser.add_argument("--teacher-score", type=float, required=True)
    parser.add_argument("--start-general-score", type=float, required=True)
    parser.add_argument("--teacher-general-score", type=float, required=True)
    parser.add_argument("--student-path", type=Path, required=True)
    parser.add_argument("--teacher-model-dir", type=Path, required=True)
    parser.add_argument("--score-diff-lb", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate_admission(
        args.domain,
        args.start_score,
        args.teacher_score,
        args.start_general_score,
        args.teacher_general_score,
        args.student_path,
        args.teacher_model_dir,
        score_diff_lower_bound=args.score_diff_lb,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(vars(result), ensure_ascii=False, indent=2), encoding="utf-8")
    require_admitted(result)


if __name__ == "__main__":
    main()
