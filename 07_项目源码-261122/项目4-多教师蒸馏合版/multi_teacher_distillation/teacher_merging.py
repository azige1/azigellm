import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import CANDIDATE_DOMAINS, DEFAULT_REWARD_SOURCES, SELF_ANCHOR_DOMAIN

_DEFAULT_PAIR_NAMES = {
    frozenset({"func_gen", "competitive_coding"}): "code_gen",
    frozenset({"math", "science"}): "math_science",
}


@dataclass(frozen=True)
class MergedTeacherSpec:
    teacher_id: str
    source_domains: list[str]
    reward_source: str
    retrain_required: bool
    split_from: str | None = None
    subclass: str | None = None


@dataclass(frozen=True)
class MergeDecision:
    pair: list[str]
    merged: bool
    reason: str
    reward_sources: list[str]
    forward_drop: float
    backward_drop: float
    forward_band: float
    backward_band: float


@dataclass(frozen=True)
class SplitDecision:
    teacher_id: str
    domain: str
    split: bool
    weak_subclasses: list[str]
    reason: str


def _merged_teacher_name(domains: Sequence[str]) -> str:
    key = frozenset(domains)
    if key in _DEFAULT_PAIR_NAMES:
        return _DEFAULT_PAIR_NAMES[key]
    if len(domains) == 1:
        return str(domains[0])
    return "_".join(sorted(str(item) for item in domains))


def _pair_within_noise_band(
    left: str,
    right: str,
    score_matrix: Mapping[str, Mapping[str, float]],
    noise_bands: Mapping[str, float],
    reward_sources: Mapping[str, str],
    inclusive: bool = False,
) -> MergeDecision:
    left_source = str(reward_sources[left])
    right_source = str(reward_sources[right])
    forward_drop = abs(float(score_matrix[left][right]) - float(score_matrix[right][right]))
    backward_drop = abs(float(score_matrix[right][left]) - float(score_matrix[left][left]))
    forward_band = float(noise_bands[right])
    backward_band = float(noise_bands[left])

    if left_source != right_source:
        return MergeDecision(
            pair=[left, right],
            merged=False,
            reason="reward_source_mismatch",
            reward_sources=[left_source, right_source],
            forward_drop=forward_drop,
            backward_drop=backward_drop,
            forward_band=forward_band,
            backward_band=backward_band,
        )

    if inclusive:
        in_band = forward_drop <= forward_band and backward_drop <= backward_band
    else:
        in_band = forward_drop < forward_band and backward_drop < backward_band

    return MergeDecision(
        pair=[left, right],
        merged=in_band,
        reason="within_noise_band" if in_band else "exceeds_noise_band",
        reward_sources=[left_source, right_source],
        forward_drop=forward_drop,
        backward_drop=backward_drop,
        forward_band=forward_band,
        backward_band=backward_band,
    )


def build_teacher_pool_from_matrix(
    score_matrix: Mapping[str, Mapping[str, float]],
    domain_stds: Mapping[str, float],
    reward_sources: Mapping[str, str] | None = None,
    start_subclass_scores: Mapping[str, Mapping[str, float]] | None = None,
    teacher_subclass_scores: Mapping[str, Mapping[str, float]] | None = None,
    noise_multiplier: float = 2.0,
    inclusive: bool = False,
) -> dict[str, Any]:
    """根据跨域能力矩阵、reward 来源类型与子类验证分执行教师池合并与拆分。"""
    if noise_multiplier <= 0.0:
        raise ValueError("噪声带标准差倍数必须为正")
    domains = list(score_matrix.keys())
    if not domains:
        raise ValueError("能力矩阵不能为空")
    sources = dict(DEFAULT_REWARD_SOURCES)
    if reward_sources:
        sources.update({str(k): str(v) for k, v in reward_sources.items()})

    missing_stds = [d for d in domains if d not in domain_stds]
    if missing_stds:
        raise ValueError(f"缺少领域标准差: {missing_stds}")
    missing_sources = [d for d in domains if d not in sources]
    if missing_sources:
        raise ValueError(f"缺少领域 reward 来源类型: {missing_sources}")
    for row_domain in domains:
        for col_domain in domains:
            if col_domain not in score_matrix[row_domain]:
                raise ValueError(f"能力矩阵缺少 ({row_domain}, {col_domain}) 评测分")

    noise_bands = {d: float(noise_multiplier) * float(domain_stds[d]) for d in domains}

    merge_decisions: list[MergeDecision] = []
    compatible: dict[tuple[str, str], bool] = {}
    for idx, left in enumerate(domains):
        for right in domains[idx + 1 :]:
            decision = _pair_within_noise_band(
                left, right, score_matrix, noise_bands, sources, inclusive=inclusive
            )
            merge_decisions.append(decision)
            compatible[(left, right)] = decision.merged
            compatible[(right, left)] = decision.merged

    assigned: set[str] = set()
    groups: list[list[str]] = []
    for domain in domains:
        if domain in assigned:
            continue
        group = [domain]
        for candidate in domains:
            if candidate in assigned or candidate == domain:
                continue
            if all(compatible[(member, candidate)] for member in group):
                group.append(candidate)
        for member in group:
            assigned.add(member)
        groups.append(group)

    teachers: list[MergedTeacherSpec] = []
    split_decisions: list[SplitDecision] = []
    domain_to_teacher: dict[str, str] = {}

    for group in groups:
        teacher_id = _merged_teacher_name(group)
        reward_source = sources[group[0]]
        retrain_required = len(group) > 1

        split_triggered = False
        if len(group) == 1 and start_subclass_scores and teacher_subclass_scores:
            single_domain = group[0]
            start_subs = start_subclass_scores.get(single_domain, {})
            teacher_subs = teacher_subclass_scores.get(single_domain, {})
            weak_subs = [
                sub
                for sub in sorted(set(start_subs) & set(teacher_subs))
                if float(teacher_subs[sub]) < float(start_subs[sub])
            ]
            if weak_subs and len(teacher_subs) > 1:
                split_triggered = True
                split_decisions.append(
                    SplitDecision(
                        teacher_id=teacher_id,
                        domain=single_domain,
                        split=True,
                        weak_subclasses=weak_subs,
                        reason="subclass_below_start",
                    )
                )
                for sub in sorted(teacher_subs):
                    sub_teacher_id = f"{teacher_id}__{sub}"
                    teachers.append(
                        MergedTeacherSpec(
                            teacher_id=sub_teacher_id,
                            source_domains=[single_domain],
                            reward_source=reward_source,
                            retrain_required=True,
                            split_from=teacher_id,
                            subclass=sub,
                        )
                    )
                    domain_to_teacher[f"{single_domain}:{sub}"] = sub_teacher_id
                domain_to_teacher[single_domain] = f"{teacher_id}__{sorted(teacher_subs)[0]}"
            elif teacher_subs:
                split_decisions.append(
                    SplitDecision(
                        teacher_id=teacher_id,
                        domain=single_domain,
                        split=False,
                        weak_subclasses=[],
                        reason="all_subclasses_above_start",
                    )
                )

        if not split_triggered:
            teachers.append(
                MergedTeacherSpec(
                    teacher_id=teacher_id,
                    source_domains=list(group),
                    reward_source=reward_source,
                    retrain_required=retrain_required,
                )
            )
            for domain in group:
                domain_to_teacher[domain] = teacher_id
            domain_to_teacher[teacher_id] = teacher_id

    domain_to_teacher[SELF_ANCHOR_DOMAIN] = SELF_ANCHOR_DOMAIN

    return {
        "candidate_domains": domains,
        "noise_multiplier": float(noise_multiplier),
        "noise_bands": noise_bands,
        "merged_teacher_total": len(teachers),
        "teachers": [asdict(item) for item in teachers],
        "domain_to_teacher": domain_to_teacher,
        "merge_decisions": [asdict(item) for item in merge_decisions],
        "split_decisions": [asdict(item) for item in split_decisions],
    }


def load_domain_to_teacher(path: Path | None = None) -> dict[str, str]:
    from . import DEFAULT_DOMAIN_TO_TEACHER

    if path is None:
        return dict(DEFAULT_DOMAIN_TO_TEACHER)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "domain_to_teacher" in payload:
        mapping = {str(k): str(v) for k, v in payload["domain_to_teacher"].items()}
    else:
        mapping = {str(k): str(v) for k, v in payload.items()}
    mapping.setdefault(SELF_ANCHOR_DOMAIN, SELF_ANCHOR_DOMAIN)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="4.1 跨域能力矩阵的教师池合并与拆分")
    parser.add_argument("--matrix", type=Path, required=True, help="包含 score_matrix/domain_stds/reward_sources 的 JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise-multiplier", type=float, default=2.0)
    parser.add_argument("--inclusive", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.matrix.read_text(encoding="utf-8"))
    report = build_teacher_pool_from_matrix(
        score_matrix=payload["score_matrix"],
        domain_stds=payload["domain_stds"],
        reward_sources=payload.get("reward_sources", {d: DEFAULT_REWARD_SOURCES[d] for d in CANDIDATE_DOMAINS if d in payload["score_matrix"]}),
        start_subclass_scores=payload.get("start_subclass_scores"),
        teacher_subclass_scores=payload.get("teacher_subclass_scores"),
        noise_multiplier=args.noise_multiplier,
        inclusive=args.inclusive,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
