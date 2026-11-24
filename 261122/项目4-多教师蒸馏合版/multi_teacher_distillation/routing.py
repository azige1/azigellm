from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import DEFAULT_DOMAIN_TO_TEACHER, ROUTED_DOMAINS, SELF_ANCHOR_DOMAIN

VALID_ROUTED_DOMAINS = frozenset(ROUTED_DOMAINS)
ROUTING_STAGES = ("data_build", "rollout_output", "teacher_request", "pre_loss")


@dataclass(frozen=True)
class RoutingMetadata:
    domain: str
    teacher: str
    teacher_version: str
    segment_index: int


def extract_routing_metadata(row: Mapping[str, Any]) -> RoutingMetadata:
    extra_fields = row.get("extra_fields") if isinstance(row.get("extra_fields"), Mapping) else {}
    extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), Mapping) else {}

    domain = row.get("domain") or extra_fields.get("domain") or extra_info.get("domain")
    teacher = row.get("teacher") or extra_fields.get("teacher") or extra_info.get("teacher")
    teacher_version = (
        row.get("teacher_version")
        or extra_fields.get("teacher_version")
        or extra_info.get("teacher_version")
        or ""
    )
    seg_raw = (
        row.get("segment_index")
        if "segment_index" in row
        else extra_fields.get("segment_index", extra_info.get("segment_index", 0))
    )
    if domain is None or str(domain).strip() == "":
        raise ValueError("样本缺少 domain 路由字段，拒绝默认路由")
    if teacher is None or str(teacher).strip() == "":
        raise ValueError("样本缺少 teacher 路由字段，拒绝默认路由")
    try:
        segment_index = int(seg_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"非法的 segment_index: {seg_raw!r}") from exc
    return RoutingMetadata(
        domain=str(domain),
        teacher=str(teacher),
        teacher_version=str(teacher_version),
        segment_index=segment_index,
    )


def propagate_segment_metadata(
    segment_extra: dict[str, Any],
    parent_meta: Mapping[str, Any],
    segment_index: int,
) -> dict[str, Any]:
    """轨迹分段时显式拷贝 domain、teacher 与 teacher_version 并写入 segment_index。"""
    if segment_index < 0:
        raise ValueError(f"segment_index 不能为负: {segment_index}")
    parent_info = parent_meta.get("extra_info") if isinstance(parent_meta.get("extra_info"), Mapping) else {}
    parent_fields = parent_meta.get("extra_fields") if isinstance(parent_meta.get("extra_fields"), Mapping) else {}

    domain = parent_meta.get("domain") or parent_fields.get("domain") or parent_info.get("domain")
    teacher = parent_meta.get("teacher") or parent_fields.get("teacher") or parent_info.get("teacher")
    teacher_version = (
        parent_meta.get("teacher_version")
        or parent_fields.get("teacher_version")
        or parent_info.get("teacher_version")
        or ""
    )
    if not domain or str(domain) not in VALID_ROUTED_DOMAINS:
        raise ValueError(f"分段父样本 domain 无效: {domain!r}")
    if not teacher:
        raise ValueError("分段父样本缺少 teacher 字段")

    updated = dict(segment_extra)
    updated["domain"] = str(domain)
    updated["teacher"] = str(teacher)
    updated["teacher_version"] = str(teacher_version)
    updated["segment_index"] = int(segment_index)
    return updated


def validate_routing_stage(
    stage: str,
    records: Sequence[Mapping[str, Any]],
    domain_to_teacher: Mapping[str, str] | None = None,
    expected_teacher_versions: Mapping[str, str] | None = None,
    strict_mode: str = "raise",
    require_teacher_version: bool = False,
) -> dict[str, Any]:
    """在数据构建、rollout 输出、教师请求、loss 前四级校验路由一致性。"""
    if stage not in ROUTING_STAGES:
        raise ValueError(f"未知路由校验阶段: {stage}")
    if strict_mode not in {"raise", "drop"}:
        raise ValueError(f"未知路由校验模式: {strict_mode}")

    mapping = dict(domain_to_teacher or DEFAULT_DOMAIN_TO_TEACHER)
    valid_indices: list[int] = []
    dropped_indices: list[int] = []
    reasons: dict[str, int] = defaultdict(int)

    for idx, row in enumerate(records):
        error_reason: str | None = None
        try:
            meta = extract_routing_metadata(row)
            if meta.domain not in VALID_ROUTED_DOMAINS:
                error_reason = f"invalid_domain:{meta.domain}"
            elif meta.segment_index < 0:
                error_reason = "negative_segment_index"
            else:
                expected_teacher = mapping.get(meta.domain)
                if expected_teacher is None or meta.teacher != expected_teacher:
                    error_reason = f"teacher_mismatch:{meta.domain}->{meta.teacher}!=expected:{expected_teacher}"
                elif require_teacher_version and meta.teacher != SELF_ANCHOR_DOMAIN and not meta.teacher_version:
                    error_reason = "missing_teacher_version"
                elif (
                    expected_teacher_versions
                    and meta.teacher != SELF_ANCHOR_DOMAIN
                    and meta.teacher in expected_teacher_versions
                    and meta.teacher_version
                    and meta.teacher_version != str(expected_teacher_versions[meta.teacher])
                ):
                    error_reason = "teacher_version_mismatch"
        except ValueError as exc:
            error_reason = str(exc)

        if error_reason is not None:
            reasons[error_reason] += 1
            if strict_mode == "raise":
                raise RuntimeError(f"[{stage}] 路由校验失败 (sample={idx}): {error_reason}")
            dropped_indices.append(idx)
        else:
            valid_indices.append(idx)

    return {
        "stage": stage,
        "total": len(records),
        "valid_count": len(valid_indices),
        "dropped_count": len(dropped_indices),
        "valid_indices": valid_indices,
        "dropped_indices": dropped_indices,
        "reasons": dict(reasons),
    }


def filter_cached_logprobs_by_version(
    records: Sequence[dict[str, Any]],
    active_teacher_versions: Mapping[str, str],
) -> dict[str, Any]:
    """教师版本或校准权重变化时，丢弃版本不一致样本的缓存 log-prob 并标记重算。"""
    recomputed_indices: list[int] = []
    cache_hit_indices: list[int] = []
    for idx, row in enumerate(records):
        meta = extract_routing_metadata(row)
        if meta.teacher == SELF_ANCHOR_DOMAIN:
            continue
        active_version = str(active_teacher_versions.get(meta.teacher, ""))
        cached_version = str(row.get("cached_teacher_version", meta.teacher_version))
        has_cached = row.get("cached_teacher_logprobs") is not None
        if has_cached and active_version and cached_version == active_version:
            cache_hit_indices.append(idx)
        else:
            if has_cached and cached_version != active_version:
                row.pop("cached_teacher_logprobs", None)
            row["needs_logprob_recompute"] = True
            if active_version:
                row["teacher_version"] = active_version
            recomputed_indices.append(idx)
    return {
        "cache_hits": len(cache_hit_indices),
        "recomputed": len(recomputed_indices),
        "recomputed_indices": recomputed_indices,
    }


def check_routing_share_drift(
    routed_token_counts: Mapping[str, int | float],
    target_shares: Mapping[str, float],
    drift_threshold: float = 0.01,
) -> dict[str, Any]:
    """对比各领域实际路由 token 份额与目标份额，偏差超过 1% 时触发告警。"""
    total_tokens = float(sum(float(v) for v in routed_token_counts.values()))
    target_norm = float(sum(float(v) for v in target_shares.values()))
    actual_shares: dict[str, float] = {}
    drifts: dict[str, float] = {}
    alarm_domains: list[str] = []

    for domain in sorted(set(target_shares) | set(routed_token_counts)):
        actual = float(routed_token_counts.get(domain, 0.0)) / total_tokens if total_tokens > 0.0 else 0.0
        target = float(target_shares.get(domain, 0.0)) / target_norm if target_norm > 0.0 else 0.0
        drift = actual - target
        actual_shares[domain] = actual
        drifts[domain] = drift
        if total_tokens > 0.0 and abs(drift) > drift_threshold:
            alarm_domains.append(domain)

    return {
        "total_tokens": total_tokens,
        "actual_shares": actual_shares,
        "drifts": drifts,
        "drift_threshold": float(drift_threshold),
        "alarm": len(alarm_domains) > 0,
        "alarm_domains": alarm_domains,
    }
