import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from . import SELF_ANCHOR_DOMAIN


@dataclass(frozen=True)
class CalibrationBucketTable:
    teacher_id: str
    bin_edges: list[float]
    bucket_lifts: list[float]
    bucket_reliabilities: list[float]
    bucket_weights: list[float]
    overall_pass_rate: float
    admission_multiplier: float
    alpha: float
    lift_cap: float
    version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationBucketTable":
        return cls(
            teacher_id=str(payload["teacher_id"]),
            bin_edges=[float(x) for x in payload["bin_edges"]],
            bucket_lifts=[float(x) for x in payload["bucket_lifts"]],
            bucket_reliabilities=[float(x) for x in payload["bucket_reliabilities"]],
            bucket_weights=[float(x) for x in payload["bucket_weights"]],
            overall_pass_rate=float(payload["overall_pass_rate"]),
            admission_multiplier=float(payload["admission_multiplier"]),
            alpha=float(payload.get("alpha", 2.0)),
            lift_cap=float(payload.get("lift_cap", 1.5)),
            version=str(payload.get("version", "v1")),
        )


def compute_admission_multiplier(admission_lb: float, lb_scale: float = 2.0) -> float:
    if lb_scale <= 0.0:
        raise ValueError("lb_scale 必须为正")
    return float(min(1.0, max(0.0, float(admission_lb)) / float(lb_scale)))


def fit_calibration_table(
    teacher_id: str,
    shifts: Sequence[float],
    passed_flags: Sequence[bool | int | float],
    num_bins: int = 10,
    alpha: float = 2.0,
    lift_cap: float = 1.5,
    admission_lb: float = 2.0,
    lb_scale: float = 2.0,
    version: str = "v1",
) -> CalibrationBucketTable:
    """按教师相对冻结学生起点的 log-prob 位移分 10 桶并统计提升倍数与可靠性权重。"""
    if num_bins < 2:
        raise ValueError("分桶数必须至少为 2")
    if lift_cap <= 0.0 or alpha <= 0.0:
        raise ValueError("lift_cap 与 alpha 必须为正")
    shift_arr = np.asarray(shifts, dtype=np.float64)
    pass_arr = np.asarray(passed_flags, dtype=np.float64)
    if shift_arr.shape != pass_arr.shape or shift_arr.size == 0:
        raise ValueError("校准集位移与通过标签必须等长且非空")

    overall_pass = float(np.mean(pass_arr))
    edges = np.quantile(shift_arr, np.linspace(0.0, 1.0, num_bins + 1)).astype(np.float64)
    inner_edges = edges[1:-1]
    bucket_ids = np.searchsorted(inner_edges, shift_arr, side="right")

    bucket_lifts: list[float] = []
    bucket_reliabilities: list[float] = []
    bucket_weights: list[float] = []

    for bucket in range(num_bins):
        in_bucket = bucket_ids == bucket
        if not np.any(in_bucket) or overall_pass <= 0.0:
            lift = 1.0
        else:
            p_pass_given_b = float(np.mean(pass_arr[in_bucket]))
            lift = p_pass_given_b / overall_pass
        reliability = min(max(lift, 0.0), float(lift_cap)) / float(lift_cap)
        weight = reliability ** float(alpha)
        bucket_lifts.append(float(lift))
        bucket_reliabilities.append(float(reliability))
        bucket_weights.append(float(weight))

    return CalibrationBucketTable(
        teacher_id=str(teacher_id),
        bin_edges=[float(x) for x in edges.tolist()],
        bucket_lifts=bucket_lifts,
        bucket_reliabilities=bucket_reliabilities,
        bucket_weights=bucket_weights,
        overall_pass_rate=overall_pass,
        admission_multiplier=compute_admission_multiplier(admission_lb, lb_scale),
        alpha=float(alpha),
        lift_cap=float(lift_cap),
        version=str(version),
    )


def score_token_reliability_weights(
    shifts: Sequence[float] | np.ndarray,
    table: CalibrationBucketTable,
    response_mask: Sequence[bool | int] | np.ndarray | None = None,
    normalize_within_domain: bool = True,
    apply_teacher_multiplier: bool = True,
) -> np.ndarray:
    """将位移序列映射为逐 token 可靠性权重（可选域内均值归一到 1 并乘教师级系数）。"""
    shift_arr = np.asarray(shifts, dtype=np.float64)
    mask = np.ones(shift_arr.shape, dtype=bool) if response_mask is None else np.asarray(response_mask, dtype=bool)
    if shift_arr.shape != mask.shape:
        raise ValueError("位移与 response_mask 必须等形")

    inner_edges = np.asarray(table.bin_edges[1:-1], dtype=np.float64)
    bucket_ids = np.searchsorted(inner_edges, shift_arr, side="right")
    bucket_table = np.asarray(table.bucket_weights, dtype=np.float64)
    weights = bucket_table[bucket_ids]
    weights = np.where(mask, weights, 0.0)

    if normalize_within_domain and np.any(mask):
        mean_val = float(np.mean(weights[mask]))
        if mean_val > 0.0:
            weights = np.where(mask, weights / mean_val, 0.0)
        else:
            weights = np.where(mask, 1.0, 0.0)

    if apply_teacher_multiplier:
        weights = weights * float(table.admission_multiplier)
    return weights.astype(np.float32)


def build_batch_reliability_weights(
    domains: Sequence[str],
    teacher_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    response_masks: torch.Tensor,
    tables: Mapping[str, CalibrationBucketTable],
    normalize_within_domain: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """按 batch 内领域逐 token 计算 4.4 可靠性权重（领域内归一到均值 1 后乘教师准入系数）。"""
    if teacher_logprobs.shape != ref_logprobs.shape or teacher_logprobs.shape != response_masks.shape:
        raise ValueError("teacher_logprobs、ref_logprobs 与 response_masks 必须等形")
    if len(domains) != teacher_logprobs.shape[0]:
        raise ValueError("domains 长度必须与 batch 大小一致")

    weights = torch.zeros_like(teacher_logprobs, dtype=torch.float32)
    mask_bool = response_masks.bool()
    shifts_all = (teacher_logprobs.float() - ref_logprobs.float()).detach().cpu().numpy()
    masks_np = mask_bool.detach().cpu().numpy()

    rows_by_domain: dict[str, list[int]] = {}
    for row_idx, domain in enumerate(domains):
        rows_by_domain.setdefault(str(domain), []).append(row_idx)

    domain_means: dict[str, float] = {}
    for domain, row_indices in rows_by_domain.items():
        if domain == SELF_ANCHOR_DOMAIN or domain not in tables:
            for r in row_indices:
                weights[r] = mask_bool[r].float()
            domain_means[domain] = 1.0
            continue
        table = tables[domain]
        inner_edges = np.asarray(table.bin_edges[1:-1], dtype=np.float64)
        bucket_table = np.asarray(table.bucket_weights, dtype=np.float64)

        raw_rows: list[np.ndarray] = []
        valid_values: list[np.ndarray] = []
        for r in row_indices:
            row_mask = masks_np[r]
            bucket_ids = np.searchsorted(inner_edges, shifts_all[r], side="right")
            row_w = np.where(row_mask, bucket_table[bucket_ids], 0.0)
            raw_rows.append(row_w)
            if np.any(row_mask):
                valid_values.append(row_w[row_mask])

        if normalize_within_domain and valid_values:
            concat_valid = np.concatenate(valid_values)
            mean_val = float(np.mean(concat_valid))
        else:
            mean_val = 1.0

        scale = (1.0 / mean_val) if mean_val > 0.0 else 1.0
        for r, row_w in zip(row_indices, raw_rows, strict=True):
            row_mask = masks_np[r]
            if mean_val <= 0.0:
                normed = np.where(row_mask, 1.0, 0.0)
            else:
                normed = np.where(row_mask, row_w * scale, 0.0)
            final_row = normed * float(table.admission_multiplier)
            weights[r] = torch.from_numpy(final_row.astype(np.float32)).to(weights.device)
        domain_means[domain] = float(table.admission_multiplier)

    return weights, domain_means


class CalibrationManager:
    def __init__(
        self,
        tables: Mapping[str, CalibrationBucketTable] | None = None,
        refresh_interval: int = 40,
        num_bins: int = 10,
        alpha: float = 2.0,
        lift_cap: float = 1.5,
        lb_scale: float = 2.0,
    ) -> None:
        self.tables: dict[str, CalibrationBucketTable] = dict(tables or {})
        self.refresh_interval = int(refresh_interval)
        self.num_bins = int(num_bins)
        self.alpha = float(alpha)
        self.lift_cap = float(lift_cap)
        self.lb_scale = float(lb_scale)
        self.last_refresh_step = 0

    def should_refresh(self, global_step: int) -> bool:
        return (
            self.refresh_interval > 0
            and global_step > 0
            and global_step % self.refresh_interval == 0
            and global_step != self.last_refresh_step
        )

    def refresh_from_records(
        self,
        records: Sequence[dict[str, Any]],
        admission_lbs: Mapping[str, float] | None = None,
        global_step: int = 0,
    ) -> dict[str, CalibrationBucketTable]:
        grouped_shifts: dict[str, list[float]] = {}
        grouped_passed: dict[str, list[float]] = {}
        for row in records:
            teacher = str(row.get("teacher", row["domain"]))
            if teacher == SELF_ANCHOR_DOMAIN:
                continue
            t_lp = np.asarray(row["teacher_logprobs"], dtype=np.float64)
            r_lp = np.asarray(row["ref_logprobs"], dtype=np.float64)
            mask = np.asarray(row["response_mask"], dtype=bool)
            if not mask.any():
                continue
            shifts = (t_lp[mask] - r_lp[mask]).tolist()
            passed = 1.0 if bool(row["passed"]) else 0.0
            grouped_shifts.setdefault(teacher, []).extend(shifts)
            grouped_passed.setdefault(teacher, []).extend([passed] * len(shifts))

        lbs = dict(admission_lbs or {})
        for teacher, shifts in grouped_shifts.items():
            prev_lb = self.tables[teacher].admission_multiplier * self.lb_scale if teacher in self.tables else 2.0
            lb = float(lbs.get(teacher, prev_lb))
            self.tables[teacher] = fit_calibration_table(
                teacher_id=teacher,
                shifts=shifts,
                passed_flags=grouped_passed[teacher],
                num_bins=self.num_bins,
                alpha=self.alpha,
                lift_cap=self.lift_cap,
                admission_lb=lb,
                lb_scale=self.lb_scale,
                version=f"step_{global_step}",
            )
        self.last_refresh_step = int(global_step)
        return dict(self.tables)

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_refresh_step": self.last_refresh_step,
            "tables": {name: table.to_dict() for name, table in self.tables.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.last_refresh_step = int(state.get("last_refresh_step", 0))
        self.tables = {
            name: CalibrationBucketTable.from_dict(payload)
            for name, payload in dict(state.get("tables", {})).items()
        }


def save_calibration_tables(tables: Mapping[str, CalibrationBucketTable], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: table.to_dict() for name, table in tables.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_calibration_tables(path: Path) -> dict[str, CalibrationBucketTable]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: CalibrationBucketTable.from_dict(item) for name, item in payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="4.4 教师校准位移可靠性分桶构建")
    parser.add_argument("--input", type=Path, required=True, help="校准集打分 JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission-lbs", type=Path, default=None, help="各教师准入分差下界 JSON")
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--lift-cap", type=float, default=1.5)
    parser.add_argument("--lb-scale", type=float, default=2.0)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    lbs = json.loads(args.admission_lbs.read_text(encoding="utf-8")) if args.admission_lbs else {}
    manager = CalibrationManager(
        num_bins=args.num_bins,
        alpha=args.alpha,
        lift_cap=args.lift_cap,
        lb_scale=args.lb_scale,
    )
    tables = manager.refresh_from_records(records, admission_lbs=lbs, global_step=0)
    save_calibration_tables(tables, args.output)


if __name__ == "__main__":
    main()
