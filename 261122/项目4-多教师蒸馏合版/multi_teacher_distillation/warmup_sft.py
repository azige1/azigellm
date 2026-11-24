import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import SELF_ANCHOR_DOMAIN


def compute_reverse_kl_and_topk_overlap(
    student_logprobs: Sequence[float],
    teacher_logprobs: Sequence[float],
    student_topk_ids: Sequence[Sequence[int]],
    teacher_topk_ids: Sequence[Sequence[int]],
    response_mask: Sequence[int],
    top_k: int = 20,
) -> dict[str, float]:
    """在有效 assistant token 上计算 reverse KL 与 top-k 词表重叠率。"""
    if top_k <= 0:
        raise ValueError("top_k 必须为正整数")
    s_lp = np.asarray(student_logprobs, dtype=np.float64)
    t_lp = np.asarray(teacher_logprobs, dtype=np.float64)
    mask = np.asarray(response_mask, dtype=bool)
    if s_lp.shape != t_lp.shape or s_lp.shape != mask.shape:
        raise ValueError("student_logprobs、teacher_logprobs 与 response_mask 必须等长")
    if len(student_topk_ids) != len(s_lp) or len(teacher_topk_ids) != len(s_lp):
        raise ValueError("top-k token 列表长度必须与序列长度一致")
    if not mask.any():
        raise ValueError("response_mask 必须至少包含一个有效 token")

    valid_indices = np.flatnonzero(mask)
    kl_sum = float((s_lp[valid_indices] - t_lp[valid_indices]).sum())
    overlap_sum = 0.0
    for idx in valid_indices:
        s_set = {int(tok) for tok in student_topk_ids[int(idx)][:top_k]}
        t_set = {int(tok) for tok in teacher_topk_ids[int(idx)][:top_k]}
        overlap_sum += len(s_set & t_set) / float(top_k)

    token_count = int(len(valid_indices))
    return {
        "token_count": token_count,
        "kl_sum": kl_sum,
        "overlap_sum": overlap_sum,
        "reverse_kl": kl_sum / token_count,
        "topk_overlap": overlap_sum / token_count,
    }


def evaluate_domain_overlap(
    records: Sequence[dict[str, Any]],
    max_reverse_kl: float = 0.15,
    min_topk_overlap: float = 0.75,
    top_k: int = 20,
) -> dict[str, Any]:
    """按领域统计学生起点 rollout 的 token 加权 reverse KL 与 top-k 重叠率。"""
    if not records:
        raise ValueError("分布重叠度检查输入不能为空")
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"tokens": 0.0, "kl_sum": 0.0, "overlap_sum": 0.0, "samples": 0})
    for row in records:
        domain = str(row["domain"])
        if domain == SELF_ANCHOR_DOMAIN:
            continue
        stats = compute_reverse_kl_and_topk_overlap(
            student_logprobs=row["student_logprobs"],
            teacher_logprobs=row["teacher_logprobs"],
            student_topk_ids=row["student_topk_ids"],
            teacher_topk_ids=row["teacher_topk_ids"],
            response_mask=row["response_mask"],
            top_k=top_k,
        )
        bucket = grouped[domain]
        bucket["tokens"] += stats["token_count"]
        bucket["kl_sum"] += stats["kl_sum"]
        bucket["overlap_sum"] += stats["overlap_sum"]
        bucket["samples"] += 1

    domains_report: dict[str, Any] = {}
    warmup_domains: list[str] = []
    for domain, agg in sorted(grouped.items()):
        tokens = max(int(agg["tokens"]), 1)
        rev_kl = agg["kl_sum"] / tokens
        overlap = agg["overlap_sum"] / tokens
        passed = (rev_kl <= max_reverse_kl) and (overlap >= min_topk_overlap)
        if not passed:
            warmup_domains.append(domain)
        domains_report[domain] = {
            "samples": int(agg["samples"]),
            "tokens": int(agg["tokens"]),
            "reverse_kl": float(rev_kl),
            "topk_overlap": float(overlap),
            "passed": bool(passed),
            "needs_warmup": not passed,
            "overlap_gap": float(max(0.0, min_topk_overlap - overlap)),
        }
    return {
        "max_reverse_kl": float(max_reverse_kl),
        "min_topk_overlap": float(min_topk_overlap),
        "top_k": int(top_k),
        "domains": domains_report,
        "warmup_domains": warmup_domains,
        "all_passed": len(warmup_domains) == 0,
    }


def allocate_warmup_samples(
    overlap_report: dict[str, Any],
    total_samples: int = 4000,
    min_topk_overlap: float = 0.75,
) -> dict[str, int]:
    """按重叠率缺口比例为未达标领域分配预热 SFT 样本数，达标领域为 0。"""
    if total_samples < 0:
        raise ValueError("预热总样本数不能为负")
    domains = overlap_report["domains"]
    gaps: dict[str, float] = {}
    for domain, info in domains.items():
        if info.get("needs_warmup", False):
            overlap_gap = max(0.0, float(min_topk_overlap) - float(info["topk_overlap"]))
            if overlap_gap <= 0.0:
                overlap_gap = max(1e-6, float(info["reverse_kl"]) - float(overlap_report.get("max_reverse_kl", 0.15)))
            gaps[domain] = overlap_gap

    allocations = {domain: 0 for domain in domains}
    if not gaps or total_samples == 0:
        return allocations

    total_gap = sum(gaps.values())
    raw = {domain: total_samples * gap / total_gap for domain, gap in gaps.items()}
    floored = {domain: int(val) for domain, val in raw.items()}
    remainder = total_samples - sum(floored.values())
    order = sorted(gaps, key=lambda d: (raw[d] - floored[d], d), reverse=True)
    for domain in order[:remainder]:
        floored[domain] += 1
    allocations.update(floored)
    return allocations


def filter_verified_teacher_samples(
    records: Sequence[dict[str, Any]],
    allocations: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """仅保留教师生成且通过测试/verifier 的样本，并按分配额裁剪。"""
    if allocations is None:
        allocations = {}
    kept_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        domain = str(row["domain"])
        target = int(allocations.get(domain, 0))
        if target <= 0 or len(kept_by_domain[domain]) >= target:
            continue
        passed = row.get("passed")
        if passed is None:
            passed = float(row.get("score", 0.0)) >= float(row.get("pass_threshold", 1.0))
        if not bool(passed):
            continue
        kept_by_domain[domain].append({
            "domain": domain,
            "prompt_id": str(row.get("prompt_id", row.get("task_id", f"{domain}_{len(kept_by_domain[domain])}"))),
            "messages": list(row["messages"]),
            "tools": row.get("tools"),
            "score": float(row.get("score", 1.0)),
            "passed": True,
        })
    selected: list[dict[str, Any]] = []
    for domain in sorted(kept_by_domain):
        selected.extend(kept_by_domain[domain])
    return selected


def verify_post_warmup_overlap(
    pre_report: dict[str, Any],
    post_records: Sequence[dict[str, Any]],
    max_reverse_kl: float = 0.15,
    min_topk_overlap: float = 0.75,
    max_compliant_overlap_drop: float = 0.02,
    top_k: int = 20,
) -> dict[str, Any]:
    """预热后复检全部领域：全部领域需达标且原已达标领域重叠率下降不超过 0.02。"""
    post_report = evaluate_domain_overlap(
        post_records,
        max_reverse_kl=max_reverse_kl,
        min_topk_overlap=min_topk_overlap,
        top_k=top_k,
    )
    compliant_drops: dict[str, float] = {}
    drift_violations: list[str] = []
    for domain, pre_info in pre_report["domains"].items():
        if domain not in post_report["domains"]:
            raise ValueError(f"复检缺少领域 {domain}")
        if pre_info["passed"]:
            drop = float(pre_info["topk_overlap"]) - float(post_report["domains"][domain]["topk_overlap"])
            compliant_drops[domain] = drop
            if drop > max_compliant_overlap_drop:
                drift_violations.append(domain)
    passed = post_report["all_passed"] and len(drift_violations) == 0
    return {
        "passed": passed,
        "post_report": post_report,
        "compliant_overlap_drops": compliant_drops,
        "drift_violations": drift_violations,
        "max_compliant_overlap_drop": float(max_compliant_overlap_drop),
    }


def render_full_conversation_with_assistant_mask(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    enable_thinking: bool = False,
) -> tuple[list[int], list[int]]:
    """整段渲染多轮对话并构造仅覆盖 assistant token 的 loss_mask，避免逐轮拼接失配。"""
    kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
    if tools:
        kwargs["tools"] = tools

    full_ids = list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            **kwargs,
        )
    )
    loss_mask = [0] * len(full_ids)
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        prefix_with_gen = list(
            tokenizer.apply_chat_template(
                messages[:idx],
                tokenize=True,
                add_generation_prompt=True,
                **kwargs,
            )
        )
        prefix_with_turn = list(
            tokenizer.apply_chat_template(
                messages[: idx + 1],
                tokenize=True,
                add_generation_prompt=False,
                **kwargs,
            )
        )
        start = len(prefix_with_gen)
        end = len(prefix_with_turn)
        if full_ids[:start] != prefix_with_gen or full_ids[:end] != prefix_with_turn:
            raise ValueError("整段渲染与前缀边界不一致，请检查对话模板")
        for pos in range(start, min(end, len(full_ids))):
            loss_mask[pos] = 1
    return full_ids, loss_mask


class DomainWarmupSFTDataset(Dataset):
    """用于 4.2 分域预热 SFT 的整段渲染多轮数据集，适配 verl SFTTrainer 与 SFTTensorCollator。"""

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: Any,
        config: Any,
        processor: Any = None,
        max_samples: int = -1,
    ) -> None:
        del processor
        config = config or {}
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        frames = [pd.read_parquet(path) for path in parquet_files]
        df = pd.concat(frames, ignore_index=True)
        if max_samples > 0 and max_samples < len(df):
            df = df.iloc[:max_samples].reset_index(drop=True)

        self.tokenizer = tokenizer
        self.messages_key = config.get("messages_key", "messages")
        self.tools_key = config.get("tools_key", "tools")
        self.pad_mode = str(config.get("pad_mode", "no_padding"))
        self.max_length = int(config.get("max_length", 16384))
        self.truncation = str(config.get("truncation", "error"))
        chat_kwargs = dict(config.get("apply_chat_template_kwargs", {}) or {})
        self.enable_thinking = bool(chat_kwargs.get("enable_thinking", False))

        self.records = df.to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = self.records[item]
        messages = list(row[self.messages_key])
        tools = row.get(self.tools_key)
        if isinstance(tools, str):
            tools = json.loads(tools)
        full_ids, mask = render_full_conversation_with_assistant_mask(
            self.tokenizer,
            messages=messages,
            tools=tools if tools else None,
            enable_thinking=self.enable_thinking,
        )
        seq_len = len(full_ids)
        if seq_len > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"预热 SFT 序列长度 {seq_len} 超过上限 {self.max_length}")
            full_ids = full_ids[: self.max_length] if self.truncation == "right" else full_ids[-self.max_length :]
            mask = mask[: self.max_length] if self.truncation == "right" else mask[-self.max_length :]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        loss_mask = torch.tensor(mask, dtype=torch.long)
        position_ids = torch.arange(len(full_ids), dtype=torch.long)

        if self.pad_mode == "right":
            pad_len = self.max_length - len(full_ids)
            if pad_len > 0:
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                attention_mask = torch.cat([torch.ones(len(full_ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
                input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                loss_mask = torch.cat([loss_mask, torch.zeros(pad_len, dtype=torch.long)])
                position_ids = torch.cat([position_ids, torch.zeros(pad_len, dtype=torch.long)])
            else:
                attention_mask = torch.ones(len(full_ids), dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="4.2 低重合领域的分域预热 SFT 工具")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check")
    check.add_argument("--input", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--max-kl", type=float, default=0.15)
    check.add_argument("--min-overlap", type=float, default=0.75)
    check.add_argument("--top-k", type=int, default=20)
    check.add_argument("--total-samples", type=int, default=4000)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--overlap-report", type=Path, required=True)
    prepare.add_argument("--teacher-samples", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--total-samples", type=int, default=4000)

    verify = sub.add_parser("verify")
    verify.add_argument("--pre-report", type=Path, required=True)
    verify.add_argument("--post-input", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--max-kl", type=float, default=0.15)
    verify.add_argument("--min-overlap", type=float, default=0.75)
    verify.add_argument("--max-compliant-drop", type=float, default=0.02)

    args = parser.parse_args()
    if args.command == "check":
        report = evaluate_domain_overlap(
            _read_jsonl(args.input),
            max_reverse_kl=args.max_kl,
            min_topk_overlap=args.min_overlap,
            top_k=args.top_k,
        )
        report["allocations"] = allocate_warmup_samples(report, total_samples=args.total_samples, min_topk_overlap=args.min_overlap)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    elif args.command == "prepare":
        overlap_report = json.loads(args.overlap_report.read_text(encoding="utf-8"))
        allocations = allocate_warmup_samples(
            overlap_report,
            total_samples=args.total_samples,
            min_topk_overlap=float(overlap_report.get("min_topk_overlap", 0.75)),
        )
        rows = filter_verified_teacher_samples(_read_jsonl(args.teacher_samples), allocations)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(args.output, index=False)
    else:
        pre_report = json.loads(args.pre_report.read_text(encoding="utf-8"))
        result = verify_post_warmup_overlap(
            pre_report=pre_report,
            post_records=_read_jsonl(args.post_input),
            max_reverse_kl=args.max_kl,
            min_topk_overlap=args.min_overlap,
            max_compliant_overlap_drop=args.max_compliant_drop,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if not result["passed"]:
            raise RuntimeError("预热 SFT 后复检未通过")


if __name__ == "__main__":
    main()
