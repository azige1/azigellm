"""配比打包：成功轨迹 + 增强样本（注入版 / 压缩段），输出 ms-swift 4.6.0 agent 格式。

数据格式（OpenAI 风格）：``{"tools": [...], "messages": [...]}``，
assistant 保留 ``tool_calls``（function.arguments 为 JSON 字符串），
工具结果为 ``{"role": "tool", "tool_call_id", "content"}``；
同一轮多个调用放在一个 assistant 的 tool_calls 数组中。
去掉 ``g``、``reasoning``、``is_error`` 等非训练字段；转换前后各做一次配对校验。
超过 --max_tokens 的样本从头部截掉早期轮次（保留 system 与任务消息）并保持调用配对，
截完仍超长则丢弃并计数（训练侧再用 --truncation_strategy delete 兜住）。
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Optional

from trajectory_tools.pair_validation import validate_pairs
from trajectory_tools.token_estimate import count_messages_tokens

SYSTEM_PROMPT_PLACEHOLDER = "system"


def to_swift_sample(traj: dict, tools: Optional[list] = None) -> dict[str, Any]:
    """把导出轨迹转成 ms-swift OpenAI 风格训练样本。"""
    messages = []
    for m in traj["messages"]:
        role = m.get("role")
        if role == "assistant":
            new_m: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                new_m["tool_calls"] = m["tool_calls"]
            messages.append(new_m)
        elif role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": m["tool_call_id"],
                    "content": m.get("content") or "",
                }
            )
        elif role in ("system", "user"):
            messages.append({"role": role, "content": m.get("content") or ""})
        # 其余角色忽略
    return {"tools": tools or [], "messages": messages}


def _round_spans(messages: list[dict]) -> list[tuple[int, int]]:
    """把消息按「assistant + 紧随的 tool 消息」切成轮次区间。"""
    spans = []
    start = None
    for idx, m in enumerate(messages):
        if m.get("role") == "assistant":
            if start is not None:
                spans.append((start, idx))
            start = idx
    if start is not None:
        spans.append((start, len(messages)))
    return spans


def count_sample_tokens(sample: dict, tokenizer: Any = None) -> int:
    """用 Qwen3 tokenizer + chat template 实计；没有 tokenizer 时用字符数/3 粗估。"""
    if tokenizer is None:
        return count_messages_tokens(sample["messages"])
    text = tokenizer.apply_chat_template(
        sample["messages"],
        tools=sample.get("tools") or None,
        tokenize=False,
        add_generation_prompt=False,
    )
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def trim_to_tokens(
    sample: dict,
    max_tokens: int,
    tokenizer: Any = None,
) -> Optional[dict]:
    """从头部截掉早期轮次（保留 system 与任务消息）直到不超 max_tokens。"""
    current = count_sample_tokens(sample, tokenizer)
    if current <= max_tokens:
        return sample
    messages = sample["messages"]
    # 保留前缀：system + 任务 user（及其后到第一个 assistant 之前的消息）
    first_assistant = next(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
        len(messages),
    )
    keep_head = messages[:first_assistant]
    spans = _round_spans(messages[first_assistant:])
    body = messages[first_assistant:]
    dropped = 0
    for start, end in spans:
        if current <= max_tokens:
            break
        removed = body[start:end]
        current -= count_messages_tokens(removed)
        dropped += 1
    if dropped:
        kept_spans = spans[dropped:]
        if kept_spans:
            body = body[kept_spans[0][0] : kept_spans[-1][1]]
        else:
            body = []
    trimmed = {"tools": sample["tools"], "messages": keep_head + body}
    if count_sample_tokens(trimmed, tokenizer) > max_tokens:
        return None
    if validate_pairs(trimmed["messages"]):
        return None
    return trimmed


def _target_augmented(n_success: float, mix: float) -> int:
    """增强样本目标条数：A / (N + A) = mix。"""
    return round(mix * n_success / (1 - mix)) if mix < 1 else n_success


def build_mix(
    success_trajs: list[dict],
    injected_trajs: list[dict],
    compacted_trajs: list[dict],
    mix: float = 0.25,
    inject_to_compact: str = "3:1",
    baseline_raw: bool = False,
    repeat_success: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """组装训练集；返回 {"samples_meta": [...], "actual_mix": float}。"""
    rng = random.Random(seed)
    chosen = list(success_trajs)
    augmented: list[dict] = []
    if baseline_raw:
        return {"chosen": chosen, "augmented": [], "actual_mix": 0.0}
    target = _target_augmented(len(chosen), mix)
    if repeat_success:
        if chosen and target:
            augmented = [rng.choice(chosen) for _ in range(target)]
        return {
            "chosen": chosen,
            "augmented": augmented,
            "actual_mix": len(augmented) / (len(chosen) + len(augmented)) if augmented else 0.0,
        }
    ratio_m, ratio_n = (int(x) for x in inject_to_compact.split(":"))
    want_inject = round(target * ratio_m / (ratio_m + ratio_n))
    want_compact = target - want_inject
    picked_inject = injected_trajs[:want_inject] if len(injected_trajs) >= want_inject else list(injected_trajs)
    rest_inject = [t for t in injected_trajs if t not in picked_inject]
    picked_compact = compacted_trajs[:want_compact] if len(compacted_trajs) >= want_compact else list(compacted_trajs)
    augmented = picked_inject + picked_compact
    if len(augmented) < target and rest_inject:
        augmented += rest_inject[: target - len(augmented)]
    actual_mix = len(augmented) / (len(chosen) + len(augmented)) if augmented else 0.0
    return {"chosen": chosen, "augmented": augmented, "actual_mix": actual_mix}


def load_trajs(directory: Optional[Path]) -> list[dict]:
    """读目录下全部轨迹 JSON。"""
    if directory is None:
        return []
    return [
        json.loads(f.read_text(encoding="utf-8"))
        for f in sorted(Path(directory).glob("*.json"))
    ]


def _load_by_ids(full_dir: Path, ids: list[str]) -> list[dict]:
    """按轨迹 id 从 exported/full 读完整轨迹。"""
    return [
        json.loads((full_dir / f"{tid}.json").read_text(encoding="utf-8"))
        for tid in ids
        if (full_dir / f"{tid}.json").exists()
    ]


def main() -> None:
    """CLI：打包训练集。"""
    parser = argparse.ArgumentParser(description="配比打包 ms-swift 训练集")
    parser.add_argument("--selected", default=None, help="selected.jsonl（同目录下需有 full/）或导出的 full 目录")
    parser.add_argument(
        "--raw_index",
        default=None,
        help="--baseline_raw 时使用的 exported/index.jsonl：只按通过率阈值取成功轨迹，不经归因筛选与盲改过滤",
    )
    parser.add_argument("--raw_pass_rate", type=float, default=0.9, help="初版数据的通过率阈值")
    parser.add_argument("--injected", default=None, help="augmented/injected 目录")
    parser.add_argument("--compacted", default=None, help="augmented/compacted 目录")
    parser.add_argument("--mix", type=float, default=0.25)
    parser.add_argument("--inject_to_compact", default="3:1")
    parser.add_argument("--baseline_raw", action="store_true")
    parser.add_argument("--repeat_success", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--tools", default=None, help="tools JSON 文件；缺省使用 rollout 时的工具声明")
    parser.add_argument("--tokenizer", default=None, help="Qwen3 tokenizer 目录或名字，给了才实计 token")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.baseline_raw:
        if not args.raw_index:
            parser.error("--baseline_raw 需要 --raw_index exported/index.jsonl")
        index_path = Path(args.raw_index)
        entries = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ids = [
            e["id"]
            for e in entries
            if e.get("hidden_pass_rate") is not None and e["hidden_pass_rate"] >= args.raw_pass_rate
        ]
        success_trajs = _load_by_ids(index_path.parent / "full", ids)
    else:
        if not args.selected:
            parser.error("需要 --selected")
        selected_path = Path(args.selected)
        if selected_path.is_dir():
            success_trajs = load_trajs(selected_path)
        else:
            # selected.jsonl：按 id 找回 exported/full 的完整轨迹
            ids = [
                json.loads(line)["id"]
                for line in selected_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            success_trajs = _load_by_ids(selected_path.parent / "full", ids)
    injected = load_trajs(Path(args.injected)) if args.injected else []
    compacted = load_trajs(Path(args.compacted)) if args.compacted else []

    if args.tools:
        tools = json.loads(Path(args.tools).read_text(encoding="utf-8"))
    else:
        from agent_rollout.tools import TOOLS_SCHEMA

        tools = TOOLS_SCHEMA

    mix_result = build_mix(
        success_trajs,
        injected,
        compacted,
        mix=args.mix,
        inject_to_compact=args.inject_to_compact,
        baseline_raw=args.baseline_raw,
        repeat_success=args.repeat_success,
    )
    print(f"成功轨迹 {len(mix_result['chosen'])} 条，增强样本 {len(mix_result['augmented'])} 条，实际占比 {mix_result['actual_mix']:.2%}")

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    kept = 0
    dropped = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for traj in mix_result["chosen"] + mix_result["augmented"]:
            sample = to_swift_sample(traj, tools)
            if validate_pairs(sample["messages"]):
                dropped += 1
                continue
            trimmed = trim_to_tokens(sample, args.max_tokens, tokenizer)
            if trimmed is None:
                dropped += 1
                continue
            if validate_pairs(trimmed["messages"]):
                dropped += 1
                continue
            f.write(json.dumps(trimmed, ensure_ascii=False) + "\n")
            kept += 1
    print(f"写出 {kept} 条，丢弃 {dropped} 条（含超长截不完的样本）")


if __name__ == "__main__":
    main()
