"""导出：把 events.jsonl 合并成带全局索引的轨迹 JSON，并做配对校验。

- 同一 message id 的 assistant 内容块合并为一条 OpenAI 风格消息；
- tool_result 事件转为 ``{"role": "tool", ...}`` 消息；
- 按顺序写全局索引 ``g``（system 为 g0），之后所有处理只追加内容、不重新编号；
- 配对校验失败整条剔除，并统计失败类型。
"""

import json
from pathlib import Path
from typing import Any, Optional

from trajectory_tools.pair_validation import validate_pairs
from trajectory_tools.behavior_metrics import compute_blind_metrics
from trajectory_tools.token_estimate import count_messages_tokens


def read_events(events_path: Path) -> list[dict[str, Any]]:
    """读取 events.jsonl 为记录列表。"""
    records = []
    for line in Path(events_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def merge_events(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并同一 assistant 回复的内容块，还原 OpenAI 风格消息序列。"""
    messages: list[dict[str, Any]] = []
    assistant_by_id: dict[str, dict[str, Any]] = {}
    compact_boundary = False
    for record in records:
        rtype = record.get("type")
        if rtype == "system":
            if record.get("subtype") == "init":
                messages.append(
                    {"role": "system", "content": record.get("system_prompt", "")}
                )
                if record.get("task"):
                    messages.append({"role": "user", "content": record["task"]})
            elif record.get("subtype") == "compact_boundary":
                compact_boundary = True
        elif rtype == "assistant":
            msg = record["message"]
            message_id = msg.get("id")
            if message_id not in assistant_by_id:
                new_msg: dict[str, Any] = {"role": "assistant", "content": ""}
                assistant_by_id[message_id] = new_msg
                messages.append(new_msg)
            target = assistant_by_id[message_id]
            for block in msg.get("content") or []:
                btype = block.get("type")
                if btype == "thinking":
                    target["reasoning"] = (target.get("reasoning") or "") + block.get("thinking", "")
                elif btype == "text":
                    target["content"] = (target.get("content") or "") + block.get("text", "")
                elif btype == "tool_use":
                    target.setdefault("tool_calls", []).append(
                        {
                            "id": block["id"],
                            "type": "function",
                            "function": {"name": block["name"], "arguments": json.dumps(block.get("input", {}), ensure_ascii=False)},
                        }
                    )
        elif rtype == "user":
            for block in record["message"].get("content") or []:
                if block.get("type") == "tool_result":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block.get("content", ""),
                            "is_error": bool(block.get("is_error")),
                        }
                    )
                elif block.get("type") == "text":
                    messages.append({"role": "user", "content": block.get("text", "")})
    return messages


def assign_global_index(messages: list[dict[str, Any]]) -> None:
    """按顺序写全局索引 g（从 0 起）。"""
    for idx, message in enumerate(messages):
        message["g"] = idx


def export_trajectory(
    events_path: Path,
    hidden_pass_rate: Optional[float] = None,
    instance_id: Optional[str] = None,
) -> tuple[Optional[dict[str, Any]], list[str]]:
    """导出单条轨迹：返回 (轨迹 dict 或 None, 校验错误列表)。"""
    records = read_events(events_path)
    messages = merge_events(records)
    assign_global_index(messages)
    errors = validate_pairs(messages)
    if errors:
        return None, errors
    blind = compute_blind_metrics(messages)
    meta = {
        "num_messages": len(messages),
        "num_assistant": sum(1 for m in messages if m.get("role") == "assistant"),
        "est_tokens": count_messages_tokens(messages),
        "has_compact_boundary": any(
            r.get("type") == "system" and r.get("subtype") == "compact_boundary"
            for r in records
        ),
        "result_subtype": next(
            (r.get("subtype") for r in reversed(records) if r.get("type") == "result"),
            None,
        ),
        "max_blind": blind["max_blind"],
        "blind_segments": blind["blind_segments"],
    }
    if hidden_pass_rate is not None:
        meta["hidden_pass_rate"] = hidden_pass_rate
    return {"messages": messages, "meta": meta, "instance_id": instance_id}, []


def _load_score(run_dir: Path) -> Optional[float]:
    """读判分通过率；判分过程本身出错（score.json 带 error 或没有 report.json）时返回 None。"""
    score_file = run_dir / "score.json"
    if not score_file.exists() or not (run_dir / "report.json").exists():
        return None
    try:
        score = json.loads(score_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if score.get("error"):
        return None
    return score.get("hidden_pass_rate")


def export_all(runs_dir: Path, out_dir: Path) -> dict[str, Any]:
    """导出 runs/ 下全部轨迹：写 exported/full/ 与 exported/index.jsonl，返回统计。"""
    full_dir = out_dir / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"
    stats: dict[str, Any] = {
        "total": 0,
        "kept": 0,
        "dropped": 0,
        "failure_types": {},
        "raw_assistant_records": 0,
        "merged_assistant_messages": 0,
    }
    with index_path.open("w", encoding="utf-8") as index:
        for instance_dir in sorted(Path(runs_dir).iterdir()):
            if not instance_dir.is_dir():
                continue
            for run_dir in sorted(instance_dir.iterdir()):
                if not (run_dir / "events.jsonl").exists():
                    continue
                stats["total"] += 1
                traj_id = f"{instance_dir.name}__{run_dir.name}"
                records = read_events(run_dir / "events.jsonl")
                stats["raw_assistant_records"] += sum(
                    1 for r in records if r.get("type") == "assistant"
                )
                traj, errors = export_trajectory(
                    run_dir / "events.jsonl",
                    hidden_pass_rate=_load_score(run_dir),
                    instance_id=instance_dir.name,
                )
                if traj is None:
                    stats["dropped"] += 1
                    for err in errors:
                        stats["failure_types"][err] = stats["failure_types"].get(err, 0) + 1
                    continue
                stats["merged_assistant_messages"] += traj["meta"]["num_assistant"]
                traj["id"] = traj_id
                (full_dir / f"{traj_id}.json").write_text(
                    json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                index.write(
                    json.dumps(
                        {
                            "id": traj_id,
                            "instance_id": traj["instance_id"],
                            "hidden_pass_rate": traj["meta"].get("hidden_pass_rate"),
                            "num_messages": traj["meta"]["num_messages"],
                            "num_assistant": traj["meta"]["num_assistant"],
                            "est_tokens": traj["meta"]["est_tokens"],
                            "has_compact_boundary": traj["meta"]["has_compact_boundary"],
                            "max_blind": traj["meta"]["max_blind"],
                            "blind_segments": traj["meta"]["blind_segments"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stats["kept"] += 1
    return stats


__all__ = []
