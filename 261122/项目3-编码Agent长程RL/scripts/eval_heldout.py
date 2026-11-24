"""Held-out 评测脚本：汇总仓库级通过率、压缩子集通过率、函数级 pass@1、篡改占比与零方差组占比。"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import typer

from long_horizon_rl.curriculum import load_jsonl
from long_horizon_rl.data_prep import build_function_prompt_messages
from long_horizon_rl.reward_eval import evaluate_function_task
from long_horizon_rl.tamper_scan import scan_workspace

app = typer.Typer(no_args_is_help=True, add_completion=False, help="评测 RL checkpoint 的仓库级与函数级指标")


def summarize_repo_eval(records: list[dict[str, Any]]) -> dict[str, float]:
    """根据逐轨迹评测记录汇总全部仓库级指标。"""
    if not records:
        return {
            "repo_avg_pass_rate": 0.0,
            "compaction_subset_pass_rate": 0.0,
            "tamper_ratio": 0.0,
            "zero_variance_group_ratio": 0.0,
        }

    pass_rates = [float(r["pass_rate"]) for r in records]
    comp_rates = [float(r["pass_rate"]) for r in records if r.get("compaction_triggered")]
    tamper_count = sum(1 for r in records if r.get("tamper_flag"))

    by_task: dict[str, list[float]] = {}
    for r in records:
        by_task.setdefault(str(r["task_id"]), []).append(float(r["pass_rate"]))
    multi_groups = [vals for vals in by_task.values() if len(vals) > 1]
    zero_var = sum(1 for vals in multi_groups if float(np.std(vals)) < 1e-8)

    return {
        "repo_avg_pass_rate": float(np.mean(pass_rates)),
        "compaction_subset_pass_rate": float(np.mean(comp_rates)) if comp_rates else 0.0,
        "tamper_ratio": float(tamper_count / len(records)),
        "zero_variance_group_ratio": float(zero_var / len(multi_groups)) if multi_groups else 0.0,
    }


@app.command()
def evaluate(
    repo_tasks: Path = typer.Option(..., "--repo-tasks", help="仓库级评测集 JSONL"),
    function_tasks: Optional[Path] = typer.Option(None, "--function-tasks", help="函数级评测集 JSONL"),
    model: str = typer.Option(..., "--model", help="待评测 RL checkpoint 路径"),
    base_url: str = typer.Option("http://127.0.0.1:8000/v1", "--base-url", help="vLLM OpenAI 兼容地址"),
    repeats: int = typer.Option(3, "--repeats", help="每道仓库题重复次数（默认 3 次）"),
    max_turns: int = typer.Option(60, "--max-turns"),
    compact_at: int = typer.Option(27000, "--compact-at"),
    out: Path = typer.Option(Path("eval_results/summary.json"), "--out"),
) -> None:
    """运行 held-out 仓库级与函数级评测并输出汇总 JSON。"""
    from openai import OpenAI

    from agent_rollout.run_loop import run_rollout

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    tasks = [t for t in load_jsonl(repo_tasks) if t.get("split", "eval") == "eval"]
    repo_records: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="lhrl-eval-") as tmp:
        for task in tasks:
            for rep in range(repeats):
                run_res = run_rollout(
                    task=task,
                    model=model,
                    client=client,
                    out_dir=Path(tmp) / task["instance_id"] / f"rep_{rep}",
                    max_turns=max_turns,
                    compact_at=compact_at,
                    keep_workspace=True,
                )
                run_dir = Path(run_res["out_dir"])
                events_path = run_dir / "events.jsonl"
                compacted = False
                if events_path.is_file():
                    compacted = "compact_boundary" in events_path.read_text(encoding="utf-8", errors="replace")
                score_path = run_dir / "score.json"
                pass_rate = 0.0
                if score_path.is_file():
                    pass_rate = float(json.loads(score_path.read_text(encoding="utf-8")).get("hidden_pass_rate", 0.0))
                tamper_flag = False
                if run_res.get("workspace"):
                    tamper_flag = scan_workspace(Path(run_res["workspace"])).flagged
                repo_records.append(
                    {
                        "task_id": task["instance_id"],
                        "repeat": rep,
                        "pass_rate": 0.0 if tamper_flag else pass_rate,
                        "compaction_triggered": compacted,
                        "tamper_flag": tamper_flag,
                    }
                )

    summary = summarize_repo_eval(repo_records)

    if function_tasks is not None and function_tasks.is_file():
        func_items = [t for t in load_jsonl(function_tasks) if t.get("split", "eval") == "eval"]
        func_passed = 0
        for item in func_items:
            msgs = build_function_prompt_messages(item)
            resp = client.chat.completions.create(model=model, messages=msgs, temperature=0.0, max_tokens=2048)
            reply = resp.choices[0].message.content or ""
            blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", reply, flags=re.DOTALL)
            code = blocks[-1].strip() if blocks else reply.strip()
            outcome = evaluate_function_task(item, code, mode="binary")
            if outcome.reward == 1.0:
                func_passed += 1
        summary["function_pass_at_1"] = float(func_passed / len(func_items)) if func_items else 0.0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": repo_records}, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
