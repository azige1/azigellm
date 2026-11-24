"""高分轨迹篡改抽检脚本（3.2 节）：每 30 步抽检 reward 最高的 50 条轨迹。"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from long_horizon_rl.tamper_scan import append_tamper_log, scan_code_snippet

app = typer.Typer(no_args_is_help=True, add_completion=False, help="抽检高分 rollout 轨迹中的篡改形态")


@app.command()
def audit(
    rollout_dir: Path = typer.Option(..., "--rollout-dir", help="verl rollout_data_dir 目录"),
    out_log: Path = typer.Option(Path("logs/audit_tamper.jsonl"), "--out", help="审计命中的篡改日志"),
    top_k: int = typer.Option(50, "--top-k", help="每步抽检最高 reward 条数"),
    step: int = typer.Option(-1, "--step", help="指定步数；-1 表示扫描目录下全部步文件"),
) -> None:
    """读取 rollout 导出 JSONL，按 reward 降序抽检 top-k 条并做静态扫描。"""
    files = sorted(rollout_dir.glob("*.jsonl")) if step < 0 else [rollout_dir / f"{step}.jsonl"]
    total_checked = 0
    total_flagged = 0

    for path in files:
        if not path.is_file():
            continue
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        records.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        step_id = int(path.stem) if path.stem.isdigit() else 0

        for item in records[:top_k]:
            total_checked += 1
            res = scan_code_snippet(str(item.get("output", "")))
            if res.flagged:
                total_flagged += 1
                append_tamper_log(
                    out_log,
                    task_id=str(item.get("uid", "")),
                    step=step_id,
                    scan_result=res,
                    trajectory_id="cli_audit",
                )

    typer.echo(f"抽检完成：共检查 {total_checked} 条高分样本，命中篡改规则 {total_flagged} 条 -> {out_log}")


if __name__ == "__main__":
    app()
