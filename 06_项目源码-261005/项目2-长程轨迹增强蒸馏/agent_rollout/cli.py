"""命令行入口：批量 rollout 采集。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="采集编码轨迹")


@app.command()
def run(
    tasks: str = typer.Option(..., "--tasks", help="任务目录，如 tasks/"),
    split: str = typer.Option("train", "--split", help="要采集的数据划分"),
    runs_per_task: int = typer.Option(2, "--runs_per_task"),
    max_turns: int = typer.Option(150, "--max_turns"),
    model: str = typer.Option(..., "--model", help="teacher 模型名（OpenAI 兼容）"),
    out: Path = typer.Option(..., "--out", help="轨迹输出目录，如 runs/"),
    workers: int = typer.Option(2, "--workers", help="并行任务数"),
    compact_at: int = typer.Option(100_000, "--compact_at", help="推理压缩阈值（token 估算）"),
    timeout_s: int = typer.Option(3 * 3600, "--timeout_s"),
) -> None:
    """在指定划分上并行采集轨迹并自动判分。"""
    from openai import OpenAI

    from rebuild_spec.core.dataset_loader import load_task_records
    from agent_rollout.run_loop import run_rollout

    client = OpenAI()  # 读 OPENAI_BASE_URL / OPENAI_API_KEY
    records = [
        one
        for one in load_task_records(tasks, "test")
        if one.get("split") == split
    ]
    if not records:
        typer.echo("没有匹配的划分记录")
        raise typer.Exit(code=1)
    out.mkdir(parents=True, exist_ok=True)

    def work(record):
        task_out = out / record["instance_id"]
        return run_rollout(
            record,
            model,
            client,
            task_out,
            max_turns,
            compact_at,
            timeout_s,
            dataset_name=tasks,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(work, record): record["instance_id"]
            for record in records
            for _ in range(runs_per_task)
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                typer.echo(f"{name} {result['run_id']}: {result['subtype']} ({result['num_turns']} 轮)")
            except Exception as e:  # noqa: BLE001
                typer.echo(f"{name} 失败：{e}")


__all__ = []
