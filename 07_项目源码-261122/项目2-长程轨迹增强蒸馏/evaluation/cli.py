"""学生评测：在 held-out 任务上 rollout 并汇总行为指标。"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="学生模型评测")


@app.command()
def run(
    tasks: str = typer.Option(..., "--tasks", help="任务目录"),
    split: str = typer.Option("eval", "--split"),
    model: str = typer.Option(..., "--model", help="vLLM 服务的 served model name（缺省为 checkpoint 路径）"),
    base_url: str = typer.Option("http://127.0.0.1:8000/v1", "--base_url", help="学生模型 vLLM 服务地址"),
    repeats: int = typer.Option(3, "--repeats"),
    out: Path = typer.Option(Path("eval_results"), "--out"),
    max_turns: int = typer.Option(150, "--max_turns"),
    compact_at: int = typer.Option(24_000, "--compact_at", help="32K 档推理压缩阈值"),
    workers: int = typer.Option(2, "--workers"),
    enable_thinking: bool = typer.Option(False, "--enable_thinking", help="学生按非 thinking 格式训练，默认关闭"),
) -> None:
    """用与 teacher 采集相同的 harness 在 held-out 任务上评测。"""
    from openai import OpenAI

    from rebuild_spec.core.dataset_loader import load_task_records
    from agent_rollout.run_loop import run_rollout

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    request_extra = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    records = [one for one in load_task_records(tasks, "test") if one.get("split") == split]
    if not records:
        typer.echo("没有匹配的划分记录")
        raise typer.Exit(code=1)
    eval_dir = out / "runs"
    eval_dir.mkdir(parents=True, exist_ok=True)

    def work(record, idx):
        return record["instance_id"], run_rollout(
            record,
            model,
            client,
            eval_dir / record["instance_id"],
            max_turns,
            compact_at,
            dataset_name=tasks,
            keep_workspace=True,
            request_extra=request_extra,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(work, record, idx)
            for record in records
            for idx in range(repeats)
        ]
        for future in as_completed(futures):
            name, result = future.result()
            typer.echo(f"{name} {result['run_id']}: {result['subtype']}")

    from evaluation.metrics_report import build_report

    report = build_report(eval_dir, Path(tasks) / "test.jsonl")
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out / "report.md"
    from evaluation.metrics_report import report_markdown

    md_path.write_text(report_markdown(report, model), encoding="utf-8")
    typer.echo(f"报告已写出：{report_path} / {md_path}")


__all__ = []
