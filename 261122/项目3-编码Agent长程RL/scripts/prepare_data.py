"""将阶段一/阶段二任务 JSONL 转换为 verl V1 parquet 数据集。"""

from __future__ import annotations

from pathlib import Path

import typer

from long_horizon_rl.curriculum import load_jsonl
from long_horizon_rl.data_prep import convert_tasks_to_verl_records, write_verl_parquet

app = typer.Typer(no_args_is_help=True, add_completion=False, help="生成 verl RL parquet 数据集")


@app.command()
def convert(
    stage: str = typer.Option(..., "--stage", help="function 或 repo"),
    input_jsonl: Path = typer.Option(..., "--input", help="输入任务 JSONL"),
    out_dir: Path = typer.Option(Path("data"), "--out-dir", help="输出目录"),
    train_split: str = typer.Option("train", "--train-split"),
    eval_split: str = typer.Option("eval", "--eval-split"),
) -> None:
    """按 split 导出 train/eval parquet，异构嵌套字段序列化为 JSON 字符串。"""
    tasks = load_jsonl(input_jsonl)
    prefix = "stage1" if stage == "function" else "stage2"

    train_rows = convert_tasks_to_verl_records(tasks, stage=stage, split=train_split)
    eval_rows = convert_tasks_to_verl_records(tasks, stage=stage, split=eval_split)

    if train_rows:
        train_path = write_verl_parquet(train_rows, out_dir / f"{prefix}_train.parquet")
        typer.echo(f"训练集已写出: {train_path} ({len(train_rows)} 条)")
    if eval_rows:
        eval_path = write_verl_parquet(eval_rows, out_dir / f"{prefix}_eval.parquet")
        typer.echo(f"评测集已写出: {eval_path} ({len(eval_rows)} 条)")


if __name__ == "__main__":
    app()
