"""命令行入口：导出、归因、筛选、注入与压缩恢复的批量执行。"""

from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="轨迹导出与增强工具")


@app.command()
def export(runs: Path = typer.Option(..., "--runs"), out: Path = typer.Option(..., "--out")) -> None:
    """把 runs/ 下的轨迹导出为带全局索引的 JSON，并做配对校验。"""
    import json

    from trajectory_tools.export import export_all

    stats = export_all(runs, out)
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


@app.command()
def attribute(
    exported: Path = typer.Option(..., "--exported", help="exported/full 目录"),
    tasks: Path = typer.Option(..., "--tasks", help="tasks/test.jsonl"),
    reports: Path = typer.Option(..., "--reports", help="runs/ 目录（report.json 所在处）"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    """对全量轨迹做失败归因。"""
    from trajectory_tools.attribution import attribute_all

    results = attribute_all(exported, tasks, reports, out)
    typer.echo(f"已归因 {len(results)} 条轨迹")


@app.command()
def select(
    index: Path = typer.Option(..., "--index", help="exported/index.jsonl"),
    attribution: Path = typer.Option(..., "--attribution"),
    out: Path = typer.Option(..., "--out"),
    pass_rate: float = typer.Option(0.9, "--pass_rate"),
    max_blind: int = typer.Option(8, "--max_blind"),
) -> None:
    """按归因结果筛选成功轨迹。"""
    import json

    from trajectory_tools.select import select as select_fn

    attribution_data = {
        item["id"]: item["modes"]
        for item in json.loads(Path(attribution).read_text(encoding="utf-8"))
    }
    stats = select_fn(index, attribution_data, out, pass_rate, max_blind)
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


def _selected_files(exported: Path, selected: Path) -> list[Path]:
    """只取 selected.jsonl 中的成功轨迹：注入与压缩恢复都只作用于筛选后的成功轨迹。"""
    import json

    ids = [
        json.loads(line)["id"]
        for line in Path(selected).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [Path(exported) / f"{tid}.json" for tid in ids if (Path(exported) / f"{tid}.json").exists()]


@app.command()
def inject(
    exported: Path = typer.Option(..., "--exported", help="exported/full 目录"),
    selected: Path = typer.Option(..., "--selected", help="select 输出的 selected.jsonl"),
    out: Path = typer.Option(..., "--out", help="augmented/injected 目录"),
    model: str = typer.Option(..., "--injection_model"),
) -> None:
    """对成功轨迹批量注入过程监督回顾。"""
    import json

    from openai import OpenAI

    from trajectory_tools.injection import inject_trajectory

    client = OpenAI()
    out.mkdir(parents=True, exist_ok=True)
    for traj_file in _selected_files(exported, selected):
        traj = json.loads(traj_file.read_text(encoding="utf-8"))
        result = inject_trajectory(traj, client, model)
        if result is not None:
            (out / traj_file.name).write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            typer.echo(f"注入完成：{traj_file.name}")


@app.command()
def compact(
    exported: Path = typer.Option(..., "--exported", help="exported/full 目录"),
    selected: Path = typer.Option(..., "--selected", help="select 输出的 selected.jsonl"),
    out: Path = typer.Option(..., "--out", help="augmented/compacted 目录"),
    model: str = typer.Option(..., "--model"),
    cut_threshold: int = typer.Option(50_000, "--cut_threshold", help="32K 学生档建议 16000"),
) -> None:
    """对成功轨迹批量生成压缩恢复 B 段。"""
    import json

    from openai import OpenAI

    from trajectory_tools.compaction import compact_trajectory

    client = OpenAI()
    out.mkdir(parents=True, exist_ok=True)
    for traj_file in _selected_files(exported, selected):
        traj = json.loads(traj_file.read_text(encoding="utf-8"))
        result = compact_trajectory(traj, client, model, cut_threshold)
        if result is not None:
            (out / f"{traj['id']}_B.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            typer.echo(f"压缩恢复完成：{traj['id']}_B")


__all__ = []
