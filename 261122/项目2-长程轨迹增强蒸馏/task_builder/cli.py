"""命令行入口：task_builder 的两个子命令。

- ``build``：从种子仓库清单构建任务目录与 test.jsonl；
- ``prepare-host-env``：为每个任务确保基准 venv、锁文件并写回 host_venv。
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import git
import typer
import yaml

from task_builder.skeleton_stub import build_skeleton_commit
from task_builder.spec_document import make_spec_markdown
from task_builder.host_environment import (
    venv_env,
    create_base_venv,
    install_task_dependencies,
    write_requirements_lock,
    collect_test_ids,
    run_reference_tests,
)
from task_builder.task_records import make_record, append_record, read_records, rewrite_records

app = typer.Typer(no_args_is_help=True, add_completion=False, help="从公开仓库构建本地任务")


PACKAGING_FILES = ["pyproject.toml", "setup.py", "setup.cfg", "pytest.ini", "tox.ini", "conftest.py"]


def _git_env(repo: git.Repo) -> None:
    """给任务仓库写一个本地 git 身份，避免依赖全局配置。"""
    with repo.config_writer() as cfg:
        cfg.set_value("user", "name", "task-builder")
        cfg.set_value("user", "email", "task-builder@localhost")


def _copy_hidden_tests(repo_dir: Path, task_dir: Path, test_dir: str) -> Path:
    """把参考实现的测试目录、conftest 与打包配置复制到 hidden_tests/。"""
    hidden = task_dir / "hidden_tests"
    hidden.mkdir(parents=True, exist_ok=True)
    src_tests = repo_dir / test_dir
    if src_tests.exists():
        shutil.copytree(src_tests, hidden / test_dir, dirs_exist_ok=True)
    for fname in PACKAGING_FILES:
        fpath = repo_dir / fname
        if fpath.exists():
            shutil.copyfile(fpath, hidden / fname)
    return hidden


def _checkout_reference(repo_dir: Path, github: str, tag: str) -> str:
    """克隆并检出参考 tag，返回参考提交号。"""
    clone_url = f"https://github.com/{github}.git"
    subprocess.run(["git", "clone", clone_url, str(repo_dir)], check=True)
    repo = git.Repo(repo_dir)
    repo.git.checkout(tag)
    return repo.head.commit.hexsha


BASE_BRANCH_NAME = "base"


def _commit_skeleton(repo_dir: Path, src_dir: str, test_dir: str) -> str:
    """在无历史的孤立分支 base 上提交骨架，返回 base_commit。

    骨架提交不挂在参考实现的历史之上，工作区只克隆 base 分支，
    Agent 无法通过 git log / git show 看到原始实现。
    """
    repo = git.Repo(repo_dir)
    _git_env(repo)
    repo.git.checkout("--orphan", BASE_BRANCH_NAME)
    build_skeleton_commit(repo_dir, src_dir)
    shutil.rmtree(repo_dir / test_dir, ignore_errors=True)
    # 参考实现上安装与跑测试留下的产物不进入骨架提交
    for pattern in ("**/__pycache__", "**/.pytest_cache", "**/*.egg-info", "build", "dist"):
        for path in repo_dir.glob(pattern):
            if ".git" not in path.parts:
                shutil.rmtree(path, ignore_errors=True)
    repo.git.add(A=True)
    repo.index.commit("base skeleton")
    return repo.head.commit.hexsha


@app.command()
def build(
    repos: Path = typer.Option(..., "--repos", help="种子仓库 yaml 路径"),
    out: Path = typer.Option(..., "--out", help="任务输出目录，如 tasks/"),
    limit: Optional[int] = typer.Option(None, "--limit", help="只处理前 N 个仓库"),
) -> None:
    """从种子仓库清单构建任务目录与 test.jsonl。"""
    tasks_dir = Path(out)
    tasks_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(repos).read_text(encoding="utf-8"))
    entries = config["repos"]
    if limit is not None:
        entries = entries[:limit]
    for entry in entries:
        name = entry["name"]
        task_dir = tasks_dir / name
        task_dir.mkdir(parents=True, exist_ok=True)
        repo_dir = task_dir / "repo"
        typer.echo(f"构建任务 {name} ...")
        if repo_dir.exists():
            typer.echo(f"  {repo_dir} 已存在，跳过（删掉目录可重建）")
            continue
        # 1. 参考实现上的准备：隐藏测试、规格文档、环境、锁文件、测试 ID 与参考通过率
        reference_commit = _checkout_reference(repo_dir, entry["github"], entry["tag"])
        _copy_hidden_tests(repo_dir, task_dir, entry["test_dir"])
        (task_dir / "spec.md").write_text(
            make_spec_markdown(repo_dir, entry["src_dir"], entry["test_dir"]), encoding="utf-8"
        )

        setup = {"python": entry["python"], "pip_packages": entry.get("pip_packages") or []}
        venv = create_base_venv(task_dir, entry["python"])
        install_task_dependencies(venv, setup)
        subprocess.run(
            ["uv", "pip", "install", "-e", "."],
            cwd=str(repo_dir),
            check=True,
            env=venv_env(venv),
        )
        lock = write_requirements_lock(venv, task_dir, name)
        test_ids = collect_test_ids(venv, repo_dir, entry["test_dir"])
        pass_rate, passed_ids, output = run_reference_tests(
            venv, repo_dir, entry["test_dir"], test_ids
        )
        # 判分集合只保留参考实现能通过的用例
        (task_dir / "test_ids.txt").write_text("\n".join(passed_ids) + "\n", encoding="utf-8")

        # 2. 参考实现处理完后再生成骨架提交（孤立分支，删除测试目录）
        base_commit = _commit_skeleton(repo_dir, entry["src_dir"], entry["test_dir"])
        if pass_rate < 0.95:
            typer.echo(
                typer.style(
                    f"  注意：{name} 参考实现通过率 {pass_rate:.2f}，低于 0.95",
                    fg=typer.colors.YELLOW,
                )
            )
        record = make_record(
            name=name,
            github=entry["github"],
            base_commit=base_commit,
            reference_commit=reference_commit,
            setup=setup,
            test_cmd=f"python -m pytest {entry['test_dir']}",
            test_dir=entry["test_dir"],
            src_dir=entry["src_dir"],
            task_dir=task_dir,
            split=entry.get("split", "train"),
            reference_pass_rate=pass_rate,
        )
        record["host_venv"] = str(venv.resolve())
        append_record(tasks_dir, record)
        typer.echo(f"  完成：base={base_commit[:8]} 参考通过率={pass_rate:.2f} 锁文件={lock.name}")


@app.command()
def prepare_host_env(
    tasks: Path = typer.Option(..., "--tasks", help="任务目录，如 tasks/"),
) -> None:
    """为每个任务确保基准 venv 与 requirements.lock 存在，并写回 host_venv。"""
    tasks_dir = Path(tasks)
    records = read_records(tasks_dir)
    for record in records:
        name = record["instance_id"]
        task_dir = tasks_dir / name
        venv = task_dir / ".venv"
        if not venv.exists():
            typer.echo(f"建基准 venv：{name}")
            create_base_venv(task_dir, record["setup"]["python"])
            install_task_dependencies(venv, record["setup"])
        lock = task_dir / "requirements.lock"
        if not lock.exists():
            typer.echo(f"导出锁文件：{name}")
            write_requirements_lock(venv, task_dir, name)
        record["host_venv"] = str(venv.resolve())
    rewrite_records(tasks_dir, records)
    typer.echo(f"已更新 {len(records)} 条记录的 host_venv")


__all__ = []
