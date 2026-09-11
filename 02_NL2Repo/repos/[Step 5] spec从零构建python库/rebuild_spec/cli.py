"""命令行入口：rebuild-spec 的七个子命令。

流程顺序：setup → build → (get-tests) → test → evaluate → lint → save。
运行状态保存在项目根的 ``.rebuild-spec.yaml`` 里。
"""

import typer
from pathlib import Path
from typing import Union, List
from typing_extensions import Annotated
import rebuild_spec.steps.run_tests
import rebuild_spec.steps.list_tests
import rebuild_spec.steps.build_images
import rebuild_spec.steps.clone
import rebuild_spec.steps.evaluate
import rebuild_spec.steps.lint
import rebuild_spec.steps.publish
from rebuild_spec.core.constants import SPLIT, SPLIT_ALL
from rebuild_spec.core.gitops import current_branch
import subprocess
import yaml
import os
import sys

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="""
    rebuild-spec：按契约文档从零重建一个 Python 库的练习工具链。
    用 -h 查看每个子命令的参数说明。
    """,
)


class Color:
    RESET = "\033[0m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    ORANGE = "\033[95m"


def highlight(text: str, color: str) -> str:
    """给终端文本套上颜色。"""
    return f"{color}{text}{Color.RESET}"


def ensure_cli_on_path() -> None:
    """检查 ``rebuild-spec`` 可执行文件是否在 PATH 上，不在则给出提示。"""
    try:
        subprocess.run(["rebuild-spec", "--help"], capture_output=True)
        return
    except FileNotFoundError:
        typer.echo(
            typer.style(
                "The `rebuild-spec` command was not found on your path!",
                fg=typer.colors.RED,
            )
            + "\n"
            + typer.style(
                "You may need to add it to your path or use `python -m rebuild_spec` as a workaround.",
                fg=typer.colors.RED,
            )
        )
    except PermissionError:
        typer.echo(
            typer.style(
                "The `rebuild-spec` command is not executable!", fg=typer.colors.RED
            )
            + "\n"
            + typer.style(
                "You may need to give it permissions or use `python -m rebuild_spec` as a workaround.",
                fg=typer.colors.RED,
            )
        )
    typer.echo("─" * 80)


def check_valid(one: str, total: Union[list[str], dict[str, list[str]]]) -> None:
    if isinstance(total, dict):
        total = list(total.keys())
    if one not in total:
        valid = ", ".join([highlight(key, Color.ORANGE) for key in total])
        raise typer.BadParameter(
            f"Invalid {highlight('REPO_OR_REPO_SPLIT', Color.RED)}. Must be one of: {valid}",
            param_hint="REPO or REPO_SPLIT",
        )


def write_state_file(dot_file_path: str, config: dict) -> None:
    with open(dot_file_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def read_state_file(dot_file_path: str) -> dict:
    if not os.path.exists(dot_file_path):
        raise FileNotFoundError(
            f"The state file '{dot_file_path}' does not exist. Run `rebuild-spec setup` first."
        )
    with open(dot_file_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def uses_known_split(dataset_name: str) -> bool:
    """数据集名带 rebuild 标记时，才按内置分组做校验。"""
    return "rebuild" in dataset_name.split("/")[-1].lower()


@app.command()
def setup(
    repo_split: str = typer.Argument(
        ...,
        help=f"仓库分组，可选: {', '.join([highlight(key, Color.ORANGE) for key in SPLIT.keys()])}",
    ),
    dataset_name: str = typer.Option(
        "<your-dataset>", help="Hugging Face 数据集名，需自行准备，见 README"
    ),
    dataset_split: str = typer.Option("test", help="数据集 split"),
    base_dir: str = typer.Option("repos/", help="仓库克隆到的根目录"),
    state_file: str = typer.Option(
        ".rebuild-spec.yaml", help="保存运行状态的文件路径"
    ),
) -> None:
    """克隆指定分组下的全部仓库。"""
    ensure_cli_on_path()
    if uses_known_split(dataset_name):
        check_valid(repo_split, SPLIT)

    base_dir = str(Path(base_dir).resolve())

    typer.echo(f"Cloning repository for split: {highlight(repo_split, Color.ORANGE)}")
    typer.echo(f"Dataset name: {highlight(dataset_name, Color.ORANGE)}")
    typer.echo(f"Dataset split: {highlight(dataset_split, Color.ORANGE)}")
    typer.echo(f"Base directory: {highlight(base_dir, Color.ORANGE)}")
    typer.echo(f"State file path: {highlight(state_file, Color.ORANGE)}")

    rebuild_spec.steps.clone.main(
        dataset_name,
        dataset_split,
        repo_split,
        base_dir,
    )

    # 克隆成功后把本次配置写入状态文件，供后续命令使用
    write_state_file(
        state_file,
        {
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "repo_split": repo_split,
            "base_dir": base_dir,
        },
    )


@app.command()
def build(
    num_workers: int = typer.Option(8, help="并发构建的线程数"),
    state_file: str = typer.Option(
        ".rebuild-spec.yaml",
        help="状态文件路径（setup 阶段写入）",
    ),
    verbose: int = typer.Option(
        1,
        "--verbose",
        "-v",
        help="设为 2 可以看到更多日志",
        count=True,
    ),
) -> None:
    """构建 setup 阶段选定分组的全部 Docker 镜像。"""
    ensure_cli_on_path()

    state = read_state_file(state_file)
    if uses_known_split(state["dataset_name"]):
        check_valid(state["repo_split"], SPLIT)

    typer.echo(
        f"Building repository for split: {highlight(state['repo_split'], Color.ORANGE)}"
    )
    typer.echo(f"Dataset name: {highlight(state['dataset_name'], Color.ORANGE)}")
    typer.echo(f"Dataset split: {highlight(state['dataset_split'], Color.ORANGE)}")
    typer.echo(f"Number of workers: {highlight(str(num_workers), Color.ORANGE)}")

    rebuild_spec.steps.build_images.main(
        state["dataset_name"],
        state["dataset_split"],
        state["repo_split"],
        num_workers,
        verbose,
    )


@app.command()
def get_tests(
    repo_name: str = typer.Argument(
        ...,
        help=f"仓库名，可选: {', '.join(highlight(key, Color.ORANGE) for key in SPLIT_ALL)}",
    ),
) -> None:
    """列出某个仓库的全部测试用例 ID。"""
    ensure_cli_on_path()

    rebuild_spec.steps.list_tests.main(repo_name, verbose=1)


@app.command()
def test(
    repo_or_repo_path: str = typer.Argument(
        ..., help="要测试的仓库（名字或本地路径）"
    ),
    test_ids: str = typer.Argument(
        None,
        help='pytest 支持的任意选测写法，请作为一个字符串传入。例: "test_mod.py", "testing/", "test_mod.py::test_func", "-k \'MyClass and not method\'"',
    ),
    branch: Union[str, None] = typer.Option(
        None, help="要测试的分支（必须显式给出，或用 --reference）"
    ),
    backend: str = typer.Option("modal", help="测试后端: local / modal / e2b"),
    timeout: int = typer.Option(1800, help="测试超时时间（秒）"),
    num_cpus: int = typer.Option(1, help="分配的 CPU 数"),
    reference: Annotated[
        bool, typer.Option("--reference", help="测试参考实现对应的提交")
    ] = False,
    coverage: Annotated[
        bool, typer.Option("--coverage", help="同时收集覆盖率信息")
    ] = False,
    rebuild: bool = typer.Option(
        False, "--rebuild", help="强制重建镜像"
    ),
    state_file: str = typer.Option(
        ".rebuild-spec.yaml",
        help="状态文件路径（setup 阶段写入）",
    ),
    verbose: int = typer.Option(
        1,
        "--verbose",
        "-v",
        help="设为 2 可以看到更多日志",
        count=True,
    ),
    stdin: bool = typer.Option(
        False,
        "--stdin",
        help="从标准输入读测试名。例: `echo 'test_mod.py' | rebuild-spec test REPO --branch BRANCH`",
    ),
) -> None:
    """在隔离环境里运行指定仓库分支上的测试。"""
    ensure_cli_on_path()
    state = read_state_file(state_file)
    if repo_or_repo_path.endswith("/"):
        repo_or_repo_path = repo_or_repo_path[:-1]
    if uses_known_split(state["dataset_name"]):
        check_valid(repo_or_repo_path.split("/")[-1], SPLIT)

    if reference:
        branch = "reference"
    else:
        dataset_name = state["dataset_name"].lower()
        if (
            "humaneval" in dataset_name
            or "mbpp" in dataset_name
            or "bigcodebench" in dataset_name
            or "codecontests" in dataset_name
        ):
            branch = repo_or_repo_path
        else:
            if branch is None and not reference:
                git_path = os.path.join(
                    state["base_dir"], repo_or_repo_path.split("/")[-1]
                )
                branch = current_branch(git_path)

    if stdin:
        test_ids = sys.stdin.read()
    elif test_ids is None:
        typer.echo("Error: test_ids must be provided or use --stdin option", err=True)
        raise typer.Exit(code=1)

    if verbose == 2:
        typer.echo(f"Running tests for repository: {repo_or_repo_path}")
        typer.echo(f"Branch: {branch}")
        typer.echo(f"Test IDs: {test_ids}")

    rebuild_spec.steps.run_tests.main(
        state["dataset_name"],
        state["dataset_split"],
        state["base_dir"],
        repo_or_repo_path,
        branch,  # type: ignore
        test_ids,
        coverage,
        backend,
        timeout,
        num_cpus,
        rebuild,
        verbose,
    )


@app.command()
def evaluate(
    branch: Union[str, None] = typer.Option(
        None, help="要评估的分支（必须显式给出，或用 --reference）"
    ),
    backend: str = typer.Option("modal", help="评估后端: local / modal / e2b"),
    timeout: int = typer.Option(1800, help="评估超时时间（秒）"),
    num_cpus: int = typer.Option(1, help="每个任务分配的 CPU 数"),
    num_workers: int = typer.Option(8, help="并发数"),
    reference: Annotated[
        bool, typer.Option("--reference", help="评估参考实现对应的提交")
    ] = False,
    coverage: Annotated[
        bool, typer.Option("--coverage", help="同时收集覆盖率信息")
    ] = False,
    state_file: str = typer.Option(
        ".rebuild-spec.yaml",
        help="状态文件路径（setup 阶段写入）",
    ),
    rebuild: bool = typer.Option(False, "--rebuild", help="强制重建镜像"),
) -> None:
    """对 setup 阶段选定的整个分组做批量评估，并汇总通过率。"""
    ensure_cli_on_path()
    if reference:
        branch = "reference"

    state = read_state_file(state_file)
    if uses_known_split(state["dataset_name"]):
        check_valid(state["repo_split"], SPLIT)

    typer.echo(f"Evaluating repository split: {state['repo_split']}")
    typer.echo(f"Branch: {branch}")

    rebuild_spec.steps.evaluate.main(
        state["dataset_name"],
        state["dataset_split"],
        state["repo_split"],
        state["base_dir"],
        branch,
        coverage,
        backend,
        timeout,
        num_cpus,
        num_workers,
        rebuild,
    )


@app.command()
def lint(
    repo_or_repo_dir: str = typer.Argument(
        ..., help="要检查的仓库（名字或本地路径）"
    ),
    files: Union[List[Path], None] = typer.Option(
        None, help="只检查这些文件；不给则检查整个源码目录"
    ),
    state_file: str = typer.Option(
        ".rebuild-spec.yaml",
        help="状态文件路径（setup 阶段写入）",
    ),
    verbose: int = typer.Option(
        1,
        "--verbose",
        "-v",
        help="设为 2 可以看到更多日志",
        count=True,
    ),
) -> None:
    """用 pre-commit（ruff + pyright）做代码风格检查。"""
    ensure_cli_on_path()
    state = read_state_file(state_file)
    appended_files = None
    if files is not None:
        appended_files = []
        for path in files:
            path = Path(state["base_dir"]) / Path(repo_or_repo_dir) / path
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {str(path)}")
            appended_files.append(path)
    if verbose == 2:
        typer.echo(f"Linting repo: {highlight(str(repo_or_repo_dir), Color.ORANGE)}")
    rebuild_spec.steps.lint.main(
        state["dataset_name"],
        state["dataset_split"],
        repo_or_repo_dir,
        appended_files,
        state["base_dir"],
    )


@app.command()
def save(
    owner: str = typer.Argument(..., help="GitHub 用户或组织名"),
    branch: str = typer.Argument(..., help="要保存的分支"),
    github_token: str = typer.Option(None, help="GitHub token，也可用 GITHUB_TOKEN 环境变量"),
    state_file: str = typer.Option(
        ".rebuild-spec.yaml",
        help="状态文件路径（setup 阶段写入）",
    ),
) -> None:
    """把本地分支推送到自己的 GitHub 账号下留档。"""
    ensure_cli_on_path()
    state = read_state_file(state_file)
    if uses_known_split(state["dataset_name"]):
        check_valid(state["repo_split"], SPLIT)

    typer.echo(f"Saving repository split: {state['repo_split']}")
    typer.echo(f"Owner: {owner}")
    typer.echo(f"Branch: {branch}")

    rebuild_spec.steps.publish.main(
        state["dataset_name"],
        state["dataset_split"],
        state["repo_split"],
        state["base_dir"],
        owner,
        branch,
        github_token,
    )


__all__ = []
