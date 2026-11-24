"""隐藏测试通过比例 reward 与判分前工作区清理（3.2 节）。

- 仓库级任务：判分前删除模型写入的 tests 目录、pytest.ini、tox.ini、conftest.py、
  setup.cfg 中的 pytest 段与 pyproject.toml 中的 pytest 配置，再在基于官方镜像新建的
  容器中还原隐藏测试、执行 ``pip install -e .`` 与仓库原有测试，取通过比例
  $r = n_{\\text{pass}} / n_{\\text{total}}$；
- 函数级任务：本地子进程调用项目一 ``codeharvest.execution.local_exec_service.RepoExecService``
  执行隐藏等价测试并取测试通过比例；
- 篡改检测：对最终代码做 9 条规则 AST + 文本静态扫描，命中任一条记 reward 0 并写篡改日志。
"""

from __future__ import annotations

import configparser
import io
import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from long_horizon_rl.tamper_scan import (
    TamperScanResult,
    append_tamper_log,
    scan_code_snippet,
    scan_workspace,
)

SCRUB_CONFIG_FILES = ("pytest.ini", "tox.ini", "conftest.py")


@dataclass
class RewardOutcome:
    """单条轨迹的判分结果。"""

    reward: Optional[float]
    pass_rate: float
    num_passed: int
    num_total: int
    tamper_flag: bool = False
    rules_hit: list[str] = field(default_factory=list)
    infra_failure: bool = False
    error: str = ""

    def to_reward_extra_info(self) -> dict[str, Any]:
        return {
            "pass_rate": self.pass_rate,
            "num_passed": self.num_passed,
            "num_total": self.num_total,
            "tamper_flag": self.tamper_flag,
            "tamper_rules": ",".join(self.rules_hit),
            "infra_failure": self.infra_failure,
        }


def strip_setup_cfg_pytest_sections(setup_cfg_text: str) -> str:
    """移除 setup.cfg 中的 [tool:pytest] 与 [pytest] 配置段，保留其余打包段。"""
    parser = configparser.ConfigParser()
    parser.read_string(setup_cfg_text)
    changed = False
    for section in ("tool:pytest", "pytest"):
        if parser.has_section(section):
            parser.remove_section(section)
            changed = True
    if not changed:
        return setup_cfg_text
    buf = io.StringIO()
    parser.write(buf)
    return buf.getvalue()


def strip_pyproject_pytest_sections(pyproject_text: str) -> str:
    """移除 pyproject.toml 中的 [tool.pytest...] 配置块。"""
    lines = pyproject_text.splitlines(keepends=True)
    kept: list[str] = []
    skipping = False
    header_re = re.compile(r"^\s*\[([^\]]+)\]\s*$")
    for line in lines:
        m = header_re.match(line)
        if m:
            section = m.group(1).strip()
            skipping = section == "tool.pytest" or section.startswith("tool.pytest.")
        if not skipping:
            kept.append(line)
    return "".join(kept)


def scrub_model_test_artifacts(workspace_root: Path, test_dir: str = "tests") -> list[str]:
    """判分前清理模型写入的测试目录与 pytest 配置文件，返回被清理的相对路径列表。"""
    root = Path(workspace_root)
    removed: list[str] = []

    model_tests = root / test_dir
    if model_tests.exists():
        shutil.rmtree(model_tests, ignore_errors=True)
        removed.append(test_dir)
    # 若仓库同时存在默认 tests/ 目录也一并清理
    default_tests = root / "tests"
    if test_dir != "tests" and default_tests.exists():
        shutil.rmtree(default_tests, ignore_errors=True)
        removed.append("tests")

    for fname in SCRUB_CONFIG_FILES:
        target = root / fname
        if target.exists():
            target.unlink()
            removed.append(fname)

    setup_cfg = root / "setup.cfg"
    if setup_cfg.is_file():
        orig = setup_cfg.read_text(encoding="utf-8", errors="replace")
        cleaned = strip_setup_cfg_pytest_sections(orig)
        if cleaned != orig:
            setup_cfg.write_text(cleaned, encoding="utf-8")
            removed.append("setup.cfg:[pytest]")

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        orig = pyproject.read_text(encoding="utf-8", errors="replace")
        cleaned = strip_pyproject_pytest_sections(orig)
        if cleaned != orig:
            pyproject.write_text(cleaned, encoding="utf-8")
            removed.append("pyproject.toml:[tool.pytest]")

    return removed


def compute_reward_from_counts(
    num_passed: int,
    num_total: int,
    *,
    mode: Literal["pass_ratio", "binary"] = "pass_ratio",
    tamper_result: Optional[TamperScanResult] = None,
) -> tuple[float, float]:
    """根据通过数与篡改扫描结果计算 ``(reward, pass_rate)``。"""
    pass_rate = float(num_passed / num_total) if num_total > 0 else 0.0
    if tamper_result is not None and tamper_result.flagged:
        return 0.0, pass_rate
    if mode == "binary":
        reward = 1.0 if (num_total > 0 and num_passed == num_total) else 0.0
        return reward, pass_rate
    return pass_rate, pass_rate


def evaluate_function_task(
    task: dict[str, Any],
    candidate_code: str,
    *,
    mode: Literal["pass_ratio", "binary"] = "pass_ratio",
    enable_tamper_scan: bool = True,
    tamper_log_path: Optional[Path] = None,
    step: int = 0,
    trajectory_id: str = "",
) -> RewardOutcome:
    """使用项目一 ``RepoExecService`` 本地判分函数级任务并返回通过比例 reward。"""
    task_id = str(task.get("task_id", "unknown"))
    tamper_res = scan_code_snippet(candidate_code) if enable_tamper_scan else TamperScanResult(flagged=False)
    if tamper_res.flagged and tamper_log_path is not None:
        append_tamper_log(
            tamper_log_path,
            task_id=task_id,
            step=step,
            scan_result=tamper_res,
            trajectory_id=trajectory_id,
        )

    source_ref = task.get("source_ref") or {}
    repo_id = source_ref.get("repo_id")
    file_path = source_ref.get("file_path")
    test_source = task.get("test_source", "")
    func_name = task.get("function_name") or _infer_function_name(task.get("signature", ""))
    repo_path = task.get("repo_path") or repo_id

    if not repo_path or not file_path or not func_name or not test_source:
        return RewardOutcome(
            reward=None,
            pass_rate=0.0,
            num_passed=0,
            num_total=0,
            infra_failure=True,
            error="函数级任务缺少 repo_path/file_path/function_name/test_source",
        )

    from codeharvest.execution.local_exec_service import RepoExecService

    service = RepoExecService()
    try:
        r1 = service.setup_repo(json.dumps({"repo_id": repo_id, "repo_path": repo_path}))
        if r1.get("error"):
            return RewardOutcome(reward=None, pass_rate=0.0, num_passed=0, num_total=0, infra_failure=True, error=r1["error"])
        r2 = service.setup_function(json.dumps({"funclass_names": [func_name], "file_path": file_path}))
        if r2.get("error"):
            return RewardOutcome(reward=None, pass_rate=0.0, num_passed=0, num_total=0, infra_failure=True, error=r2["error"])
        r3 = service.setup_test(json.dumps({"generated_tests": {"0": test_source}}))
        if r3.get("error"):
            return RewardOutcome(reward=None, pass_rate=0.0, num_passed=0, num_total=0, infra_failure=True, error=r3["error"])
        r_init = service.init()
        if r_init.get("error"):
            return RewardOutcome(reward=None, pass_rate=0.0, num_passed=0, num_total=0, infra_failure=True, error=r_init["error"])

        service.setup_codegen_mode()
        exec_res = service.execute(candidate_code)
        if exec_res.get("error"):
            reward, pass_rate = compute_reward_from_counts(0, 1, mode=mode, tamper_result=tamper_res)
            return RewardOutcome(
                reward=reward,
                pass_rate=pass_rate,
                num_passed=0,
                num_total=1,
                tamper_flag=tamper_res.flagged,
                rules_hit=tamper_res.rules_hit,
                error=exec_res["error"],
            )

        submit_res = service.submit()
        logs = json.loads(submit_res.get("logs") or "{}")
        run_logs = logs.get("run_tests_logs") or {}
        num_passed = sum(int(v.get("passed_count", 0)) for v in run_logs.values())
        num_failed = sum(int(v.get("failed_count", 0)) for v in run_logs.values())
        num_errored = sum(int(v.get("errored_count", 0)) for v in run_logs.values())
        num_total = max(1, num_passed + num_failed + num_errored)
        reward, pass_rate = compute_reward_from_counts(num_passed, num_total, mode=mode, tamper_result=tamper_res)
        return RewardOutcome(
            reward=reward,
            pass_rate=pass_rate,
            num_passed=num_passed,
            num_total=num_total,
            tamper_flag=tamper_res.flagged,
            rules_hit=tamper_res.rules_hit,
        )
    except Exception as exc:  # noqa: BLE001
        return RewardOutcome(
            reward=None,
            pass_rate=0.0,
            num_passed=0,
            num_total=0,
            infra_failure=True,
            error=repr(exc),
        )
    finally:
        service.stop_server()


def _is_candidate_timeout(error_text: str) -> bool:
    """测试执行超时由候选实现引起；容器、依赖与镜像错误属于基础设施失败。"""
    lowered = error_text.lower()
    return "timeout" in lowered or "timed out" in lowered or "超时" in error_text


def evaluate_repo_workspace(
    workspace_dir: Path,
    task: dict[str, Any],
    *,
    mode: Literal["pass_ratio", "binary"] = "pass_ratio",
    enable_tamper_scan: bool = True,
    tamper_log_path: Optional[Path] = None,
    scoring_backend: Literal["docker", "host", "local"] = "host",
    sandbox_pool: Any = None,
    timeout_s: int = 600,
    step: int = 0,
    trajectory_id: str = "",
) -> RewardOutcome:
    """清理模型测试产物、执行 9 条篡改静态扫描并在全新隔离环境中运行隐藏测试。"""
    task_id = str(task.get("instance_id", "unknown"))
    test_dir = (task.get("test") or {}).get("test_dir", "tests")

    # 1. 先清理模型写入的测试与配置，再扫描实现代码，避免对将删除的 tests/ 误报。
    #    清理失败（如模型写出无法解析的 setup.cfg）属于候选产物问题，记 0 分。
    try:
        scrub_model_test_artifacts(workspace_dir, test_dir=test_dir)
    except Exception as exc:  # noqa: BLE001
        return RewardOutcome(
            reward=0.0,
            pass_rate=0.0,
            num_passed=0,
            num_total=0,
            error=f"候选工作区配置无法清理: {str(exc)[:300]}",
        )
    tamper_res = scan_workspace(workspace_dir) if enable_tamper_scan else TamperScanResult(flagged=False)
    if tamper_res.flagged:
        if tamper_log_path is not None:
            append_tamper_log(
                tamper_log_path,
                task_id=task_id,
                step=step,
                scan_result=tamper_res,
                trajectory_id=trajectory_id,
            )
        # 命中篡改规则：不再执行候选代码，直接记 0 分并保留篡改标记
        return RewardOutcome(
            reward=0.0,
            pass_rate=0.0,
            num_passed=0,
            num_total=0,
            tamper_flag=True,
            rules_hit=tamper_res.rules_hit,
        )

    test_ids_file = Path(task["local_path"]).parent / "test_ids.txt" if task.get("local_path") else None
    test_ids = (
        [line.strip() for line in test_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if test_ids_file is not None and test_ids_file.exists()
        else []
    )

    # 2. Docker 新建官方容器判分路径（3.2 + 3.5 节）
    if scoring_backend == "docker" and sandbox_pool is not None:
        return _score_in_fresh_docker_container(
            workspace_dir=workspace_dir,
            task=task,
            test_ids=test_ids,
            sandbox_pool=sandbox_pool,
            tamper_res=tamper_res,
            mode=mode,
            timeout_s=timeout_s,
        )

    # 3. 宿主机隔离判分路径（复用项目二 agent_rollout.judge + rebuild_spec）
    from agent_rollout.judge import judge_run

    with tempfile.TemporaryDirectory(prefix="lhrl-score-") as out_tmp:
        try:
            score = judge_run(
                workspace=workspace_dir,
                task=task,
                run_id=trajectory_id or "rl",
                out_dir=Path(out_tmp),
                dataset_name=str(Path(task["local_path"]).parent.parent),
                timeout=timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            return RewardOutcome(
                reward=None,
                pass_rate=0.0,
                num_passed=0,
                num_total=len(test_ids),
                infra_failure=True,
                error=str(exc)[:500],
            )

    error_text = str(score.get("error") or "")
    if error_text and _is_candidate_timeout(error_text):
        # 候选实现导致测试超时（如死循环），按未通过计 0 分
        return RewardOutcome(
            reward=0.0,
            pass_rate=0.0,
            num_passed=0,
            num_total=len(test_ids),
            error=error_text,
        )
    if error_text:
        return RewardOutcome(
            reward=None,
            pass_rate=0.0,
            num_passed=0,
            num_total=len(test_ids),
            infra_failure=True,
            error=str(score["error"]),
        )

    num_passed = int(score.get("num_passed", 0))
    num_total = int(score.get("num_tests", len(test_ids)))
    reward, pass_rate = compute_reward_from_counts(num_passed, num_total, mode=mode, tamper_result=tamper_res)
    return RewardOutcome(
        reward=reward,
        pass_rate=pass_rate,
        num_passed=num_passed,
        num_total=num_total,
        tamper_flag=tamper_res.flagged,
        rules_hit=tamper_res.rules_hit,
    )


def _score_in_fresh_docker_container(
    *,
    workspace_dir: Path,
    task: dict[str, Any],
    test_ids: list[str],
    sandbox_pool: Any,
    tamper_res: TamperScanResult,
    mode: Literal["pass_ratio", "binary"],
    timeout_s: int,
) -> RewardOutcome:
    """在基于官方镜像新建的容器内还原隐藏测试、执行 ``pip install -e .`` 并跑 pytest。"""
    import git

    from agent_rollout.judge import commit_run_branch
    from rebuild_spec.core.containers import exec_run_with_timeout
    from rebuild_spec.core.gitops import diff_between_commits
    from rebuild_spec.core.report_parser import parse_report

    container = None
    try:
        branch = commit_run_branch(workspace_dir, "score")
        repo = git.Repo(workspace_dir)
        patch = diff_between_commits(repo, task["base_commit"], repo.commit(branch).hexsha)

        with tempfile.TemporaryDirectory(prefix="lhrl-docker-score-") as tmp:
            tmp_path = Path(tmp)
            patch_file = tmp_path / "patch.diff"
            patch_file.write_text(patch, encoding="utf-8", errors="ignore")
            container = sandbox_pool.create_scoring_container(task, patch_file)

            test_dir = (task.get("test") or {}).get("test_dir", "tests")
            test_cmd = (task.get("test") or {}).get("test_cmd", f"python -m pytest {test_dir}")
            hidden_dir = task.get("hidden_tests_dir")
            if hidden_dir and Path(hidden_dir).exists():
                sandbox_pool.collect_from_container  # type: ignore[truthy-function]
                from rebuild_spec.core.containers import copy_to_container

                for sub in Path(hidden_dir).iterdir():
                    copy_to_container(container, sub, Path("/hidden_tests") / sub.name)

            eval_cmd = (
                "bash -c 'cd /testbed && source .venv/bin/activate && "
                f"git reset --hard {task['base_commit']} && git apply --allow-empty /patch.diff && "
                f"rm -rf {test_dir} pytest.ini tox.ini conftest.py setup.cfg pyproject.toml && "
                "cp -r /hidden_tests/. /testbed/ && uv pip install -e . && "
                f"{test_cmd} --json-report --json-report-file=report.json --continue-on-collection-errors'"
            )
            _, timed_out, _ = exec_run_with_timeout(container, eval_cmd, timeout=timeout_s)
            if timed_out:
                reward, pass_rate = compute_reward_from_counts(
                    0, max(1, len(test_ids)), mode=mode, tamper_result=tamper_res
                )
                return RewardOutcome(
                    reward=reward,
                    pass_rate=pass_rate,
                    num_passed=0,
                    num_total=len(test_ids),
                    tamper_flag=tamper_res.flagged,
                    rules_hit=tamper_res.rules_hit,
                    error="隐藏测试执行超时",
                )

            report_local = tmp_path / "report.json"
            sandbox_pool.collect_from_container(container, Path("/testbed/report.json"), report_local)
            summary = parse_report(report_local, test_ids)
            num_passed = int(summary["num_passed"])
            num_total = int(summary["num_tests"])
            reward, pass_rate = compute_reward_from_counts(num_passed, num_total, mode=mode, tamper_result=tamper_res)
            return RewardOutcome(
                reward=reward,
                pass_rate=pass_rate,
                num_passed=num_passed,
                num_total=num_total,
                tamper_flag=tamper_res.flagged,
                rules_hit=tamper_res.rules_hit,
            )
    except Exception as exc:  # noqa: BLE001
        return RewardOutcome(
            reward=None,
            pass_rate=0.0,
            num_passed=0,
            num_total=len(test_ids),
            infra_failure=True,
            error=str(exc)[:500],
        )
    finally:
        if container is not None:
            sandbox_pool.destroy_container(container)


def _infer_function_name(signature: str) -> str:
    m = re.search(r"(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)", signature or "")
    return m.group(1) if m else ""
