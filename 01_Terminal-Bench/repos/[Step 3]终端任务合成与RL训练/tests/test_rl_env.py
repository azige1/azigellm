"""rl/ 包的单元测试。

skyrl_gym、skyrl_train、ray 与容器运行时全部打桩，可在无 GPU、无
Docker 的机器上运行。
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import yaml

try:
    import datasets as _datasets_mod  # 提前导入，避免 exec 时 pyarrow 重复注册
except ImportError:  # 环境无 datasets 时用桩代替
    _datasets_mod = mock.MagicMock()

# ---------------------------------------------------------------------------
# skyrl_gym / skyrl_train 的最小桩，使 rl/ 模块可以导入
# ---------------------------------------------------------------------------

def _make_stub_modules():
    """在导入 rl/ 之前注册轻量桩模块。"""

    class BaseTextEnvStepOutput:
        def __init__(self, *, observations, reward, done, metadata):
            self.observations = observations
            self.reward = reward
            self.done = done
            self.metadata = metadata

    class BaseTextEnv:
        def __init__(self):
            self.turns = 0

    # skyrl_gym 层级
    skyrl_gym = types.ModuleType("skyrl_gym")
    skyrl_gym_envs = types.ModuleType("skyrl_gym.envs")
    skyrl_gym_envs_base = types.ModuleType("skyrl_gym.envs.base_text_env")
    skyrl_gym_envs_base.BaseTextEnv = BaseTextEnv
    skyrl_gym_envs_base.BaseTextEnvStepOutput = BaseTextEnvStepOutput
    skyrl_gym_envs.base_text_env = skyrl_gym_envs_base
    skyrl_gym_envs.register = lambda id, entry_point: None
    skyrl_gym.envs = skyrl_gym_envs

    # skyrl_train 层级
    skyrl_train = types.ModuleType("skyrl_train")
    skyrl_train_utils = types.ModuleType("skyrl_train.utils")
    skyrl_train_utils.initialize_ray = lambda cfg: None
    skyrl_train_entrypoints = types.ModuleType("skyrl_train.entrypoints")
    skyrl_train_main_base = types.ModuleType("skyrl_train.entrypoints.main_base")
    skyrl_train_main_base.BasePPOExp = mock.MagicMock()
    skyrl_train_main_base.config_dir = "/tmp"
    skyrl_train_main_base.validate_cfg = lambda cfg: None
    skyrl_train_entrypoints.main_base = skyrl_train_main_base
    skyrl_train.utils = skyrl_train_utils
    skyrl_train.entrypoints = skyrl_train_entrypoints

    # ray 桩
    ray = types.ModuleType("ray")
    ray.remote = lambda *a, **kw: (lambda fn: fn)  # 空装饰器
    ray.get = lambda x: x
    ray.init = lambda **kw: None

    for name, mod in [
        ("skyrl_gym", skyrl_gym),
        ("skyrl_gym.envs", skyrl_gym_envs),
        ("skyrl_gym.envs.base_text_env", skyrl_gym_envs_base),
        ("skyrl_train", skyrl_train),
        ("skyrl_train.utils", skyrl_train_utils),
        ("skyrl_train.entrypoints", skyrl_train_entrypoints),
        ("skyrl_train.entrypoints.main_base", skyrl_train_main_base),
        ("ray", ray),
    ]:
        sys.modules.setdefault(name, mod)

    return BaseTextEnv, BaseTextEnvStepOutput


BaseTextEnv, BaseTextEnvStepOutput = _make_stub_modules()


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

RL_DIR = Path(__file__).parent.parent / "rl"
REPO_ROOT = Path(__file__).parent.parent


def _make_extras(tmp_path, max_time=600, max_turns=16, verbose=False):
    """构造指向临时任务目录的 extras 字典。"""
    task_dir = tmp_path / "task_001"
    task_dir.mkdir()
    (task_dir / "environment").mkdir()
    (task_dir / "tests").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (task_dir / "tests" / "test_final_state.py").write_text("def test_done(): pass\n")
    return {
        "extra_info": {
            "task_dir": str(task_dir),
            "max_time": max_time,
            "verbose": verbose,
        },
        "max_turns": max_turns,
    }


def _import_rl_env():
    """全新导入 TerminalTaskEnv（桩模块须已就位）。"""
    if "rl.env" in sys.modules:
        del sys.modules["rl.env"]
    sys.path.insert(0, str(REPO_ROOT))
    from rl.env import TerminalTaskEnv
    return TerminalTaskEnv


# ---------------------------------------------------------------------------
# YAML 配置校验
# ---------------------------------------------------------------------------

class TestTrainConfigs:
    @pytest.mark.parametrize("name", [
        "base.yaml",
        "base_qwen.yaml",
        "base_qwen3_8b.yaml",
        "base_t4.yaml",
    ])
    def test_yaml_parses(self, name):
        path = RL_DIR / "confs" / name
        assert path.exists(), f"Config missing: {name}"
        doc = yaml.safe_load(path.read_text())
        assert isinstance(doc, dict)

    def test_base_has_data_section(self):
        doc = yaml.safe_load((RL_DIR / "confs" / "base.yaml").read_text())
        assert "data" in doc
        assert "train_data" in doc["data"]
        assert "val_data" in doc["data"]

    def test_base_has_trainer_section(self):
        doc = yaml.safe_load((RL_DIR / "confs" / "base.yaml").read_text())
        assert "trainer" in doc
        assert "algorithm" in doc["trainer"]

    def test_base_has_generator_section(self):
        doc = yaml.safe_load((RL_DIR / "confs" / "base.yaml").read_text())
        assert "generator" in doc
        assert doc["generator"]["max_turns"] == 16

    def test_base_env_class_is_terminal_task(self):
        doc = yaml.safe_load((RL_DIR / "confs" / "base.yaml").read_text())
        assert doc["environment"]["env_class"] == "terminal_task"

    def test_base_train_batch_size_positive(self):
        doc = yaml.safe_load((RL_DIR / "confs" / "base.yaml").read_text())
        assert doc["trainer"]["train_batch_size"] > 0


# ---------------------------------------------------------------------------
# TerminalTaskEnv 构造
# ---------------------------------------------------------------------------

class TestTerminalTaskEnvInit:
    def test_basic_construction(self, tmp_path):
        TerminalTaskEnv = _import_rl_env()
        mock_env = mock.MagicMock()
        mock_env.instance_name = None

        with mock.patch("rl.env.DockerTaskSandbox", return_value=mock_env):
            env = TerminalTaskEnv(extras=_make_extras(tmp_path))

        assert env._initialized is False
        assert env.reward == 0
        assert env.max_turns == 16

    def test_max_time_string_converted_to_int(self, tmp_path):
        TerminalTaskEnv = _import_rl_env()
        extras = _make_extras(tmp_path)
        extras["extra_info"]["max_time"] = "300"

        with mock.patch("rl.env.DockerTaskSandbox"):
            env = TerminalTaskEnv(extras=extras)

        assert env.max_time == 300
        assert isinstance(env.max_time, int)

    def test_max_output_length_default(self, tmp_path):
        TerminalTaskEnv = _import_rl_env()
        with mock.patch("rl.env.DockerTaskSandbox"):
            env = TerminalTaskEnv(extras=_make_extras(tmp_path))
        assert env.max_output_length == 50_000

    def test_custom_max_output_length(self, tmp_path):
        TerminalTaskEnv = _import_rl_env()
        extras = _make_extras(tmp_path)
        extras["extra_info"]["max_output_length"] = 1000
        with mock.patch("rl.env.DockerTaskSandbox"):
            env = TerminalTaskEnv(extras=extras)
        assert env.max_output_length == 1000


# ---------------------------------------------------------------------------
# TerminalTaskEnv.step 逻辑
# ---------------------------------------------------------------------------

class TestTerminalTaskEnvStep:
    def _env(self, tmp_path, mock_container=None):
        TerminalTaskEnv = _import_rl_env()
        if mock_container is None:
            mock_container = mock.MagicMock()
            mock_container.instance_name = "ctr-1"
            mock_container.initialize.return_value = True
            mock_container.verbose = False
        with mock.patch("rl.env.DockerTaskSandbox", return_value=mock_container):
            env = TerminalTaskEnv(extras=_make_extras(tmp_path))
        env.env = mock_container
        return env

    def test_init_failure_returns_done(self, tmp_path):
        mock_container = mock.MagicMock()
        mock_container.instance_name = None
        mock_container.initialize.return_value = False
        mock_container.verbose = False
        env = self._env(tmp_path, mock_container)

        result = env.step("<command>ls</command>")

        assert result.done is True
        assert result.reward == 0
        assert "Failed to initialize" in result.observations[0]["content"]

    def test_done_action_triggers_test_run(self, tmp_path):
        mock_container = mock.MagicMock()
        mock_container.instance_name = "ctr-1"
        mock_container.initialize.return_value = True
        mock_container.run_final_tests.return_value = (True, "passed")
        mock_container.verbose = False
        env = self._env(tmp_path, mock_container)
        env._initialized = True

        result = env.step("<action>done</action>")

        assert result.done is True
        assert result.reward == 1
        assert result.metadata["goal_reached"] is True
        mock_container.cleanup.assert_called_once()

    def test_command_success_observation(self, tmp_path):
        mock_container = mock.MagicMock()
        mock_container.instance_name = "ctr-1"
        mock_container.initialize.return_value = True
        mock_container.exec.return_value = (True, "file.txt\n")
        mock_container.verbose = False
        env = self._env(tmp_path, mock_container)
        env._initialized = True

        result = env.step("<command>ls</command>")

        assert "Command executed successfully" in result.observations[0]["content"]
        assert result.done is False

    def test_command_failure_observation(self, tmp_path):
        mock_container = mock.MagicMock()
        mock_container.instance_name = "ctr-1"
        mock_container.initialize.return_value = True
        mock_container.exec.return_value = (False, "No such file\n")
        mock_container.verbose = False
        env = self._env(tmp_path, mock_container)
        env._initialized = True

        result = env.step("<command>cat missing.txt</command>")

        assert "Command failed" in result.observations[0]["content"]

    def test_output_truncation(self, tmp_path):
        long_output = "x" * 100_000
        mock_container = mock.MagicMock()
        mock_container.instance_name = "ctr-1"
        mock_container.initialize.return_value = True
        mock_container.exec.return_value = (True, long_output)
        mock_container.verbose = False
        env = self._env(tmp_path, mock_container)
        env._initialized = True
        env.max_output_length = 1000

        result = env.step("<command>cat big.txt</command>")

        obs = result.observations[0]["content"]
        assert "Output truncated" in obs

    def test_invalid_action_returns_parse_error(self, tmp_path):
        mock_container = mock.MagicMock()
        mock_container.instance_name = "ctr-1"
        mock_container.initialize.return_value = True
        mock_container.verbose = False
        env = self._env(tmp_path, mock_container)
        env._initialized = True

        result = env.step("just some text with no tags")

        assert "Could not parse" in result.observations[0]["content"]

    def test_max_turns_terminates(self, tmp_path):
        mock_container = mock.MagicMock()
        mock_container.instance_name = "ctr-1"
        mock_container.initialize.return_value = True
        mock_container.exec.return_value = (True, "ok")
        mock_container.run_final_tests.return_value = (False, "fail")
        mock_container.verbose = False
        env = self._env(tmp_path, mock_container)
        env._initialized = True
        env.max_turns = 1  # 第一步即触顶

        result = env.step("<command>ls</command>")

        assert result.done is True

    def test_cleanup_called_after_done(self, tmp_path):
        mock_container = mock.MagicMock()
        mock_container.instance_name = "ctr-1"
        mock_container.initialize.return_value = True
        mock_container.exec.return_value = (True, "ok")
        mock_container.run_final_tests.return_value = (False, "")
        mock_container.verbose = False
        env = self._env(tmp_path, mock_container)
        env._initialized = True
        env.max_turns = 1

        env.step("<command>ls</command>")

        mock_container.cleanup.assert_called_once()
        assert env._initialized is False


# ---------------------------------------------------------------------------
# prepare_data 辅助函数（不触发 __main__）
# ---------------------------------------------------------------------------

def _load_prepare_data_namespace():
    """以 exec 方式加载 rl/prepare_data.py 的主逻辑之前的命名空间。"""
    source = (RL_DIR / "prepare_data.py").read_text()
    pre_main = source.split('if __name__ == "__main__":')[0]

    mock_rollout = mock.MagicMock()
    mock_rollout.AGENT_SYSTEM_PROMPT = "sys"
    mock_rollout.AGENT_USER_TEMPLATE = "{task_description}"
    mock_rollout.parse_agent_action = lambda x: {"type": "done", "command": None}

    ns: dict = {}
    with mock.patch.dict(sys.modules, {
        "task_factory": mock.MagicMock(),
        "task_factory.rollout_agent": mock_rollout,
        "datasets": _datasets_mod,
    }):
        exec(compile(pre_main, "prepare_data.py", "exec"), ns)
    return ns


class TestPrepareData:
    def test_build_task_image_missing_dockerfile(self, tmp_path):
        """任务目录缺 environment/Dockerfile 时返回失败。"""
        ns = _load_prepare_data_namespace()

        task_name = "task_001_abc"
        (tmp_path / task_name).mkdir()

        result_name, success = ns["build_task_image"](task_name, str(tmp_path))
        assert success is False
        assert result_name == task_name

    def test_build_task_image_success(self, tmp_path):
        """Dockerfile 存在且构建命令成功时返回成功。"""
        ns = _load_prepare_data_namespace()

        task_name = "task_002_xyz"
        task_path = tmp_path / task_name
        (task_path / "environment").mkdir(parents=True)
        (task_path / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")

        with mock.patch.object(ns["subprocess"], "run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0, stderr="")
            result_name, success = ns["build_task_image"](task_name, str(tmp_path))

        assert success is True
        assert result_name == task_name


# ---------------------------------------------------------------------------
# prepare_data 的任务过滤逻辑（apptainer / harbor 两种来源）
# ---------------------------------------------------------------------------

def _make_harbor_task(tasks_root: Path, name: str, num_success: int) -> Path:
    """构造带 solution/solution.json 的最小 Harbor 任务目录。"""
    d = tasks_root / name
    d.mkdir(parents=True)
    (d / "task.json").write_text(json.dumps({"description": f"desc {name}"}))
    sol_dir = d / "solution"
    sol_dir.mkdir()
    (sol_dir / "solution.json").write_text(
        json.dumps({"num_success": num_success, "num_runs": 8, "pass_at_k": {"1": 0.1}})
    )
    return d


def _make_apptainer_task(tasks_root: Path, name: str, pass_at_16: float) -> Path:
    """构造带 solutions/o3_summary.json 的最小 SIF 任务目录。"""
    d = tasks_root / name
    d.mkdir(parents=True)
    (d / "task.json").write_text(json.dumps({"description": f"desc {name}"}))
    sol_dir = d / "solutions"
    sol_dir.mkdir()
    (sol_dir / "o3_summary.json").write_text(
        json.dumps({"pass_at_k": {"16": pass_at_16}})
    )
    return d


def _run_filter(tmp_path: Path, source: str) -> list[str]:
    """执行过滤逻辑的最小复刻，返回过滤后的任务名列表。"""
    filter_only = """
import os, json, random
from pathlib import Path

task_dir = str(task_dir_arg)
task_dir_names = [f for f in os.listdir(task_dir) if "task" in f]

source_mode = source_arg

if source_mode == "harbor":
    def _harbor_solved(f):
        p = Path(task_dir) / f / "solution" / "solution.json"
        if not p.exists():
            return False
        try:
            return json.load(open(p)).get("num_success", 0) > 0
        except (json.JSONDecodeError, OSError):
            return False
    task_dir_names = [f for f in task_dir_names if _harbor_solved(f)]
else:
    task_dir_names = [
        f for f in task_dir_names
        if (Path(task_dir) / f / "solutions" / "o3_summary.json").exists()
    ]
    task_dir_names = [
        f for f in task_dir_names
        if json.load(open(Path(task_dir) / f / "solutions" / "o3_summary.json"))["pass_at_k"]["16"] > 0
    ]

task_dir_names = list(sorted(task_dir_names))
"""
    ns = {"task_dir_arg": tmp_path, "source_arg": source}
    exec(compile(filter_only, "filter_test", "exec"), ns)
    return ns["task_dir_names"]


class TestPrepareDataFiltering:
    def test_harbor_keeps_solved_tasks(self, tmp_path):
        _make_harbor_task(tmp_path, "task_001", num_success=2)
        _make_harbor_task(tmp_path, "task_002", num_success=0)
        _make_harbor_task(tmp_path, "task_003", num_success=1)

        result = _run_filter(tmp_path, "harbor")

        assert "task_001" in result
        assert "task_003" in result
        assert "task_002" not in result

    def test_harbor_excludes_tasks_without_solution_json(self, tmp_path):
        _make_harbor_task(tmp_path, "task_001", num_success=3)
        no_sol = tmp_path / "task_002"
        no_sol.mkdir()
        (no_sol / "task.json").write_text("{}")

        result = _run_filter(tmp_path, "harbor")

        assert "task_001" in result
        assert "task_002" not in result

    def test_harbor_result_is_sorted(self, tmp_path):
        _make_harbor_task(tmp_path, "task_003", num_success=1)
        _make_harbor_task(tmp_path, "task_001", num_success=1)
        _make_harbor_task(tmp_path, "task_002", num_success=1)

        result = _run_filter(tmp_path, "harbor")

        assert result == sorted(result)

    def test_apptainer_keeps_passing_tasks(self, tmp_path):
        _make_apptainer_task(tmp_path, "task_001", pass_at_16=0.5)
        _make_apptainer_task(tmp_path, "task_002", pass_at_16=0.0)
        _make_apptainer_task(tmp_path, "task_003", pass_at_16=1.0)

        result = _run_filter(tmp_path, "apptainer")

        assert "task_001" in result
        assert "task_003" in result
        assert "task_002" not in result

    def test_apptainer_excludes_tasks_without_o3_summary(self, tmp_path):
        _make_apptainer_task(tmp_path, "task_001", pass_at_16=0.5)
        no_sol = tmp_path / "task_002"
        no_sol.mkdir()
        (no_sol / "task.json").write_text("{}")

        result = _run_filter(tmp_path, "apptainer")

        assert "task_001" in result
        assert "task_002" not in result

    def test_harbor_empty_dir_returns_empty(self, tmp_path):
        result = _run_filter(tmp_path, "harbor")
        assert result == []

    def test_apptainer_empty_dir_returns_empty(self, tmp_path):
        result = _run_filter(tmp_path, "apptainer")
        assert result == []
