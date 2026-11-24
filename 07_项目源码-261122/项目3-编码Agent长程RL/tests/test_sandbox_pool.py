"""测试 3.5 节：沙箱容器池预热、快照借用与丢弃可写层复位。"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

# 提供轻量 rebuild_spec 占位，确保无 Docker 与完整项目二环境时也可测试容器池状态机
if "rebuild_spec" not in sys.modules:
    rs_mod = types.ModuleType("rebuild_spec")
    rs_core = types.ModuleType("rebuild_spec.core")
    rs_const = types.ModuleType("rebuild_spec.core.constants")
    rs_const.ABSOLUTE_REPO_DIR = "/testbed"
    rs_cont = types.ModuleType("rebuild_spec.core.containers")
    rs_cont.cleanup_container = lambda client, container, logger: container.remove(force=True)
    rs_cont.copy_from_container = MagicMock()
    rs_cont.copy_to_container = MagicMock()
    rs_cont.create_container = lambda client, image_name, container_name, logger, **kw: client.containers.run(
        image=image_name, name=container_name, detach=True
    )
    rs_cont.exec_run_with_timeout = MagicMock(return_value=("ok", False, 0.1))
    rs_cont.image_exists_locally = MagicMock(return_value=True)
    rs_specs = types.ModuleType("rebuild_spec.core.specs")
    rs_specs.build_spec = lambda rec, dtype, absolute=True: types.SimpleNamespace(
        repo_image_key=f"official.{rec['instance_id']}:v0"
    )
    sys.modules["rebuild_spec"] = rs_mod
    sys.modules["rebuild_spec.core"] = rs_core
    sys.modules["rebuild_spec.core.constants"] = rs_const
    sys.modules["rebuild_spec.core.containers"] = rs_cont
    sys.modules["rebuild_spec.core.specs"] = rs_specs

from long_horizon_rl.sandbox_pool import SandboxPool


def _make_mock_docker_client() -> MagicMock:
    client = MagicMock()
    counter = {"n": 0}

    def fake_run(**kwargs):
        counter["n"] += 1
        c = MagicMock()
        c.id = f"cid-{counter['n']}"
        c.name = kwargs.get("name", f"cname-{counter['n']}")
        c.exec_run.return_value = (0, b"ok")
        return c

    client.containers.run.side_effect = fake_run
    return client


def test_sandbox_pool_prewarm_acquire_and_reset() -> None:
    client = _make_mock_docker_client()
    pool = SandboxPool(pool_size_per_task=10, use_docker=True, docker_client=client)
    task = {"instance_id": "tinydb", "src_dir": "tinydb"}

    created = pool.prewarm_task(task)
    assert created == 10
    assert pool.pool_depth("tinydb") == 10

    lease = pool.acquire(task)
    assert lease.from_pool is True
    assert pool.pool_depth("tinydb") == 9

    # 释放后销毁旧容器丢弃可写层，并由池补齐
    pool.release(lease, replenish=False)
    lease.container.remove.assert_called_once_with(force=True)
    pool.close()


def test_scoring_container_uses_official_image_not_pool(tmp_path) -> None:
    client = _make_mock_docker_client()
    pool = SandboxPool(pool_size_per_task=2, use_docker=True, docker_client=client)
    task = {"instance_id": "tinydb", "src_dir": "tinydb"}
    patch_file = tmp_path / "patch.diff"
    patch_file.write_text("", encoding="utf-8")

    score_container = pool.create_scoring_container(task, patch_file)
    assert score_container is not None
    # 官方判分容器镜像与快照池镜像不同
    call_kwargs = client.containers.run.call_args.kwargs
    assert call_kwargs["image"] == "official.tinydb:v0"
    pool.close()


def test_exit_code_marker_is_parsed_from_container_output() -> None:
    from long_horizon_rl.sandbox_pool import _EXIT_MARKER, _split_exit_marker

    out, code, timed_out = _split_exit_marker(f"FAILED test_core.py\n{_EXIT_MARKER}1\n")
    assert out == "FAILED test_core.py"
    assert code == 1
    assert timed_out is False
    assert _split_exit_marker("no marker")[1] == -1
