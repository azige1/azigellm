import sys
from types import ModuleType

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

from multi_teacher_distillation.domains.project3_adapter import (
    IssueFixContainerConfig,
    Project3AdapterConfig,
    Project3RepoAdapter,
    attach_segment_routing_metadata,
)


def test_project3_adapter_loads_configurable_dotted_module(monkeypatch):
    fake_pkg = ModuleType("custom_project3")
    fake_loop_mod = ModuleType("custom_project3.loop")
    fake_sbx_mod = ModuleType("custom_project3.sbx")

    class DummyRepoLoop:
        pass

    class DummySandboxPool:
        def __init__(self, pool_size: int = 10):
            self.pool_size = pool_size

    fake_loop_mod.CustomLoop = DummyRepoLoop
    fake_sbx_mod.CustomPool = DummySandboxPool
    monkeypatch.setitem(sys.modules, "custom_project3", fake_pkg)
    monkeypatch.setitem(sys.modules, "custom_project3.loop", fake_loop_mod)
    monkeypatch.setitem(sys.modules, "custom_project3.sbx", fake_sbx_mod)

    cfg = Project3AdapterConfig(
        agent_loop_path="custom_project3.loop:CustomLoop",
        sandbox_pool_path="custom_project3.sbx:CustomPool",
    )
    adapter = Project3RepoAdapter(cfg)
    assert adapter.resolve_loop_class() is DummyRepoLoop
    pool = adapter.build_sandbox_pool(pool_size=8)
    assert pool.pool_size == 8


def test_segment_routing_metadata_and_issue_fix_container_config():
    outputs = [
        AgentLoopOutput(prompt_ids=[1, 2], response_ids=[3, 4], response_mask=[1, 1], metrics=AgentLoopMetrics()),
        AgentLoopOutput(prompt_ids=[1, 2], response_ids=[5, 6], response_mask=[1, 1], metrics=AgentLoopMetrics()),
    ]
    parent_meta = {"domain": "repo_impl", "teacher": "repo_impl", "teacher_version": "p3_rl_v1", "prompt_id": "r-1"}
    annotated = attach_segment_routing_metadata(
        outputs,
        parent_meta=parent_meta,
        response_length=16,
        terminal_ids={99},
        is_distillation=True,
    )
    assert [item.extra_fields["segment_index"] for item in annotated] == [0, 1]
    assert all(item.extra_fields["domain"] == "repo_impl" for item in annotated)
    assert all(item.extra_fields["teacher_version"] == "p3_rl_v1" for item in annotated)
    assert all(item.reward_score == 0.0 for item in annotated)

    container_cfg = IssueFixContainerConfig.from_mapping({"warm_pool_size": 10, "snapshot_reset": True})
    assert container_cfg.warm_pool_size == 10
    assert container_cfg.snapshot_reset is True
