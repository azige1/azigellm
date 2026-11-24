import importlib
import os
from dataclasses import dataclass, field
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

from ..rollout_masking import is_truncated
from ..routing import propagate_segment_metadata
from ..termination import terminal_token_ids

DEFAULT_PROJECT3_AGENT_LOOP_PATH = "long_horizon_rl.agent_loop:CodingCompactionAgentLoop"
DEFAULT_PROJECT3_SANDBOX_POOL_PATH = "long_horizon_rl.sandbox_pool:SandboxPool"

# 每个 rollout worker 进程共用一个沙箱容器池，避免每个 loop 实例各自预热容器。
_PROCESS_SANDBOX_POOL: Any = None


def load_dotted_object(target: str) -> Any:
    """按 'module.path:Attr' 或 'module.path.Attr' 动态加载项目三适配对象。"""
    if ":" in target:
        module_name, attr_name = target.split(":", 1)
    elif "." in target:
        module_name, attr_name = target.rsplit(".", 1)
    else:
        raise ValueError(f"模块路径格式无效，期望 'pkg.mod:Obj' 或 'pkg.mod.Obj'，实际为: {target!r}")
    module = importlib.import_module(module_name)
    obj = module
    for part in attr_name.split("."):
        obj = getattr(obj, part)
    return obj


@dataclass(frozen=True)
class Project3AdapterConfig:
    """项目三仓库级 agent loop 与沙箱的独立适配配置（包名通过配置项指定）。"""

    agent_loop_path: str = DEFAULT_PROJECT3_AGENT_LOOP_PATH
    sandbox_pool_path: str = DEFAULT_PROJECT3_SANDBOX_POOL_PATH
    compaction_threshold_tokens: int = 27000
    max_segments: int = 4
    tail_turns: int = 4
    max_turns: int = 60
    tool_timeout_s: float = 300.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, Any]) -> "Project3AdapterConfig":
        return cls(
            agent_loop_path=str(
                kwargs.get(
                    "agent_loop_path",
                    os.environ.get("PROJECT3_AGENT_LOOP_PATH", DEFAULT_PROJECT3_AGENT_LOOP_PATH),
                )
            ),
            sandbox_pool_path=str(
                kwargs.get(
                    "sandbox_pool_path",
                    os.environ.get("PROJECT3_SANDBOX_POOL_PATH", DEFAULT_PROJECT3_SANDBOX_POOL_PATH),
                )
            ),
            compaction_threshold_tokens=int(kwargs.get("compaction_threshold_tokens", 27000)),
            max_segments=int(kwargs.get("max_segments", 4)),
            tail_turns=int(kwargs.get("tail_turns", 4)),
            max_turns=int(kwargs.get("max_turns", 60)),
            tool_timeout_s=float(kwargs.get("tool_timeout_s", 300.0)),
            extra=dict(kwargs.get("project3_kwargs", {}) or {}),
        )


@dataclass(frozen=True)
class IssueFixContainerConfig:
    """issue 修复领域的容器配置接口，供本地沙箱池与评测新容器复用。"""

    image_prefix: str = "swebench/instance"
    work_dir: str = "/testbed"
    warm_pool_size: int = 10
    snapshot_reset: bool = True
    test_command: str = "pytest -x"
    timeout_s: float = 300.0
    max_turns: int = 120
    compaction_threshold_tokens: int = 27000
    max_segments: int = 4
    tail_turns: int = 4
    env: dict[str, str] = field(default_factory=lambda: {"PYTHONDONTWRITEBYTECODE": "1"})

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "IssueFixContainerConfig":
        cfg = dict(data or {})
        return cls(
            image_prefix=str(cfg.get("image_prefix", "swebench/instance")),
            work_dir=str(cfg.get("work_dir", "/testbed")),
            warm_pool_size=int(cfg.get("warm_pool_size", 10)),
            snapshot_reset=bool(cfg.get("snapshot_reset", True)),
            test_command=str(cfg.get("test_command", "pytest -x")),
            timeout_s=float(cfg.get("timeout_s", 300.0)),
            max_turns=int(cfg.get("max_turns", 120)),
            compaction_threshold_tokens=int(cfg.get("compaction_threshold_tokens", 56000)),
            max_segments=int(cfg.get("max_segments", 4)),
            tail_turns=int(cfg.get("tail_turns", 4)),
            env=dict(cfg.get("env", {"PYTHONDONTWRITEBYTECODE": "1"})),
        )


def attach_segment_routing_metadata(
    outputs: list[AgentLoopOutput],
    parent_meta: dict[str, Any],
    response_length: int,
    terminal_ids: set[int],
    is_distillation: bool,
) -> list[AgentLoopOutput]:
    """为分段输出的每一段显式写入 domain、teacher、teacher_version 与 segment_index，并检查截断。"""
    if not outputs:
        return []
    annotated: list[AgentLoopOutput] = []
    for seg_idx, output in enumerate(outputs):
        truncated = is_truncated(output.response_ids, response_length, terminal_ids)
        if truncated:
            output.response_mask = [0] * len(output.response_ids)
        if truncated or is_distillation:
            output.reward_score = 0.0
        extra = propagate_segment_metadata(dict(output.extra_fields), parent_meta, segment_index=seg_idx)
        extra["truncated"] = truncated
        extra["prompt_id"] = str(parent_meta.get("prompt_id", output.extra_fields.get("prompt_id", "")))
        output.extra_fields = extra
        annotated.append(output)
    return annotated


class Project3RepoAdapter:
    """隔离仓库级领域对项目三 agent loop 与沙箱池的动态导入与实例化。"""

    def __init__(self, adapter_config: Project3AdapterConfig) -> None:
        self.adapter_config = adapter_config
        self._loop_cls = None
        self._sandbox_pool_cls = None

    def resolve_loop_class(self) -> type[AgentLoopBase]:
        if self._loop_cls is None:
            self._loop_cls = load_dotted_object(self.adapter_config.agent_loop_path)
        return self._loop_cls

    def resolve_sandbox_pool_class(self) -> Any:
        if self._sandbox_pool_cls is None:
            self._sandbox_pool_cls = load_dotted_object(self.adapter_config.sandbox_pool_path)
        return self._sandbox_pool_cls

    def build_sandbox_pool(self, **kwargs: Any) -> Any:
        pool_cls = self.resolve_sandbox_pool_class()
        merged = {**self.adapter_config.extra, **kwargs}
        return pool_cls(**merged)


class RepoImplLoop(AgentLoopBase):
    """仓库级实现领域 rollout：经独立适配模块调用项目三 agent loop 与沙箱，并保证分段路由传递。"""

    domain = "repo_impl"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.response_length = int(self.rollout_config.response_length)
        self.adapter_config = Project3AdapterConfig.from_kwargs(kwargs)
        self.adapter = Project3RepoAdapter(self.adapter_config)
        self._delegate_args = args
        self._delegate_kwargs = kwargs
        self._delegate: AgentLoopBase | None = None

    def _get_delegate(self) -> AgentLoopBase:
        if self._delegate is None:
            global _PROCESS_SANDBOX_POOL
            loop_cls = self.adapter.resolve_loop_class()
            if _PROCESS_SANDBOX_POOL is None:
                _PROCESS_SANDBOX_POOL = self.adapter.build_sandbox_pool()
            self._delegate = loop_cls(
                *self._delegate_args, sandbox_pool=_PROCESS_SANDBOX_POOL, **self._delegate_kwargs
            )
        return self._delegate

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs: Any):
        delegate = self._get_delegate()
        result = await delegate.run(sampling_params, priority=priority, **kwargs)
        if not result:
            return []
        outputs = list(result) if isinstance(result, list) else [result]
        extra_info = kwargs.get("extra_info", {}) or {}
        parent_meta = {
            "domain": str(kwargs.get("domain", extra_info.get("domain", self.domain))),
            "teacher": str(kwargs.get("teacher", extra_info.get("teacher", self.domain))),
            "teacher_version": str(kwargs.get("teacher_version", extra_info.get("teacher_version", ""))),
            "prompt_id": str(extra_info.get("prompt_id", kwargs.get("prompt_id", ""))),
        }
        is_distill = bool(self.config.get("distillation", {}).get("enabled", False))
        annotated = attach_segment_routing_metadata(
            outputs,
            parent_meta=parent_meta,
            response_length=self.response_length,
            terminal_ids=terminal_token_ids(self.tokenizer),
            is_distillation=is_distill,
        )
        return annotated if len(annotated) > 1 or isinstance(result, list) else annotated[0]
