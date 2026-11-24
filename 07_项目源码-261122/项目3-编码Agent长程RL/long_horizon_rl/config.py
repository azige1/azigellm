"""项目三长程 RL 扩展配置结构与默认超参。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CurriculumConfig:
    """3.1 两阶段 curriculum 配置。"""

    stage: Literal["function", "repo"] = "repo"
    pass_at_8_min: int = 1
    pass_at_8_max: int = 6
    prune_interval_steps: int = 30
    stage1_max_turns: int = 8
    stage2_max_turns: int = 60
    stage1_tools: list[str] = field(default_factory=lambda: ["bash", "read_file", "edit_file"])
    stage2_tools: list[str] = field(
        default_factory=lambda: [
            "bash",
            "read_file",
            "write_file",
            "edit_file",
            "glob_files",
            "grep_text",
            "todo_write",
        ]
    )


@dataclass
class RewardConfig:
    """3.2 隐藏测试通过比例 reward 与篡改检测配置。"""

    mode: Literal["pass_ratio", "binary"] = "pass_ratio"
    enable_tamper_scan: bool = True
    tamper_log_path: str = "logs/tamper_events.jsonl"
    audit_interval_steps: int = 30
    audit_top_k: int = 50
    test_timeout_s: int = 600
    scoring_backend: Literal["docker", "host", "local"] = "docker"


@dataclass
class CompactionConfig:
    """3.3 context compaction 分段训练配置。"""

    enable: bool = True
    window_tokens: int = 32768
    compact_threshold_tokens: int = 27000
    summary_max_tokens: int = 4096
    keep_recent_turns: int = 4
    max_segments: int = 4
    normalize_by_trajectory_tokens: bool = True


@dataclass
class StepAdvantageConfig:
    """3.4 规范化报错 anchor 的 GiGPO step 级分组配置。"""

    enable: bool = True
    gamma: float = 0.95
    omega: float = 0.5
    granularity: Literal["full", "exception_only"] = "full"
    min_group_size: int = 2


@dataclass
class SandboxPoolConfig:
    """3.5 沙箱容器池配置。"""

    enable: bool = True
    pool_size_per_task: int = 10
    tool_timeout_s: int = 300
    max_output_chars: int = 10000
    image_prefix: str = "lhrl.task"
    nano_cpus: int = 2_000_000_000
    mem_limit: str = "4g"
    use_docker: bool = True
    tasks_root: str = "tasks"
