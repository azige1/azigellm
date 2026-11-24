"""支持上下文压缩分段的多轮编码 AgentLoop 与 TQ Worker/Manager 适配（3.3 + 3.4 + 3.5 节）。

关键行为：
1. 工具定义与动作格式沿用项目二 ``agent_rollout.tools.TOOLS_SCHEMA``（Hermes 工具调用）；
2. 使用 verl Continuous Token 接口（``ct_build_initial_tokens``、``ct_merge_assistant_token``、
   ``ct_merge_context_msg``）维护每段序列；
3. 累计 token 达到窗口约 85% 时插入九段摘要指令，由策略自己生成摘要（摘要 token 的
   ``response_mask == 1`` 并进入 loss），之后以 ``[system, task_spec, summary, 最近 4 轮原文]``
   开启新段，每条轨迹至多 4 段；
4. 返回 ``list[AgentLoopOutput]``，基础设施失败返回 ``[]`` 丢弃该 session；
5. 记录每个工具交互步的 token 区间与规范化报错 anchor，供 ``CodingPPOTrainer`` 计算
   GiGPO step 级 advantage。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

import ray
from agent_rollout.compact_history import (
    CONTINUATION_PREFIX,
    KEEP_RECENT_TURNS,
    LAST_READ_CONTENT_CHARS,
    find_last_read,
    recent_turns_start,
)
from agent_rollout.tools import TOOLS_SCHEMA, ToolResult, Workspace, execute_tool
from trajectory_tools.prompts.nine_section_summary import nine_section_prompt
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQ
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

from long_horizon_rl.error_anchor import extract_error_anchor
from long_horizon_rl.reward_eval import (
    RewardOutcome,
    evaluate_function_task,
    evaluate_repo_workspace,
)
from long_horizon_rl.sandbox_pool import ContainerLease, SandboxPool

logger = logging.getLogger(__name__)

_WORKER_SANDBOX_POOL: Optional[SandboxPool] = None


def set_worker_sandbox_pool(pool: Optional[SandboxPool]) -> None:
    """设置当前 Ray worker 进程共享的沙箱容器池。"""
    global _WORKER_SANDBOX_POOL
    _WORKER_SANDBOX_POOL = pool


def get_worker_sandbox_pool() -> Optional[SandboxPool]:
    """读取当前 Ray worker 进程共享的沙箱容器池。"""
    return _WORKER_SANDBOX_POOL


def format_compaction_request(
    messages: list[dict[str, Any]],
    *,
    last_read_content: str = "",
) -> str:
    """构造插入给策略的九段摘要生成指令（复用项目二模板与最近读取文件提示）。"""
    prompt = nine_section_prompt().strip()
    last_read_path, _ = find_last_read(messages)
    reminder = ""
    if last_read_path and last_read_content:
        reminder = (
            f"\n\nLast read file: {last_read_path}\n"
            f"Content snippet:\n{last_read_content[:LAST_READ_CONTENT_CHARS]}"
        )
    return (
        f"{prompt}\n\n"
        "Please write the 9-section recap directly based on the conversation above so far. "
        "Do not emit tool calls in this turn."
        f"{reminder}"
    )


def build_compacted_segment_messages(
    system_message: dict[str, Any],
    task_spec_message: dict[str, Any],
    messages: list[dict[str, Any]],
    summary_text: str,
    *,
    keep_recent_turns: int = KEEP_RECENT_TURNS,
) -> list[dict[str, Any]]:
    """按 3.3 节规则组装压缩后新段的上下文：system + task spec + 摘要 + 最近 4 轮原文。"""
    keep_from = recent_turns_start(messages, keep_turns=keep_recent_turns)
    # 若前两条本身就是 system 与 task_spec，确保最近原文不会重复包含系统或任务消息
    keep_from = max(2, keep_from)
    recent_messages = [dict(m) for m in messages[keep_from:]]
    summary_msg = {
        "role": "user",
        "content": f"{CONTINUATION_PREFIX}\n\n{summary_text.strip()}",
    }
    return [dict(system_message), dict(task_spec_message), summary_msg, *recent_messages]


def _extract_code_block_or_text(text: str) -> str:
    """从函数级任务的最终回复中提取 Python 代码。"""
    matches = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


# 容器内文件工具脚本：参数为 JSON 字符串与工具名，路径限制在 /testbed 内。
_CONTAINER_FILE_TOOL_SCRIPT = """
import fnmatch, json, pathlib, re, sys
root = pathlib.Path('/testbed').resolve()
args = json.loads(sys.argv[1])
op = sys.argv[2]
p = (root / args.get('path', '.')).resolve()
if not str(p).startswith(str(root)):
    print('路径越出工作区')
    sys.exit(1)
if op == 'read_file':
    lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
    off = int(args.get('offset', 0))
    lim = args.get('limit')
    end = None if lim is None else off + int(lim)
    print('\\n'.join(f'{i + 1 + off}\\t{l}' for i, l in enumerate(lines[off:end])))
elif op == 'write_file':
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(args['content'], encoding='utf-8')
    print(f"已写入 {args['path']}（{len(args['content'])} 字符）")
elif op == 'edit_file':
    txt = p.read_text(encoding='utf-8', errors='replace')
    cnt = txt.count(args['old_string'])
    if cnt != 1:
        print(f'old_string 匹配 {cnt} 处，需要恰好 1 处')
        sys.exit(1)
    p.write_text(txt.replace(args['old_string'], args['new_string'], 1), encoding='utf-8')
    print(f"已替换 {args['path']} 中的 1 处内容")
elif op == 'glob_files':
    pat = args['pattern']
    ms = sorted(str(f.relative_to(root)) for f in root.rglob('*')
                if f.is_file() and '.git' not in f.parts and fnmatch.fnmatch(str(f.relative_to(root)), pat))
    print('\\n'.join(ms[:500]) or '(无匹配)')
elif op == 'grep_text':
    rx = re.compile(args['pattern'])
    fs = [p] if p.is_file() else [f for f in p.rglob('*') if f.is_file() and '.git' not in f.parts]
    hits = []
    for f in fs:
        for i, l in enumerate(f.read_text(encoding='utf-8', errors='replace').splitlines()):
            if rx.search(l):
                hits.append(f'{f.relative_to(root)}:{i + 1}: {l.strip()}')
    print('\\n'.join(hits[:500]) or '(无匹配)')
else:
    print(f'未知工具 {op}')
    sys.exit(1)
"""


class _DockerLeaseWorkspace:
    """把项目二 7 个工具的操作映射到预热 Docker 容器工作区（/testbed）。"""

    def __init__(self, pool: SandboxPool, lease: ContainerLease, timeout_s: int = 300):
        self.pool = pool
        self.lease = lease
        self.timeout_s = timeout_s
        self.todo_items: list[str] = []
        self.local_mirror_dir: Optional[Path] = None

    def execute(self, name: str, arguments_json: str) -> ToolResult:
        try:
            args = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as exc:
            return ToolResult(f"参数不是合法 JSON：{exc}", is_error=True)

        if name == "todo_write":
            items = list(args.get("items") or [])
            self.todo_items = items
            return ToolResult(f"已更新规划清单（{len(items)} 项）")

        if name == "bash":
            cmd = str(args.get("command", ""))
            if "hidden_tests" in cmd:
                return ToolResult("不允许访问该路径", is_error=True)
            out, code, timed_out = self.pool.exec_in_lease(self.lease, cmd, timeout_s=self.timeout_s)
            if timed_out:
                return ToolResult(f"命令超时（{self.timeout_s} 秒）已被终止\n{out[:10000]}", is_error=True)
            return ToolResult(f"{out[:10000]}\n[退出码 {code}]", is_error=code != 0)

        # 文件读/写/编辑/搜索在容器内执行等价 python 单行脚本，保证路径越界检查一致
        payload = json.dumps(args, ensure_ascii=False)
        py_helper = _CONTAINER_FILE_TOOL_SCRIPT
        import shlex

        cmd = f"python -c {shlex.quote(py_helper)} {shlex.quote(payload)} {shlex.quote(name)}"
        out, code, timed_out = self.pool.exec_in_lease(self.lease, cmd, timeout_s=self.timeout_s)
        if timed_out:
            return ToolResult(f"工具执行超时（{self.timeout_s} 秒）", is_error=True)
        return ToolResult(out.strip(), is_error=code != 0)

    def read_file_content(self, rel_path: str) -> str:
        res = self.execute("read_file", json.dumps({"path": rel_path, "offset": 0, "limit": 200}))
        return "" if res.is_error else res.content

    def export_workspace_to_local(self) -> Path:
        """从容器打包拷贝 /testbed 到本地临时目录用于静态扫描与判分。"""
        if self.local_mirror_dir is not None:
            shutil.rmtree(self.local_mirror_dir, ignore_errors=True)
        mirror = Path(tempfile.mkdtemp(prefix="lhrl-mirror-"))
        tar_local = mirror / "testbed.tar"
        stream, _ = self.lease.container.get_archive("/testbed")
        with tar_local.open("wb") as f:
            for chunk in stream:
                f.write(chunk)
        import tarfile

        with tarfile.open(tar_local, "r") as tf:
            tf.extractall(mirror)
        tar_local.unlink(missing_ok=True)
        self.local_mirror_dir = mirror
        return mirror / "testbed"


class CodingAgentLoopWorker(AgentLoopWorkerTQ.__ray_actor_class__):
    """扩展 TQ Worker，按通用约束继承 ``AgentLoopWorkerTQ.__ray_actor_class__``。

    持有进程级共享的 ``SandboxPool`` 并注册给每个 ``CodingCompactionAgentLoop`` 实例。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sb_cfg = self.config.get("coding_rl", {}).get("sandbox", {})
        self.sandbox_pool = SandboxPool(
            pool_size_per_task=int(sb_cfg.get("pool_size_per_task", 10)),
            image_prefix=str(sb_cfg.get("image_prefix", "lhrl.task")),
            nano_cpus=int(sb_cfg.get("nano_cpus", 2_000_000_000)),
            mem_limit=str(sb_cfg.get("mem_limit", "4g")),
            use_docker=bool(sb_cfg.get("use_docker", True)),
        )
        set_worker_sandbox_pool(self.sandbox_pool)

    def prewarm_tasks(self, task_records: list[dict[str, Any]]) -> None:
        """为下一步已确定的任务列表在后台预热沙箱容器。"""
        if self.sandbox_pool is not None and task_records:
            self.sandbox_pool.prewarm_tasks_async(task_records)


class CodingAgentLoopManager(AgentLoopManagerTQ):
    """使用 ``CodingAgentLoopWorker`` 的 TQ AgentLoopManager。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = ray.remote(CodingAgentLoopWorker)

    def prewarm_next_tasks(self, task_records: list[dict[str, Any]]) -> None:
        """向全部 AgentLoopWorker 分发下一批任务的容器预热请求。"""
        if not task_records or not getattr(self, "agent_loop_workers", None):
            return
        per_worker = max(1, len(task_records) // len(self.agent_loop_workers))
        for idx, worker in enumerate(self.agent_loop_workers):
            subset = task_records[idx::len(self.agent_loop_workers)] or task_records[:per_worker]
            worker.prewarm_tasks.remote(subset)


@register("coding_compaction_agent")
class CodingCompactionAgentLoop(AgentLoopBase):
    """支持 9 段摘要上下文压缩与 GiGPO step 记录的多轮编码 AgentLoop。"""

    def __init__(self, *args, sandbox_pool: Optional[SandboxPool] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sandbox_pool = sandbox_pool if sandbox_pool is not None else get_worker_sandbox_pool()
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_schemas_by_name = {s["function"]["name"]: s for s in TOOLS_SCHEMA}

        coding_cfg = self.config.get("coding_rl", {})
        comp_cfg = coding_cfg.get("compaction", {})
        rew_cfg = coding_cfg.get("reward", {})
        cur_cfg = coding_cfg.get("curriculum", {})
        sb_cfg = coding_cfg.get("sandbox", {})
        step_cfg = coding_cfg.get("step_advantage", {})

        self.compact_threshold_tokens = int(comp_cfg.get("compact_threshold_tokens", 27000))
        self.summary_max_tokens = int(comp_cfg.get("summary_max_tokens", 4096))
        self.keep_recent_turns = int(comp_cfg.get("keep_recent_turns", KEEP_RECENT_TURNS))
        self.max_segments = int(comp_cfg.get("max_segments", 4))
        self.compaction_enabled = bool(comp_cfg.get("enable", True))

        self.reward_mode = str(rew_cfg.get("mode", "pass_ratio"))
        self.enable_tamper_scan = bool(rew_cfg.get("enable_tamper_scan", True))
        self.tamper_log_path = Path(str(rew_cfg.get("tamper_log_path", "logs/tamper_events.jsonl")))
        self.scoring_backend = str(rew_cfg.get("scoring_backend", "docker"))
        self.test_timeout_s = int(rew_cfg.get("test_timeout_s", 600))

        self.default_stage = str(cur_cfg.get("stage", "repo"))
        self.stage1_max_turns = int(cur_cfg.get("stage1_max_turns", 8))
        self.stage2_max_turns = int(cur_cfg.get("stage2_max_turns", 60))
        self.tool_timeout_s = int(sb_cfg.get("tool_timeout_s", 300))
        self.anchor_granularity = str(step_cfg.get("granularity", "full"))

        self.prompt_length = int(self.rollout_config.prompt_length)
        self.response_length = int(self.rollout_config.response_length)

    def _select_tool_schemas(self, tool_selection: Optional[list[str]]) -> list[dict[str, Any]]:
        if not tool_selection:
            return list(TOOLS_SCHEMA)
        return [self.tool_schemas_by_name[name] for name in tool_selection if name in self.tool_schemas_by_name]

    def _prepare_runtime_env(
        self, stage: str, task: dict[str, Any]
    ) -> tuple[Optional[Workspace], Optional[ContainerLease], Optional[_DockerLeaseWorkspace], Optional[Path]]:
        """为本条轨迹准备函数级本地沙箱或仓库级预热沙箱。"""
        if stage == "function":
            work_dir = Path(tempfile.mkdtemp(prefix="lhrl-func-"))
            target_file = (task.get("source_ref") or {}).get("file_path", "solution.py")
            stub_path = work_dir / target_file
            stub_path.parent.mkdir(parents=True, exist_ok=True)
            stub_path.write_text(f"{task.get('signature', '')}\n    raise NotImplementedError\n", encoding="utf-8")
            return Workspace(root=work_dir), None, None, work_dir

        if self.sandbox_pool is not None and self.sandbox_pool.use_docker:
            lease = self.sandbox_pool.acquire(task)
            docker_ws = _DockerLeaseWorkspace(self.sandbox_pool, lease, timeout_s=self.tool_timeout_s)
            return None, lease, docker_ws, None

        from agent_rollout.run_loop import prepare_workspace

        repo_dir, ws = prepare_workspace(task)
        return ws, None, None, repo_dir.parent

    def _execute_in_env(
        self,
        *,
        local_ws: Optional[Workspace],
        docker_ws: Optional[_DockerLeaseWorkspace],
        name: str,
        arguments_json: str,
    ) -> ToolResult:
        if docker_ws is not None:
            return docker_ws.execute(name, arguments_json)
        assert local_ws is not None
        return execute_tool(local_ws, name, arguments_json)

    def _read_last_read_file(
        self,
        *,
        local_ws: Optional[Workspace],
        docker_ws: Optional[_DockerLeaseWorkspace],
        messages: list[dict[str, Any]],
    ) -> str:
        last_path, _ = find_last_read(messages)
        if not last_path:
            return ""
        if docker_ws is not None:
            return docker_ws.read_file_content(last_path)
        if local_ws is not None:
            candidate = local_ws.root / last_path
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8", errors="replace")
        return ""

    async def _generate_once(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        metrics: dict[str, Any],
    ) -> TokenOutput:
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0
        return output

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> list[AgentLoopOutput]:
        """执行完整多轮 rollout（含策略自摘要压缩分段），返回 ``list[AgentLoopOutput]``。"""
        raw_prompt = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info") or {}
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        stage = str(extra_info.get("stage", self.default_stage))
        task_payload_raw = extra_info.get("task_payload_json")
        task: dict[str, Any] = json.loads(task_payload_raw) if isinstance(task_payload_raw, str) else (task_payload_raw or {})
        tool_selection = extra_info.get("tool_selection")
        active_schemas = self._select_tool_schemas(tool_selection)
        max_turns = self.stage1_max_turns if stage == "function" else self.stage2_max_turns
        global_step = int(kwargs.get("global_steps", 0) or 0)
        trajectory_id = uuid.uuid4().hex[:12]

        local_ws: Optional[Workspace] = None
        lease: Optional[ContainerLease] = None
        docker_ws: Optional[_DockerLeaseWorkspace] = None
        temp_root: Optional[Path] = None

        try:
            local_ws, lease, docker_ws, temp_root = await self.loop.run_in_executor(
                None, lambda: self._prepare_runtime_env(stage, task)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("沙箱初始化失败，丢弃该轨迹 task=%s: %s", extra_info.get("task_id"), exc)
            return []

        metrics: dict[str, Any] = {"generate_sequences": 0.0, "tool_calls": 0.0, "compute_score": 0.0}
        segments: list[AgentLoopOutput] = []
        system_msg = dict(raw_prompt[0])
        task_spec_msg = dict(raw_prompt[1]) if len(raw_prompt) > 1 else {"role": "user", "content": ""}

        messages: list[dict[str, Any]] = [dict(m) for m in raw_prompt]
        seg_initial_prompt_ids = await self.ct_build_initial_tokens(messages, tools=active_schemas)
        runtime_token_ids = list(seg_initial_prompt_ids)
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        seg_step_spans: list[dict[str, Any]] = []

        assistant_turns_total = 0
        user_turns_total = 0
        global_step_counter = 0
        compaction_triggered = False
        last_assistant_text = ""
        # GiGPO 的 anchor 是动作之前看到的报错状态：上一轮工具返回的规范化报错
        state_anchor: Optional[str] = None

        def _finalize_segment(is_compacted_boundary: bool) -> AgentLoopOutput:
            nonlocal response_mask, response_logprobs, seg_step_spans
            resp_len = len(response_mask)
            prompt_ids = runtime_token_ids[: len(runtime_token_ids) - resp_len]
            response_ids = runtime_token_ids[-resp_len:] if resp_len > 0 else []
            out = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids[: self.response_length],
                response_mask=response_mask[: self.response_length],
                response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
                num_turns=assistant_turns_total + user_turns_total + 1,
                metrics=AgentLoopMetrics(**{k: v for k, v in metrics.items() if k in AgentLoopMetrics.model_fields}),
                extra_fields={
                    "segment_index": len(segments),
                    "compaction_triggered": is_compacted_boundary,
                    "step_spans_json": json.dumps(seg_step_spans, ensure_ascii=False),
                    "trajectory_id": trajectory_id,
                },
            )
            return out

        try:
            while assistant_turns_total < max_turns:
                req_id = f"{trajectory_id}-s{len(segments)}-t{assistant_turns_total}"
                gen_out = await self._generate_once(req_id, runtime_token_ids, sampling_params, metrics)
                assistant_turns_total += 1

                turn_start_tok = len(response_mask)
                merge_res, response_mask, response_logprobs_opt = await self.ct_merge_assistant_token(
                    runtime_token_ids,
                    gen_out.token_ids,
                    response_mask,
                    response_logprobs if (response_logprobs or gen_out.log_probs) else None,
                    assistant_logprobs=gen_out.log_probs if gen_out.log_probs else None,
                )
                runtime_token_ids = merge_res.token_ids
                response_logprobs = response_logprobs_opt or []
                turn_end_tok = len(response_mask)

                active_tool_schemas_obj = [
                    # ToolParser 只需 name/description/parameters 结构；使用占位模型属性即可
                    type("SchemaProxy", (), {"name": s["function"]["name"], "model_dump": lambda self, **_: s})()
                    for s in active_schemas
                ]
                assistant_content, tool_calls = await self.tool_parser.extract_tool_calls(
                    gen_out.token_ids, active_tool_schemas_obj
                )
                last_assistant_text = assistant_content or self.tokenizer.decode(
                    gen_out.token_ids, skip_special_tokens=True
                )

                assistant_msg: dict[str, Any] = {"role": "assistant", "content": assistant_content or ""}
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.tool_call_id or f"call_{assistant_turns_total}_{idx}",
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for idx, tc in enumerate(tool_calls)
                    ]
                messages.append(assistant_msg)

                if not tool_calls:
                    break

                # 执行本轮工具调用并提取规范化报错 anchor
                previous_messages = list(messages)
                tool_messages: list[dict[str, Any]] = []
                turn_anchors: list[str] = []
                with simple_timer("tool_calls", metrics):
                    for idx, tc in enumerate(tool_calls):
                        call_id = assistant_msg["tool_calls"][idx]["id"]
                        res = await self.loop.run_in_executor(
                            None,
                            lambda c_name=tc.name, c_args=tc.arguments: self._execute_in_env(
                                local_ws=local_ws, docker_ws=docker_ws, name=c_name, arguments_json=c_args
                            ),
                        )
                        tool_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": res.content,
                            }
                        )
                        anchor = extract_error_anchor(res.content, granularity=self.anchor_granularity)
                        if anchor:
                            turn_anchors.append(anchor)

                seg_step_spans.append(
                    {
                        "segment_idx": len(segments),
                        "start_token": turn_start_tok,
                        "end_token": turn_end_tok,
                        "anchor": state_anchor,
                        "step_index": global_step_counter,
                    }
                )
                # 本轮工具返回的报错成为下一轮动作所面对的状态；多个失败只取第一个
                state_anchor = turn_anchors[0] if turn_anchors else None
                global_step_counter += 1

                messages.extend(tool_messages)
                user_turns_total += 1
                merge_ctx, response_mask, response_logprobs_opt = await self.ct_merge_context_msg(
                    previous_messages,
                    messages,
                    runtime_token_ids,
                    response_mask,
                    response_logprobs if response_logprobs else None,
                    tools=active_schemas,
                )
                runtime_token_ids = merge_ctx.token_ids
                response_logprobs = response_logprobs_opt or []

                if len(response_mask) >= self.response_length:
                    break

                # 3.3 检查是否到达上下文压缩阈值（默认 27K / 约 85% 窗口）
                if self.compaction_enabled and len(runtime_token_ids) >= self.compact_threshold_tokens:
                    if len(segments) + 1 >= self.max_segments:
                        # 已达最大 4 段上限，截断当前轨迹并按当前状态判分
                        break

                    compaction_triggered = True
                    last_read_content = await self.loop.run_in_executor(
                        None,
                        lambda: self._read_last_read_file(local_ws=local_ws, docker_ws=docker_ws, messages=messages),
                    )
                    compact_instruction = format_compaction_request(messages, last_read_content=last_read_content)
                    prev_before_compact = list(messages)
                    messages.append({"role": "user", "content": compact_instruction})
                    merge_inst, response_mask, response_logprobs_opt = await self.ct_merge_context_msg(
                        prev_before_compact,
                        messages,
                        runtime_token_ids,
                        response_mask,
                        response_logprobs if response_logprobs else None,
                        tools=active_schemas,
                    )
                    runtime_token_ids = merge_inst.token_ids
                    response_logprobs = response_logprobs_opt or []

                    # 由策略自己生成九段摘要；merge_assistant_token 使其 mask=1 并计入 loss
                    summary_sampling = {**sampling_params, "max_tokens": self.summary_max_tokens}
                    sum_req_id = f"{trajectory_id}-sum{len(segments)}"
                    sum_out = await self._generate_once(sum_req_id, runtime_token_ids, summary_sampling, metrics)
                    merge_sum, response_mask, response_logprobs_opt = await self.ct_merge_assistant_token(
                        runtime_token_ids,
                        sum_out.token_ids,
                        response_mask,
                        response_logprobs if (response_logprobs or sum_out.log_probs) else None,
                        assistant_logprobs=sum_out.log_probs if sum_out.log_probs else None,
                    )
                    runtime_token_ids = merge_sum.token_ids
                    response_logprobs = response_logprobs_opt or []

                    summary_text = self.tokenizer.decode(sum_out.token_ids, skip_special_tokens=True)
                    segments.append(_finalize_segment(is_compacted_boundary=True))

                    # 重组新段上下文：system + task spec + 摘要 + 最近 4 轮原文
                    messages = build_compacted_segment_messages(
                        system_msg,
                        task_spec_msg,
                        prev_before_compact,
                        summary_text,
                        keep_recent_turns=self.keep_recent_turns,
                    )
                    seg_initial_prompt_ids = await self.ct_build_initial_tokens(messages, tools=active_schemas)
                    runtime_token_ids = list(seg_initial_prompt_ids)
                    response_mask = []
                    response_logprobs = []
                    seg_step_spans = []

            if response_mask or not segments:
                segments.append(_finalize_segment(is_compacted_boundary=False))

            # 在全新判分环境评测终局结果（含删 tests + 9 条篡改静态扫描）
            with simple_timer("compute_score", metrics):
                outcome = await self.loop.run_in_executor(
                    None,
                    lambda: self._score_trajectory(
                        stage=stage,
                        task=task,
                        local_ws=local_ws,
                        docker_ws=docker_ws,
                        last_reply=last_assistant_text,
                        global_step=global_step,
                        trajectory_id=trajectory_id,
                    ),
                )

            if outcome.infra_failure or outcome.reward is None:
                logger.warning("判分基础设施失败，丢弃轨迹 %s: %s", trajectory_id, outcome.error)
                return []

            extra_reward_info = outcome.to_reward_extra_info()
            extra_reward_info["compaction_triggered"] = compaction_triggered
            extra_reward_info["num_segments"] = len(segments)
            for seg in segments:
                seg.reward_score = float(outcome.reward)
                seg.extra_fields["reward_extra_info"] = dict(extra_reward_info)
                seg.extra_fields["compaction_triggered"] = compaction_triggered
            return segments
        finally:
            if lease is not None and self.sandbox_pool is not None:
                self.sandbox_pool.release(lease, replenish=True)
            if docker_ws is not None and docker_ws.local_mirror_dir is not None:
                shutil.rmtree(docker_ws.local_mirror_dir, ignore_errors=True)
            if temp_root is not None and temp_root.exists():
                shutil.rmtree(temp_root, ignore_errors=True)

    def _score_trajectory(
        self,
        *,
        stage: str,
        task: dict[str, Any],
        local_ws: Optional[Workspace],
        docker_ws: Optional[_DockerLeaseWorkspace],
        last_reply: str,
        global_step: int,
        trajectory_id: str,
    ) -> RewardOutcome:
        if stage == "function":
            candidate_code = ""
            if local_ws is not None:
                target_file = (task.get("source_ref") or {}).get("file_path", "")
                if target_file and (local_ws.root / target_file).is_file():
                    candidate_code = (local_ws.root / target_file).read_text(encoding="utf-8", errors="replace")
            if not candidate_code.strip():
                candidate_code = _extract_code_block_or_text(last_reply)
            return evaluate_function_task(
                task,
                candidate_code,
                mode=self.reward_mode,
                enable_tamper_scan=self.enable_tamper_scan,
                tamper_log_path=self.tamper_log_path,
                step=global_step,
                trajectory_id=trajectory_id,
            )

        ws_dir = docker_ws.export_workspace_to_local() if docker_ws is not None else local_ws.root
        return evaluate_repo_workspace(
            ws_dir,
            task,
            mode=self.reward_mode,
            enable_tamper_scan=self.enable_tamper_scan,
            tamper_log_path=self.tamper_log_path,
            scoring_backend=self.scoring_backend,
            sandbox_pool=self.sandbox_pool,
            timeout_s=self.test_timeout_s,
            step=global_step,
            trajectory_id=trajectory_id,
        )
