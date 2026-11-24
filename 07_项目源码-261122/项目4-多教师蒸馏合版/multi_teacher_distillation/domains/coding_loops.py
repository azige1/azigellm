from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

from ..rollout_masking import is_truncated
from ..routing import propagate_segment_metadata
from ..termination import terminal_token_ids
from .common import single_turn_record, task_from_kwargs
from .project3_adapter import IssueFixContainerConfig, attach_segment_routing_metadata


class VerifiedDomainLoop(SingleTurnAgentLoop):
    """单轮可验证/judge 领域的蒸馏与 RL rollout 基类，显式保留路由元数据。"""

    domain = ""

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs: Any):
        task = task_from_kwargs(kwargs)
        limit = int(kwargs.get("max_tokens", self.response_length))
        params = dict(sampling_params)
        params.update({"max_tokens": limit, "logprobs": 0})
        output = await super().run(params, priority=priority, **kwargs)
        if not output:
            return []

        text = self.tokenizer.decode(output.response_ids, skip_special_tokens=True)
        truncated = is_truncated(output.response_ids, limit, terminal_token_ids(self.tokenizer))
        from .verifiers import verifier_for

        result = await verifier_for(self.domain, self.config)(task, text)
        is_distill = bool(self.config.get("distillation", {}).get("enabled", False))
        output.reward_score = 0.0 if (truncated or is_distill or self.domain == "self_anchor") else float(result.score)
        if truncated:
            output.response_mask = [0] * len(output.response_ids)

        extra_info = kwargs.get("extra_info", {}) or {}
        parent_meta = {
            "domain": str(kwargs.get("domain", extra_info.get("domain", self.domain))),
            "teacher": str(kwargs.get("teacher", extra_info.get("teacher", self.domain))),
            "teacher_version": str(kwargs.get("teacher_version", extra_info.get("teacher_version", ""))),
        }
        updated_extra = propagate_segment_metadata(dict(output.extra_fields), parent_meta, segment_index=0)
        updated_extra.update({
            "task_type": self.domain,
            "prompt_id": str(extra_info.get("prompt_id", task.get("task_id", ""))),
            "truncated": truncated,
            "turn_records": [single_turn_record(text, len(output.response_ids))],
            "stop_reason": "length" if truncated else "answered",
            "reward_extra_info": {"success": result.passed, "score": result.score, **result.details},
        })
        output.extra_fields = updated_extra
        return output


class CodeGenLoop(VerifiedDomainLoop):
    domain = "code_gen"


class TestGenLoop(VerifiedDomainLoop):
    domain = "test_gen"


class MathScienceLoop(VerifiedDomainLoop):
    domain = "math_science"


class InstructionFollowingLoop(VerifiedDomainLoop):
    domain = "instruction_following"


class ToolCallingLoop(VerifiedDomainLoop):
    domain = "tool_calling"


class GeneralChatLoop(VerifiedDomainLoop):
    domain = "general_chat"


class SelfAnchorLoop(VerifiedDomainLoop):
    domain = "self_anchor"


class IssueFixLoop(AgentLoopBase):
    """issue 修复多轮轨迹 loop：使用 continuous-token 接口拼接序列与压缩分段，并携带容器配置。"""

    domain = "issue_fix"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.response_length = int(self.rollout_config.response_length)
        container_cfg = self.config.get("issue_fix_container", {}) if hasattr(self.config, "get") else {}
        self.container_config = IssueFixContainerConfig.from_mapping(
            {**dict(container_cfg or {}), **dict(kwargs.get("container_config", {}) or {})}
        )

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs: Any):
        del priority
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {}) or {}
        parent_meta = {
            "domain": str(kwargs.get("domain", extra_info.get("domain", self.domain))),
            "teacher": str(kwargs.get("teacher", extra_info.get("teacher", self.domain))),
            "teacher_version": str(kwargs.get("teacher_version", extra_info.get("teacher_version", ""))),
            "prompt_id": str(extra_info.get("prompt_id", "")),
        }
        prompt_ids = await self.ct_build_initial_tokens(messages, tools=kwargs.get("tools"))
        runtime_ids = list(prompt_ids)
        response_mask: list[int] = []
        response_logprobs: list[float] | None = [] if sampling_params.get("logprobs") else None

        segments: list[AgentLoopOutput] = []
        current_messages = list(messages)
        turns = 0
        max_turns = min(int(self.container_config.max_turns), int(self.rollout_config.multi_turn.max_assistant_turns))
        executor = kwargs.get("tool_executor")

        while turns < max_turns and len(segments) < self.container_config.max_segments:
            turns += 1
            token_out = await self.server_manager.generate(
                request_id=f"{parent_meta['prompt_id'] or 'issue'}-t{turns}",
                prompt_ids=runtime_ids,
                sampling_params=sampling_params,
            )
            merge_res, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                runtime_ids,
                token_out.token_ids,
                response_mask,
                response_logprobs,
                assistant_logprobs=token_out.log_probs,
            )
            runtime_ids = merge_res.token_ids
            reply_text = self.tokenizer.decode(token_out.token_ids, skip_special_tokens=True)

            if executor is None:
                break
            tool_observation, finished = await executor(reply_text, kwargs)
            if finished:
                break

            updated_messages = [
                *current_messages,
                {"role": "assistant", "content": reply_text},
                {"role": "tool", "content": str(tool_observation)},
            ]
            ctx_res, response_mask, response_logprobs = await self.ct_merge_context_msg(
                current_messages,
                updated_messages,
                runtime_ids,
                response_mask,
                response_logprobs,
                tools=kwargs.get("tools"),
            )
            runtime_ids = ctx_res.token_ids
            current_messages = updated_messages

            if len(runtime_ids) >= self.container_config.compaction_threshold_tokens and len(segments) + 1 < self.container_config.max_segments:
                seg_resp = runtime_ids[-len(response_mask) :] if response_mask else []
                seg_prompt = runtime_ids[: len(runtime_ids) - len(seg_resp)]
                segments.append(
                    AgentLoopOutput(
                        prompt_ids=seg_prompt,
                        response_ids=seg_resp[: self.response_length],
                        response_mask=response_mask[: self.response_length],
                        response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
                        num_turns=len(current_messages),
                        metrics={},
                    )
                )
                tail = current_messages[-min(len(current_messages), self.container_config.tail_turns * 2) :]
                current_messages = [*messages[:1], *tail] if messages and messages[0].get("role") == "system" else tail
                runtime_ids = await self.ct_build_initial_tokens(current_messages, tools=kwargs.get("tools"))
                response_mask = []
                response_logprobs = [] if sampling_params.get("logprobs") else None

        resp_ids = runtime_ids[-len(response_mask) :] if response_mask else []
        prompt_slice = runtime_ids[: len(runtime_ids) - len(resp_ids)]
        segments.append(
            AgentLoopOutput(
                prompt_ids=prompt_slice,
                response_ids=resp_ids[: self.response_length],
                response_mask=response_mask[: self.response_length],
                response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
                num_turns=len(current_messages),
                metrics={},
            )
        )

        is_distill = bool(self.config.get("distillation", {}).get("enabled", False))
        annotated = attach_segment_routing_metadata(
            segments,
            parent_meta=parent_meta,
            response_length=self.response_length,
            terminal_ids=terminal_token_ids(self.tokenizer),
            is_distillation=is_distill,
        )
        return annotated if len(annotated) > 1 else annotated[0]
