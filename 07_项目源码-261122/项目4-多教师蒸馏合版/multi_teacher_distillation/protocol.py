from typing import Any


def build_loop_output(prompt_ids: list[int], response_ids: list[int], response_mask: list[int],
                      response_logprobs: list[float] | None, extra_fields: dict[str, Any] | None = None,
                      reward_score: float | None = None, num_turns: int = 2):
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

    if not len(response_ids) == len(response_mask) == len(response_logprobs or response_ids):
        raise ValueError("响应 token、mask 与 log-prob 必须等长")
    return AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        routed_experts=None,
        multi_modal_data={},
        mm_processor_kwargs={},
        mm_processor_output=None,
        reward_score=reward_score,
        num_turns=num_turns,
        metrics=AgentLoopMetrics(),
        extra_fields=extra_fields or {},
    )
