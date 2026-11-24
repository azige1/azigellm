"""Translation GRPO package implementing 5.1-5.5 components on verl 0.9.1 V1."""

from .advantage_mask import (
    AnchorAdvantageConfig,
    compute_translation_grpo_advantage,
    verify_zero_grad_on_masked_batch,
)
from .judge_client import AsyncOpenAIJudgeClient, JudgeClientConfig
from .mqm_protocol import (
    MQM_DIMENSIONS,
    SEVERITY_WEIGHTS,
    HybridPairwiseResult,
    MQMErrorSpan,
    MQMPointwiseResult,
    PairwiseResult,
    compare_pairwise_swapped,
    compute_mqm_penalty,
    hybrid_compare_pair,
    score_group_with_anchor_hybrid,
    score_pointwise_mqm,
)
from .reward_scoring import (
    RewardWeightsConfig,
    compute_ngram_repetition_rate,
    compute_score,
    compute_seven_dim_reward,
    count_text_units,
    detect_invalid_output,
)

__all__ = [
    "AnchorAdvantageConfig",
    "AsyncOpenAIJudgeClient",
    "HybridPairwiseResult",
    "JudgeClientConfig",
    "MQMErrorSpan",
    "MQMPointwiseResult",
    "MQM_DIMENSIONS",
    "PairwiseResult",
    "RewardWeightsConfig",
    "SEVERITY_WEIGHTS",
    "compare_pairwise_swapped",
    "compute_mqm_penalty",
    "compute_ngram_repetition_rate",
    "compute_score",
    "compute_seven_dim_reward",
    "compute_translation_grpo_advantage",
    "count_text_units",
    "detect_invalid_output",
    "hybrid_compare_pair",
    "score_group_with_anchor_hybrid",
    "score_pointwise_mqm",
    "verify_zero_grad_on_masked_batch",
]
