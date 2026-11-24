CANDIDATE_DOMAINS = (
    "func_gen",
    "competitive_coding",
    "repo_impl",
    "issue_fix",
    "test_gen",
    "math",
    "science",
    "instruction_following",
    "tool_calling",
    "general_chat",
)

DEFAULT_REWARD_SOURCES = {
    "func_gen": "execution_test",
    "competitive_coding": "execution_test",
    "repo_impl": "execution_test",
    "issue_fix": "execution_test",
    "test_gen": "execution_test",
    "math": "rule_verifier",
    "science": "rule_verifier",
    "instruction_following": "rule_verifier",
    "tool_calling": "rule_verifier",
    "general_chat": "llm_judge",
}

DISTILL_TEACHER_DOMAINS = (
    "code_gen",
    "repo_impl",
    "issue_fix",
    "test_gen",
    "math_science",
    "instruction_following",
    "tool_calling",
    "general_chat",
)

SELF_ANCHOR_DOMAIN = "self_anchor"
ROUTED_DOMAINS = (*DISTILL_TEACHER_DOMAINS, SELF_ANCHOR_DOMAIN)

DEFAULT_DOMAIN_TO_TEACHER = {
    "func_gen": "code_gen",
    "competitive_coding": "code_gen",
    "code_gen": "code_gen",
    "repo_impl": "repo_impl",
    "issue_fix": "issue_fix",
    "test_gen": "test_gen",
    "math": "math_science",
    "science": "math_science",
    "math_science": "math_science",
    "instruction_following": "instruction_following",
    "tool_calling": "tool_calling",
    "general_chat": "general_chat",
    SELF_ANCHOR_DOMAIN: SELF_ANCHOR_DOMAIN,
}

AGENT_TRAJECTORY_DOMAINS = frozenset({"repo_impl", "issue_fix"})

DEFAULT_DOMAIN_SHARES = {
    "code_gen": 0.14,
    "repo_impl": 0.18,
    "issue_fix": 0.14,
    "test_gen": 0.10,
    "math_science": 0.12,
    "instruction_following": 0.08,
    "tool_calling": 0.08,
    "general_chat": 0.08,
    "self_anchor": 0.08,
}

DEFAULT_POOL_SIZES = {
    "code_gen": 13440,
    "repo_impl": 17280,
    "issue_fix": 13440,
    "test_gen": 9600,
    "math_science": 11520,
    "instruction_following": 7680,
    "tool_calling": 7680,
    "general_chat": 7680,
    "self_anchor": 7680,
}

MODEL_8B = "Qwen/Qwen3-8B"
MODEL_32B = "Qwen/Qwen3-32B"
TRAIN_JUDGE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
EVAL_JUDGE_MODEL = "GLM-5"
