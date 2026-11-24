#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/shared_env.sh"
require_variables MODEL_WEIGHTS_PATH FIXED_PROMPTS TERMINATION_REPORT
require_variables CODE_GEN_TEACHER_MODEL REPO_IMPL_TEACHER_MODEL ISSUE_FIX_TEACHER_MODEL TEST_GEN_TEACHER_MODEL
require_variables MATH_SCIENCE_TEACHER_MODEL INSTRUCTION_FOLLOWING_TEACHER_MODEL TOOL_CALLING_TEACHER_MODEL GENERAL_CHAT_TEACHER_MODEL
python -m multi_teacher_distillation.termination --student "$MODEL_WEIGHTS_PATH" \
  --teacher "code_gen=$CODE_GEN_TEACHER_MODEL" \
  --teacher "repo_impl=$REPO_IMPL_TEACHER_MODEL" \
  --teacher "issue_fix=$ISSUE_FIX_TEACHER_MODEL" \
  --teacher "test_gen=$TEST_GEN_TEACHER_MODEL" \
  --teacher "math_science=$MATH_SCIENCE_TEACHER_MODEL" \
  --teacher "instruction_following=$INSTRUCTION_FOLLOWING_TEACHER_MODEL" \
  --teacher "tool_calling=$TOOL_CALLING_TEACHER_MODEL" \
  --teacher "general_chat=$GENERAL_CHAT_TEACHER_MODEL" \
  --fixed-prompts "$FIXED_PROMPTS" --output "$TERMINATION_REPORT"
