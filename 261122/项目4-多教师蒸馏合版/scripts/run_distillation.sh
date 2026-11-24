#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/shared_env.sh"
CONFIG="${CONFIG:-distill_8xa800}"
# 8xA800 档只向 Ray 暴露 GPU 0-3，GPU 4-7 保留给教师常驻服务
if [[ "$CONFIG" == *8xa800* ]]; then export CUDA_VISIBLE_DEVICES="${STUDENT_ROLLOUT_GPUS:-0,1,2,3}"; fi
require_variables MODEL_WEIGHTS_PATH TRAINING_DATA_FILE VALIDATION_DATA_FILE JUDGE_BASE_URL
require_variables CODE_GEN_TEACHER_MODEL REPO_IMPL_TEACHER_MODEL ISSUE_FIX_TEACHER_MODEL TEST_GEN_TEACHER_MODEL
require_variables MATH_SCIENCE_TEACHER_MODEL INSTRUCTION_FOLLOWING_TEACHER_MODEL TOOL_CALLING_TEACHER_MODEL GENERAL_CHAT_TEACHER_MODEL
if [[ "${SKIP_PREFLIGHT:-false}" != true ]]; then
  require_variables TERMINATION_REPORT
  python -c 'import json,sys; report=json.load(open(sys.argv[1], encoding="utf-8")); assert report.get("passed") is True, "终止 token 核对未通过"' "$TERMINATION_REPORT"
fi
python -m multi_teacher_distillation.main --config-name "$CONFIG" "$@"
