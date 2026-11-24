#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/shared_env.sh"
DOMAIN="${1:?用法: train_teacher.sh <code_gen|issue_fix|test_gen|math_science|instruction_following|tool_calling|general_chat> [Hydra 覆盖]}"
shift
case "$DOMAIN" in
  code_gen|issue_fix|test_gen|math_science|instruction_following|tool_calling|general_chat) ;;
  repo_impl) echo "repo_impl 直接读取项目三 RL checkpoint，无需重训" >&2; exit 2 ;;
  *) echo "未知领域: $DOMAIN" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="${TEACHER_TRAINING_GPUS:-0,1,2,3,4,5,6,7}"
require_variables TRAINING_DATA_FILE VALIDATION_DATA_FILE MODEL_WEIGHTS_PATH JUDGE_BASE_URL
python -m multi_teacher_distillation.main --config-name "${CONFIG:-teacher_rl_8xa800}" \
  trainer.total_training_steps="${TEACHER_STEPS:-150}" \
  trainer.experiment_name="teacher-${DOMAIN}" \
  trainer.default_local_dir="${CHECKPOINT_ROOT:-checkpoints}/teacher_${DOMAIN}" \
  evaluation_log="${CHECKPOINT_ROOT:-checkpoints}/teacher_${DOMAIN}/evaluations.jsonl" \
  actor_rollout_ref.rollout.agent.default_agent_loop="$DOMAIN" \
  "$@"
