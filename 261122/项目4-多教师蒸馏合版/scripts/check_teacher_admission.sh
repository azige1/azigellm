#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/shared_env.sh"
DOMAIN="${1:?请提供教师领域}"
shift
require_variables START_SCORE TEACHER_SCORE START_GENERAL_SCORE TEACHER_GENERAL_SCORE STUDENT_PATH TEACHER_MODEL_DIR ADMISSION_OUTPUT
python -m multi_teacher_distillation.admission --domain "$DOMAIN" \
  --start-score "$START_SCORE" --teacher-score "$TEACHER_SCORE" \
  --start-general-score "$START_GENERAL_SCORE" --teacher-general-score "$TEACHER_GENERAL_SCORE" \
  --student-path "$STUDENT_PATH" --teacher-model-dir "$TEACHER_MODEL_DIR" --output "$ADMISSION_OUTPUT" "$@"
