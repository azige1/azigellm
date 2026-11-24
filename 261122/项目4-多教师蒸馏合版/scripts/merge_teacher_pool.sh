#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/shared_env.sh"
require_variables ABILITY_MATRIX_JSON TEACHER_MAP_OUTPUT
python -m multi_teacher_distillation.teacher_merging \
  --matrix "$ABILITY_MATRIX_JSON" \
  --output "$TEACHER_MAP_OUTPUT" \
  --noise-multiplier "${NOISE_MULTIPLIER:-2.0}" "$@"
