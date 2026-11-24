#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/shared_env.sh"
require_variables CALIBRATION_ROLLOUTS CALIBRATION_TABLES_OUTPUT
python -m multi_teacher_distillation.reliability_weighting \
  --input "$CALIBRATION_ROLLOUTS" \
  --output "$CALIBRATION_TABLES_OUTPUT" \
  ${ADMISSION_LBS_JSON:+--admission-lbs "$ADMISSION_LBS_JSON"} "$@"
