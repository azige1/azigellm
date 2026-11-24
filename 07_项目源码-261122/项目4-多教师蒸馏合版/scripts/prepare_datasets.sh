#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/shared_env.sh"
require_variables INPUT_MANIFEST OUTPUT_DATA POOL_STATS
python -m multi_teacher_distillation.data.build \
  --input-manifest "$INPUT_MANIFEST" \
  --output "$OUTPUT_DATA" --stats "$POOL_STATS" \
  ${DOMAIN_MAP_PATH:+--domain-map "$DOMAIN_MAP_PATH"} \
  ${REQUIRE_DEFAULT_POOL_SIZE:+--require-default-size} "$@"
