#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

python -m math_reasoning_rl.data_prep \
  --input "${PROJECT_DIR}/data/sample_math_problems.jsonl" \
  --output-parquet "${PROJECT_DIR}/data/processed/train.parquet" \
  --output-jsonl "${PROJECT_DIR}/data/processed/train_clean.jsonl" \
  "$@"
