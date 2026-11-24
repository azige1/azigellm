#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -m translation_grpo.calibration \
  --gold-file "${1:-data/gold_calibration_sample.jsonl}" \
  --base-url "${JUDGE_BASE_URL:-http://127.0.0.1:8000/v1}" \
  --model "${JUDGE_MODEL:-Qwen3-32B}" \
  --cache-path "outputs/judge_eval_cache/calibration_cache.jsonl"
