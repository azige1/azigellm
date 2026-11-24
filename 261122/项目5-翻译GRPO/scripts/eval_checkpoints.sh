#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

MODE="${1:-select}"
INPUT_FILE="${2:-outputs/eval_manifest.jsonl}"

python -m translation_grpo.checkpoint_eval \
  --mode "${MODE}" \
  --input-file "${INPUT_FILE}" \
  --strong-base-url "${STRONG_JUDGE_BASE_URL:-https://api.openai.com/v1}" \
  --strong-model "${STRONG_JUDGE_MODEL:-gpt-5}" \
  --strong-api-key-env "STRONG_JUDGE_API_KEY" \
  --online-base-url "${JUDGE_BASE_URL:-http://127.0.0.1:8000/v1}" \
  --online-model "${JUDGE_MODEL:-Qwen3-32B}"
