#!/usr/bin/env bash
set -euo pipefail

# Launch Translation GRPO training on GPU0-5 (6 GPUs), connecting to judge on GPU6-7.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1,2,3,4,5}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:8000/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-Qwen3-32B}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
export POLICY_SFT_MODEL_PATH="${POLICY_SFT_MODEL_PATH:-Qwen/Qwen3-8B}"

python -m translation_grpo.train_main "$@"
