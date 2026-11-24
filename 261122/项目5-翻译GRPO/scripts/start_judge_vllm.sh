#!/usr/bin/env bash
set -euo pipefail

# Launch vLLM OpenAI-compatible online judge service on GPU6-7 (TP=2).
export CUDA_VISIBLE_DEVICES="${JUDGE_GPUS:-6,7}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-Qwen/Qwen3-32B}"
JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-Qwen3-32B}"
JUDGE_PORT="${JUDGE_PORT:-8000}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"

python -m vllm.entrypoints.openai.api_server \
  --model "${JUDGE_MODEL_PATH}" \
  --served-model-name "${JUDGE_SERVED_NAME}" \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port "${JUDGE_PORT}" \
  --api-key "${JUDGE_API_KEY}"
