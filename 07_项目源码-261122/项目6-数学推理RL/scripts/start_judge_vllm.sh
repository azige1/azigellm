#!/usr/bin/env bash
set -euo pipefail

# Deploy Qwen3-32B (TP=2) on GPU 6-7 as OpenAI-compatible per-subquestion judge
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3-32B}
JUDGE_PORT=${JUDGE_PORT:-8001}
JUDGE_HOST=${JUDGE_HOST:-0.0.0.0}

python -m vllm.entrypoints.openai.api_server \
  --model "${JUDGE_MODEL}" \
  --tensor-parallel-size 2 \
  --host "${JUDGE_HOST}" \
  --port "${JUDGE_PORT}" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --trust-remote-code
