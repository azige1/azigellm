#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${JUDGE_GPUS:-0,1,2,3}"
python -m vllm.entrypoints.openai.api_server \
  --model "${JUDGE_MODEL_PATH:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}" \
  --served-model-name "${JUDGE_SERVED_NAME:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}" \
  --tensor-parallel-size "${JUDGE_TP:-4}" --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8001
