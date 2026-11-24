#!/usr/bin/env bash
set -euo pipefail

# Stage 2 (16K context) DAPO training resuming from Stage 1 checkpoint
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
export TRANSFER_QUEUE_ENABLE=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

python -m math_reasoning_rl.train_entry \
  --config "${PROJECT_DIR}/configs/dapo_8xa800_stage2_16k.yaml" \
  "$@"
