#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/shared_env.sh"
export CUDA_VISIBLE_DEVICES="${WARMUP_SFT_GPUS:-0,1,2,3}"
require_variables MODEL_WEIGHTS_PATH WARMUP_SFT_DATA
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" \
  -m verl.trainer.sft_trainer \
  --config-path "${ROOT}/experiments/configs" \
  --config-name warmup_sft_8xa800 \
  "$@"
