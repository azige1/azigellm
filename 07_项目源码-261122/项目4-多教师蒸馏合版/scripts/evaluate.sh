#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/shared_env.sh"
DOMAIN="${1:-all}"
require_variables MODEL_WEIGHTS_PATH EVAL_ROOT
mkdir -p "$EVAL_ROOT"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
python -m vllm.entrypoints.openai.api_server --model "$MODEL_WEIGHTS_PATH" --served-model-name student \
  --host 127.0.0.1 --port 8000 --max-model-len "${EVAL_MAX_MODEL_LEN:-49152}" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1 && break
  sleep 5
done
for d in code_gen test_gen math_science instruction_following tool_calling general_chat; do
  if [[ "$DOMAIN" == "$d" || "$DOMAIN" == all ]]; then
    VAR_NAME="$(echo "$d" | tr '[:lower:]' '[:upper:]')_EVAL_DATA"
    if [[ -n "${!VAR_NAME:-}" ]]; then
      python -m multi_teacher_distillation.evaluate text --domain "$d" --input "${!VAR_NAME}" \
        --output "$EVAL_ROOT/${d}.json" --limit "${EVAL_LIMIT:-500}" --model student \
        ${JUDGE_BASE_URL:+--judge-base-url "${EVAL_JUDGE_BASE_URL:-$JUDGE_BASE_URL}"} \
        --judge-model "${EVAL_JUDGE_MODEL:-GLM-5}"
    fi
  fi
done
if [[ "$DOMAIN" == all && -n "${SUMMARY_INPUTS:-}" ]]; then
  read -r -a INPUTS <<< "$SUMMARY_INPUTS"
  python -m multi_teacher_distillation.evaluate summarize --inputs "${INPUTS[@]}" --output "$EVAL_ROOT/summary.json" \
    ${MIXED_RL_NAME:+--mixed-rl-name "$MIXED_RL_NAME"}
fi
