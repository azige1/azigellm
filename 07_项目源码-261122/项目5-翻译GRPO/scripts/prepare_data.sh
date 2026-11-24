#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -m translation_grpo.data_prep \
  --inputs data/sample_zh2en.jsonl data/sample_en2zh.jsonl \
  --output-dir data/processed \
  --val-ratio 0.1 \
  --seed 42 \
  "$@"
