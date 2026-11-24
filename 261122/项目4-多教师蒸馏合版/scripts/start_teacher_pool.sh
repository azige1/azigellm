#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/shared_env.sh"
MANIFEST_PATH="${TEACHER_MANIFEST_PATH:-outputs/teachers.json}"
python "${ROOT}/scripts/prepare_teacher_manifest.py" --output "$MANIFEST_PATH"
export TEACHER_MANIFEST
TEACHER_MANIFEST="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))' "$MANIFEST_PATH")"
export TEACHER_TOKEN_BUDGET="${TEACHER_TOKEN_BUDGET:-32768}"
export TEACHER_ROW_CHUNK_SIZE="${TEACHER_ROW_CHUNK_SIZE:-256}"
export CUDA_VISIBLE_DEVICES="${TEACHER_GPUS:-4,5,6,7}"
python -m uvicorn multi_teacher_distillation.teacher.pool_service:app --host 0.0.0.0 --port "${TEACHER_POOL_PORT:-8400}"
