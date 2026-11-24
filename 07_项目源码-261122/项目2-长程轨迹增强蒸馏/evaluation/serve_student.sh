#!/usr/bin/env bash
# 启动学生模型 vLLM 服务（Hermes 工具调用解析，32K 上下文）。
set -euo pipefail

CHECKPOINT=${1:?用法: serve_student.sh <checkpoint路径> [port]}
PORT=${2:-8000}

vllm serve "${CHECKPOINT}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 32768 \
  --port "${PORT}"
