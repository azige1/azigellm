#!/usr/bin/env bash
# 启动两个 vLLM OpenAI 兼容服务：30B 合成模型（2 卡 TP，端口 8000）与 8B 可解性检查模型（1 卡，端口 8001）。
# 合成请求上下文上限 32K token，加上最多 4K 输出，合成服务的 max-model-len 取 64K。
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --tensor-parallel-size 2 \
  --port 8000 \
  --max-model-len 65536 \
  --served-model-name Qwen/Qwen3-Coder-30B-A3B-Instruct &

CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-8B \
  --port 8001 \
  --max-model-len 32768 \
  --served-model-name Qwen/Qwen3-8B &

wait
