#!/bin/bash
# 启动本地 vLLM 服务（任务生成 / 解题 rollout 共用）
# 用法：bash scripts/launch_vllm_server.sh <tensor_parallel> <data_parallel>
vllm serve Qwen/Qwen3-32B-AWQ --tensor-parallel-size $1 --data-parallel-size $2 --gpu-memory-utilization 0.95 --api-key nokey --enable-prefix-caching --disable-log-requests --enable-chunked-prefill
