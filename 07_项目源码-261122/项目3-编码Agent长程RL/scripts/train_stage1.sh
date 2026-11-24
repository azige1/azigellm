#!/usr/bin/env bash
set -euo pipefail

# 阶段一：项目一函数级任务 8xA800 全参 GRPO（120 步）
SFT_CKPT="${1:-checkpoints/project2-qwen3-8b-sft}"
FUNC_TASKS="${2:-../项目1-函数级任务合成/workspace/tasks/exp1_train.jsonl}"
FUNC_EVAL="${3:-../项目1-函数级任务合成/workspace/tasks/exp1_eval.jsonl}"

mkdir -p data logs

# 1. 开训前用 SFT checkpoint 做 pass@8 筛选（仅保留 1/8 至 6/8）
python scripts/filter_pass_at_8.py \
  --stage function \
  --input "${FUNC_TASKS}" \
  --output data/stage1_filtered_train.jsonl \
  --report logs/stage1_pass_at_8.json \
  --model "${SFT_CKPT}"

# 2. 构造 verl parquet（异构嵌套字段存 JSON 字符串）
python scripts/prepare_data.py \
  --stage function \
  --input data/stage1_filtered_train.jsonl \
  --out-dir data

python scripts/prepare_data.py \
  --stage function \
  --input "${FUNC_EVAL}" \
  --out-dir data \
  --train-split eval \
  --eval-split eval

# 3. 启动 8xA800 阶段一 RL 训练
python -m long_horizon_rl.main \
  --config-name stage1_function \
  actor_rollout_ref.model.path="${SFT_CKPT}"
