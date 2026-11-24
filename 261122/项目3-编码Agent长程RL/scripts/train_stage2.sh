#!/usr/bin/env bash
set -euo pipefail

# 阶段二：项目二仓库级任务 8xA800 全参 GRPO（200 步）
STAGE1_CKPT="${1:-checkpoints/stage1_function/global_step_120/actor/huggingface}"
REPO_TASKS="${2:-../项目2-长程轨迹增强蒸馏/tasks/test.jsonl}"

mkdir -p data logs

# 1. 开训前用阶段一末 checkpoint 重测仓库级题池 pass@8（仅保留 1/8 至 6/8）
python scripts/filter_pass_at_8.py \
  --stage repo \
  --input "${REPO_TASKS}" \
  --output data/stage2_filtered.jsonl \
  --report logs/stage2_pass_at_8.json \
  --model "${STAGE1_CKPT}"

# 2. 构造 verl parquet
python scripts/prepare_data.py \
  --stage repo \
  --input data/stage2_filtered.jsonl \
  --out-dir data

# 3. 启动 8xA800 阶段二长程 RL 训练（完成后 HF checkpoint 位于 checkpoints/stage2_repo/global_step_200/actor/huggingface）
python -m long_horizon_rl.main \
  --config-name stage2_repo \
  actor_rollout_ref.model.path="${STAGE1_CKPT}"
