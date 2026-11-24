#!/usr/bin/env bash
# Qwen3-8B 全参 SFT：4×A800 80G，ms-swift 4.6.0。
# 32K 上下文不开 YaRN。
# SP=2 时数据并行组为 2，global batch = 1（per_device）× 2（DP）× 8（累积）= 16 条轨迹。
set -euo pipefail

DATASET=${1:?用法: sft_qwen3_8b.sh <train.jsonl> [output_dir]}
OUTPUT_DIR=${2:-output/qwen3-8b-sft}
# 第三个参数起原样传给 swift sft，例如小规模数据时用 --save_steps 5 --eval_steps 5 覆盖默认值

NPROC_PER_NODE=4 swift sft \
  --model Qwen/Qwen3-8B \
  --tuner_type full \
  --template qwen3 \
  --agent_template hermes \
  --dataset "${DATASET}" \
  --split_dataset_ratio 0.02 \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --deepspeed zero3 \
  --sequence_parallel_size 2 \
  --max_length 32768 \
  --truncation_strategy delete \
  --loss_scale default \
  --packing false \
  --padding_free true \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing true \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --num_train_epochs 2 \
  --save_strategy steps \
  --save_steps 50 \
  --eval_strategy steps \
  --eval_steps 50 \
  --logging_steps 5 \
  --output_dir "${OUTPUT_DIR}" \
  "${@:3}"
