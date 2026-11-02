#!/usr/bin/env bash
# Qwen3-32B + YaRN 128K 参考配置：需要 16×H100 量级资源，4×A800 不要运行。
# 仅作长上下文蒸馏的参考：YaRN 因子 4，原始位置 32K 外推到 128K。
set -euo pipefail

DATASET=${1:?用法: sft_qwen3_32b_yarn128k_reference.sh <train.jsonl> [output_dir]}
OUTPUT_DIR=${2:-output/qwen3-32b-yarn128k-sft}

NPROC_PER_NODE=16 swift sft \
  --model Qwen/Qwen3-32B \
  --tuner_type full \
  --template qwen3 \
  --agent_template hermes \
  --dataset "${DATASET}" \
  --split_dataset_ratio 0.02 \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --deepspeed zero3 \
  --max_model_len 131072 \
  --rope_scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --max_length 131072 \
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
  --output_dir "${OUTPUT_DIR}"
