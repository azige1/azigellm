#!/usr/bin/env bash
# 配比扫描：0.15 / 0.25 / 0.35 / 0.45 四档依次打包与训练，
# 外加初版数据（--baseline_raw）与 --repeat_success 两个对照，其余超参不变。
# 用法: run_mix_sweep.sh <exported/selected.jsonl> <augmented/injected> <augmented/compacted> <exported/index.jsonl>
set -euo pipefail

SELECTED=${1:?用法: run_mix_sweep.sh <selected.jsonl> <injected_dir> <compacted_dir> <index.jsonl>}
INJECTED=${2:?缺少 injected 目录}
COMPACTED=${3:?缺少 compacted 目录}
RAW_INDEX=${4:?缺少 exported/index.jsonl}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TOKENIZER=${TOKENIZER:-Qwen/Qwen3-8B}
mkdir -p data

for MIX in 0.15 0.25 0.35 0.45; do
  DATA="data/train_mix_${MIX}.jsonl"
  python "${SCRIPT_DIR}/build_dataset.py" \
    --selected "${SELECTED}" --injected "${INJECTED}" --compacted "${COMPACTED}" \
    --mix "${MIX}" --inject_to_compact 3:1 --tokenizer "${TOKENIZER}" --out "${DATA}"
  bash "${SCRIPT_DIR}/sft_qwen3_8b.sh" "${DATA}" "output/qwen3-8b-mix-${MIX}"
done

# 对照 1：初版数据（只按通过率阈值取成功轨迹，不经归因筛选、盲改过滤与增强）
python "${SCRIPT_DIR}/build_dataset.py" \
  --baseline_raw --raw_index "${RAW_INDEX}" --tokenizer "${TOKENIZER}" --out data/train_baseline.jsonl
bash "${SCRIPT_DIR}/sft_qwen3_8b.sh" data/train_baseline.jsonl output/qwen3-8b-baseline

# 对照 2：增强部分换成等量成功轨迹重复采样
python "${SCRIPT_DIR}/build_dataset.py" \
  --selected "${SELECTED}" --mix 0.25 --repeat_success --tokenizer "${TOKENIZER}" --out data/train_repeat.jsonl
bash "${SCRIPT_DIR}/sft_qwen3_8b.sh" data/train_repeat.jsonl output/qwen3-8b-repeat
