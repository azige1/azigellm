#!/usr/bin/env bash
# 四组实验的启动命令。用法: EXP=exp1 ./scripts/run_ablation.sh <1|2|3|4>
#   1 = 基线: full 上下文 + literal 测试风格，单轮
#   2 = 默认: sliced + equivalence + 采集 + 3 轮反馈 + 规格提炼
#   3 = 核心消融: 在基线上逐项累加等价断言 / 切片 / 采集（均为单轮，比较一次执行通过率）
#   4 = 技术点逐项对照: 每个技术点关闭后的配置
# 前置: 已对 $EXP 执行 fetch / build --local / harvest / capture（见 run_pipeline.sh 前四步）。
# 各组实验共用 ${EXP}_extracted.json 与 $EXP 的采集记录，结果按组写到不同的实验 ID 下。
set -euo pipefail

EXP="${EXP:-exp1}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
GROUP="${1:-2}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

COMMON=(--local --model_name "$MODEL" --in-file "${EXP}_extracted.json" --capture_exp_id "$EXP")

case "$GROUP" in
  1)
    # 基线：整仓上下文 + 字面期望值，单轮，不附采集记录
    codeharvest agent-loop -e "${EXP}_baseline" "${COMMON[@]}" \
      --context_type full --test_style literal --max_rounds 1 --disable_captured_calls
    ;;
  2)
    # 默认配置：3 轮反馈 + 规格提炼 + 可解性检查 + 导出
    codeharvest agent-loop -e "${EXP}_default" "${COMMON[@]}"
    codeharvest refine-specs -e "${EXP}_default" --model_name "$MODEL"
    codeharvest check-solvability -e "${EXP}_default" --local
    codeharvest export-tasks -e "${EXP}_default" --eval_ratio 0.1 --blocklist data/benchmark_repo_blocklist.txt
    ;;
  3)
    # + 等价性断言与自等价校验（整仓上下文，无采集）
    codeharvest agent-loop -e "${EXP}_equiv" "${COMMON[@]}" \
      --context_type full --test_style equivalence --max_rounds 1 --disable_captured_calls
    # + 依赖切片 3 层（无采集）
    codeharvest agent-loop -e "${EXP}_sliced" "${COMMON[@]}" \
      --context_type sliced --test_style equivalence --max_rounds 1 --disable_captured_calls
    # + 运行时插桩采集
    codeharvest agent-loop -e "${EXP}_capture" "${COMMON[@]}" \
      --context_type sliced --test_style equivalence --max_rounds 1
    # 切片深度与上下文长度对照
    codeharvest slice-report -e "$EXP" --depths 1,2,3,5,full
    ;;
  4)
    # 关闭多轮反馈（单轮）
    codeharvest agent-loop -e "${EXP}_single_round" "${COMMON[@]}" --max_rounds 1
    # 关闭真实调用采集
    codeharvest agent-loop -e "${EXP}_no_capture" "${COMMON[@]}" --disable_captured_calls
    # 关闭切片（整仓上下文）
    codeharvest agent-loop -e "${EXP}_no_slice" "${COMMON[@]}" --context_type full
    ;;
  *)
    echo "未知实验组: $GROUP（可选 1/2/3/4）" >&2
    exit 1
    ;;
esac
