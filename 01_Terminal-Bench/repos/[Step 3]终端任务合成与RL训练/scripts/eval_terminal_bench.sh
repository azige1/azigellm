#!/usr/bin/env bash
# 在 tmux 窗口里跑 Terminal-Bench 评测（训练前后对比用）。
#
# 两种模式：
#   --mode base        直接评测基座模型（HF hub 模型名）
#   --mode checkpoint  先用本地 vLLM 拉起 HF 格式检查点，再评测
#
# 用法：
#   bash scripts/eval_terminal_bench.sh --mode base --model Qwen/Qwen3-8B --job-name tb-base
#   bash scripts/eval_terminal_bench.sh --mode checkpoint \
#       --checkpoint /path/to/exports/global_step_100 --model Qwen/Qwen3-8B
#
# 结果写入 <jobs-dir>/<job-name>/。
# 汇总 pass@k：python -m harbor_bridge.collect_results --jobs-dir <jobs-dir>

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO/.venv"
SESSION="tb-eval"
LOG_DIR="$REPO/harbor_logs"
JOBS_DIR="tb_jobs"
DATASET="terminal-bench/terminal-bench@latest"
N_CONCURRENT=10
AGENT="harbor_bridge.shell_agent:TerminalShellAgent"
VLLM_PORT=8100
VLLM_API_KEY="nokey"

MODE=""
MODEL=""
CHECKPOINT=""
JOB_NAME=""

usage() {
    cat <<EOF
Usage: $0 --mode <base|checkpoint> --model <model> [options]

Required:
  --mode base|checkpoint   Eval mode
  --model MODEL            HF model name

For checkpoint mode:
  --checkpoint DIR         Path to HF-format checkpoint dir (trainer.export_path/global_step_N)

Optional:
  --job-name NAME          Job name (default: tb-<mode>-<model-basename>)
  --n-concurrent N         Concurrent Harbor tasks (default: $N_CONCURRENT)
  --dataset DATASET        Harbor dataset spec (default: $DATASET)
  --vllm-port PORT         Local vLLM port for checkpoint mode (default: $VLLM_PORT)
  --jobs-dir DIR           Results directory (default: $JOBS_DIR)
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)         MODE="$2";         shift 2 ;;
        --model)        MODEL="$2";        shift 2 ;;
        --checkpoint)   CHECKPOINT="$2";   shift 2 ;;
        --job-name)     JOB_NAME="$2";     shift 2 ;;
        --n-concurrent) N_CONCURRENT="$2"; shift 2 ;;
        --dataset)      DATASET="$2";      shift 2 ;;
        --vllm-port)    VLLM_PORT="$2";    shift 2 ;;
        --jobs-dir)     JOBS_DIR="$2";     shift 2 ;;
        --help|-h)      usage ;;
        *) echo "Unknown arg: $1"; usage ;;
    esac
done

[[ -z "$MODE" ]]  && echo "Error: --mode is required"  && usage
[[ -z "$MODEL" ]] && echo "Error: --model is required" && usage

MODEL_BASENAME="$(basename "$MODEL")"
JOB_NAME="${JOB_NAME:-tb-${MODE}-${MODEL_BASENAME}}"
LOG_FILE="$LOG_DIR/tb_run_${JOB_NAME}.log"
mkdir -p "$LOG_DIR" "$REPO/$JOBS_DIR"

# ── checkpoint 模式校验 ─────────────────────────────────────────────
if [[ "$MODE" == "checkpoint" ]]; then
    [[ -z "$CHECKPOINT" ]] && echo "Error: --checkpoint required for checkpoint mode" && usage
    [[ ! -d "$CHECKPOINT" ]] && echo "Error: checkpoint dir not found: $CHECKPOINT" && exit 1
    MODEL_OR_ENDPOINT="http://localhost:${VLLM_PORT}/v1"
elif [[ "$MODE" == "base" ]]; then
    MODEL_OR_ENDPOINT="$MODEL"
else
    echo "Error: --mode must be 'base' or 'checkpoint'"
    usage
fi

# ── 组装 harbor run 命令 ────────────────────────────────────────────
# checkpoint 模式：agent 的 model 传 vLLM 端点（用于补全），
# 另用 tokenizer_model kwarg 传 HF 模型名（用于加载分词器）。
TOKENIZER_KWARG=""
if [[ "$MODE" == "checkpoint" ]]; then
    TOKENIZER_KWARG="--ak tokenizer_model=$MODEL"
fi

HARBOR_CMD="$VENV/bin/harbor run \
  -d $DATASET \
  --agent-import-path $AGENT \
  --model $MODEL_OR_ENDPOINT \
  --n-concurrent $N_CONCURRENT \
  --jobs-dir $JOBS_DIR \
  --job-name $JOB_NAME \
  $TOKENIZER_KWARG"

echo "=================================================="
echo "Terminal-Bench Evaluation"
echo "=================================================="
echo "Mode:         $MODE"
echo "Model:        $MODEL"
if [[ "$MODE" == "checkpoint" ]]; then
    echo "Checkpoint:   $CHECKPOINT"
    echo "vLLM port:    $VLLM_PORT"
fi
echo "Dataset:      $DATASET"
echo "Concurrent:   $N_CONCURRENT"
echo "Job name:     $JOB_NAME"
echo "Results dir:  $JOBS_DIR"
echo "Log:          $LOG_FILE"
echo "=================================================="

# ── 确保 tmux 会话存在 ──────────────────────────────────────────────
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "main" -c "$REPO"
fi

# ── 拼出 tmux 内执行的完整命令 ──────────────────────────────────────
if [[ "$MODE" == "checkpoint" ]]; then
    # 先启动 vLLM 服务，等待就绪后再跑 Harbor
    FULL_CMD=$(cat <<TMUXCMD
cd $REPO
echo "[tb-eval] Starting vLLM server from checkpoint: $CHECKPOINT"
CUDA_VISIBLE_DEVICES=1,2,3 $VENV/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$CHECKPOINT" \
  --served-model-name "$MODEL_BASENAME" \
  --port $VLLM_PORT \
  --api-key $VLLM_API_KEY \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 &
VLLM_PID=\$!
echo "[tb-eval] vLLM PID: \$VLLM_PID — waiting for server to be ready..."

WAIT_SECS=0
until curl -sf http://localhost:$VLLM_PORT/health >/dev/null 2>&1; do
    sleep 5; WAIT_SECS=\$((WAIT_SECS + 5))
    if ! kill -0 \$VLLM_PID 2>/dev/null; then
        echo "[tb-eval] ERROR: vLLM process died after \${WAIT_SECS}s — aborting eval."
        exit 1
    fi
    if [[ \$WAIT_SECS -ge 600 ]]; then
        echo "[tb-eval] ERROR: vLLM did not become ready after 600s — killing and aborting."
        kill \$VLLM_PID 2>/dev/null || true
        exit 1
    fi
done
echo "[tb-eval] vLLM server ready. Running Harbor eval..."
$HARBOR_CMD 2>&1 | tee $LOG_FILE
echo "[tb-eval] Harbor eval done. Stopping vLLM server (PID \$VLLM_PID)..."
kill \$VLLM_PID 2>/dev/null || true
TMUXCMD
)
else
    FULL_CMD="cd $REPO && $HARBOR_CMD 2>&1 | tee $LOG_FILE"
fi

# ── 在新 tmux 窗口中启动 ────────────────────────────────────────────
WINDOW_NAME="tb-${JOB_NAME//\./-}"
WINDOW_IDX=$(tmux new-window -t "$SESSION" -n "$WINDOW_NAME" -c "$REPO" -P -F "#{window_index}")
tmux send-keys -t "$SESSION:$WINDOW_IDX" "$FULL_CMD" Enter

echo ""
echo "Started in tmux window '$WINDOW_NAME' of session '$SESSION'."
echo ""
echo "  Attach:  tmux attach -t $SESSION"
echo "  Switch:  tmux select-window -t $SESSION:$WINDOW_NAME"
echo "  Logs:    tail -f $LOG_FILE"
echo ""
echo "After completion, collect results with:"
echo "  python -m harbor_bridge.collect_results --jobs-dir $JOBS_DIR"
