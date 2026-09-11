#!/bin/bash
# 用 TerminalShellAgent 在 Harbor 基准上并行评测本地模型

set -e

# 默认值（均可被同名环境变量覆盖）
TASKS_DIR="${TASKS_DIR:-./tasks}"
RESULTS_FILE="${RESULTS_FILE:-results.csv}"
AGENT="${AGENT:-harbor_bridge.shell_agent:TerminalShellAgent}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PARALLEL="${PARALLEL:-8}"
LOG_DIR="${LOG_DIR:-./harbor_logs}"
DATASET="${DATASET:-terminal-bench@2.0}"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run a local model with TerminalShellAgent on a Harbor dataset, in parallel"
    echo ""
    echo "Options:"
    echo "  --tasks-dir DIR        Directory containing tasks (default: $TASKS_DIR)"
    echo "  --results-file FILE    Output CSV file (default: $RESULTS_FILE)"
    echo "  --agent AGENT          Agent import path (default: $AGENT)"
    echo "  --model MODEL          Model name/path (default: $MODEL)"
    echo "  --parallel N           Number of parallel workers (default: $PARALLEL)"
    echo "  --log-dir DIR          Directory for task logs (default: $LOG_DIR)"
    echo "  --dataset DATASET      Harbor dataset (default: $DATASET)"
    echo "  --help                 Show this help message"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks-dir) TASKS_DIR="$2"; shift 2 ;;
        --results-file) RESULTS_FILE="$2"; shift 2 ;;
        --agent) AGENT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

echo "=========================================="
echo "Harbor Parallel Benchmark Runner"
echo "=========================================="
echo "Tasks Dir:    $TASKS_DIR"
echo "Results File: $RESULTS_FILE"
echo "Agent:        $AGENT"
echo "Model:        $MODEL"
echo "Parallel:     $PARALLEL"
echo "Log Dir:      $LOG_DIR"
echo "Dataset:      $DATASET"
echo "=========================================="

mkdir -p "$LOG_DIR"
echo "task_name,reward,errors,exit_code" > "$RESULTS_FILE"
LOCKFILE=$(mktemp)

run_task() {
    task_name=$1

    echo "Starting: $task_name"

    output=$(harbor jobs start \
        -d "$DATASET" \
        -t "$task_name" \
        --agent-import-path "$AGENT" \
        --env "daytona" \
        -m "$MODEL" 2>&1)

    exit_code=$?

    echo "$output" > "$LOG_DIR/${task_name}.log"

    reward=$(echo "$output" | grep -oP 'Mean[:\s│]+\K[0-9.]+' | head -1)
    [[ -z "$reward" ]] && reward="NA"

    errors=$(echo "$output" | grep -oP '│ Errors\s+│\s+\K[0-9]+' || echo "NA")

    # 加锁追加 CSV，避免并发写交错
    (
        flock 200
        echo "$task_name,$reward,$errors,$exit_code" >> "$RESULTS_FILE"
    ) 200>"$LOCKFILE"

    if [[ "$reward" == "NA" || $exit_code -ne 0 ]]; then
        echo "FAILED: $task_name (exit: $exit_code) - check $LOG_DIR/${task_name}.log"
        tail -10 "$LOG_DIR/${task_name}.log"
    else
        echo "Finished: $task_name -> reward: $reward"
    fi
}

tasks=()
for task_path in "$TASKS_DIR"/*/; do
    task_name=$(basename "$task_path")
    [[ "$task_name" =~ ^(\.|__) ]] && continue
    tasks+=("$task_name")
done

echo "Running ${#tasks[@]} tasks with $PARALLEL parallel jobs..."
echo "Logs will be saved to $LOG_DIR/"

running=0
for task_name in "${tasks[@]}"; do
    run_task "$task_name" &
    ((running++))
    sleep 5

    if ((running >= PARALLEL)); then
        wait -n
        ((running--))
    fi
done

wait
rm -f "$LOCKFILE"

echo ""
echo "Done. Results in $RESULTS_FILE"
echo "Check $LOG_DIR/ for full output of each task"
