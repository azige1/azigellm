#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DELIVERY_ROOT="$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

require_hf_checkpoint() {
    local path="$1"
    [[ "$path" == */global_step_*/huggingface ]] || {
        echo "模型路径必须指向 global_step_N/huggingface: $path" >&2
        return 2
    }
    [[ -d "$path" ]] || { echo "模型目录不存在: $path" >&2; return 2; }
}

require_variables() {
    local name
    for name in "$@"; do
        [[ -n "${!name:-}" ]] || { echo "缺少环境变量: $name" >&2; return 2; }
    done
}
