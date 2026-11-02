#!/usr/bin/env bash
# 一键流水线：fetch → build --local → harvest → capture → agent-loop → refine-specs → check-solvability → export-tasks
# 用法: EXP=exp1 ./scripts/run_pipeline.sh （先执行 scripts/serve_models.sh 起好两个模型服务）
set -euo pipefail

EXP="${EXP:-exp1}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
# 合成模型服务（8000）；可解性检查默认连 8001 上的 Qwen3-8B，见 SolvabilityConfig.base_url
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

codeharvest fetch --repo_urls_file data/seed_repos.json --run_pycg_for_repos
codeharvest build --local
codeharvest harvest -e "$EXP" --overwrite_extracted
codeharvest capture -e "$EXP"
codeharvest agent-loop -e "$EXP" --local --model_name "$MODEL"
codeharvest refine-specs -e "$EXP" --model_name "$MODEL"
codeharvest check-solvability -e "$EXP" --local
codeharvest export-tasks -e "$EXP" --eval_ratio 0.1 --blocklist data/benchmark_repo_blocklist.txt
