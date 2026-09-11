#!/bin/bash
# 双节点 GRPO 训练示例（Harbor 集成路线：terminus-2 agent + Docker 环境）。
# 走 SkyRL 自带的 harbor 训练集成，rollout 由 Harbor 编排。
#
# 用法：
#   head 节点：  ray start --head --port=6379
#   worker 节点：ray start --address=<head_private_ip>:6379
#   然后在 head 节点执行本脚本。
set -e

cd "$(dirname "$0")/../.."
source /tmp/sky/bin/activate

# 定位 CUDA_HOME
if [ -d "/usr/local/cuda" ] && [ -f "/usr/local/cuda/bin/nvcc" ]; then
  export CUDA_HOME=/usr/local/cuda
elif [ -f "/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13/bin/nvcc" ]; then
  export CUDA_HOME=/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13
else
  NVCC_PATH=$(which nvcc 2>/dev/null || true)
  if [ -n "$NVCC_PATH" ]; then
    export CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
  else
    echo "ERROR: Could not find nvcc. Set CUDA_HOME manually and re-run." >&2
    exit 1
  fi
fi
export PATH="$CUDA_HOME/bin:$PATH"
echo "Using CUDA_HOME=$CUDA_HOME"
rm -rf ~/.cache/flashinfer

# 训练前确认 Ray 集群两个节点都在线
echo "Checking Ray cluster..."
ray status
TOTAL_GPUS=$(python3 -c "import ray; ray.init(address='auto'); resources = ray.available_resources(); print(int(resources.get('GPU', 0)))")
EXPECTED_GPUS="${EXPECTED_GPUS:-16}"
if [ "$TOTAL_GPUS" -lt "$EXPECTED_GPUS" ]; then
  echo "ERROR: Expected $EXPECTED_GPUS GPUs (2 nodes), found $TOTAL_GPUS."
  echo "On worker node run: ray start --address=<head_private_ip>:6379"
  exit 1
fi
echo "Ray cluster OK: $TOTAL_GPUS GPUs."

# 任务目录列表写成 JSON 文件传给训练器，避免命令行参数过长
DATA_DIR="${DATA_DIR:-./data}"
TRAIN_DIRS_FILE="${TRAIN_DIRS_FILE:-/tmp/train_task_dirs.json}"
VAL_DIRS_FILE="${VAL_DIRS_FILE:-/tmp/val_task_dirs.json}"
python3.13 -c "
import pandas as pd, json
df = pd.read_parquet('$DATA_DIR/train.parquet')
dirs = list(df['extra_info'].apply(lambda x: x['task_dir']).unique())
json.dump(dirs, open('$TRAIN_DIRS_FILE','w'))
print('Train tasks:', len(dirs))
"
python3.13 -c "
import pandas as pd, json
df = pd.read_parquet('$DATA_DIR/validation.parquet')
dirs = list(df['extra_info'].apply(lambda x: x['task_dir']).unique())
json.dump(dirs, open('$VAL_DIRS_FILE','w'))
print('Val tasks:', len(dirs))
"

CKPT_DIR="${CKPT_DIR:-./checkpoints/grpo_2node}"
EXPORT_DIR="${EXPORT_DIR:-./exports/grpo_2node}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
mkdir -p "$CKPT_DIR" "$EXPORT_DIR"

if [ -f "$CKPT_DIR/latest_ckpt_global_step.txt" ]; then
  RESUME_MODE=latest
  echo "Resuming from latest checkpoint."
else
  RESUME_MODE=null
  echo "Starting fresh."
fi

LOG_FILE="$CKPT_DIR/train_debug.log"

# 多节点关键配置：
#   trainer.placement.colocate_all=false —— colocate 依赖 CUDA IPC，跨节点不可用
#   weight_sync_backend=broadcast        —— 用 NCCL 广播替代 CUDA IPC
#   NCCL_SOCKET_IFNAME                   —— 指定 NCCL 走 IPv4 网卡
cd SkyRL
RAY_memory_usage_threshold=0.99 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME=/tmp/hf_cache \
WANDB_MODE=offline \
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens5}" \
SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1 \
VLLM_ATTENTION_BACKEND=TORCH_SDPA \
VLLM_USE_FLASHINFER_SAMPLER=0 \
MSWEA_API_KEY=nokey \
python -m examples.train_integrations.harbor.entrypoints.main_harbor \
  "data.train_data=[\"$TRAIN_DIRS_FILE\"]" \
  "data.val_data=[\"$VAL_DIRS_FILE\"]" \
  trainer.policy.model.path=$MODEL \
  trainer.strategy=fsdp \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.policy_num_nodes=2 \
  trainer.placement.ref_num_gpus_per_node=8 \
  trainer.placement.ref_num_nodes=2 \
  trainer.flash_attn=false \
  trainer.use_sample_packing=false \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=4096 \
  trainer.epochs=2 \
  trainer.ckpt_interval=50 \
  trainer.eval_interval=20 \
  trainer.max_ckpts_to_keep=5 \
  trainer.logger=console \
  "trainer.project_name=terminal-task-rl" \
  "trainer.run_name=grpo-harbor-2node" \
  "trainer.ckpt_path=$CKPT_DIR" \
  "trainer.export_path=$EXPORT_DIR" \
  trainer.resume_mode=$RESUME_MODE \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.weight_sync_backend=broadcast \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.n_samples_per_prompt=8 \
  generator.max_turns=16 \
  "generator.sampling_params.max_generate_length=2048" \
  "generator.sampling_params.temperature=0.6" \
  2>&1 | tee "$LOG_FILE"

echo "Training complete. Checkpoints in $CKPT_DIR, exports in $EXPORT_DIR."
