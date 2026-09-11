#!/bin/bash
# 单机 PPO 训练示例（3B 级模型，直连 Docker 环境 rl/env.py）。
# 前置：bash scripts/install_skyrl.sh 已执行；rl/prepare_data.py 已产出 parquet。
set -e

cd "$(dirname "$0")/../.."
source /tmp/sky/bin/activate

# 定位 CUDA_HOME，供 flashinfer JIT 运行时找到 nvcc
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

# 清理过期的 flashinfer JIT 缓存
rm -rf ~/.cache/flashinfer

# 数据与输出目录（可用环境变量覆盖）
DATA_DIR="${DATA_DIR:-./data}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/ppo_single_node}"
EXPORT_DIR="${EXPORT_DIR:-./exports/ppo_single_node}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
mkdir -p "$CKPT_DIR" "$EXPORT_DIR"

# 自动判断续训
if [ -f "$CKPT_DIR/latest_ckpt_global_step.txt" ]; then
  RESUME_MODE=latest
  echo "Found existing checkpoint, resuming from latest."
else
  RESUME_MODE=null
  echo "No checkpoint found, starting fresh."
fi

LOG_FILE="$CKPT_DIR/train_debug.log"

RAY_memory_usage_threshold=0.99 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME=/tmp/hf_cache \
WANDB_MODE=offline \
SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1 \
VLLM_ATTENTION_BACKEND=TORCH_SDPA \
python -m rl.main \
  "data.train_data=['$DATA_DIR/train.parquet']" \
  "data.val_data=['$DATA_DIR/validation.parquet']" \
  environment.env_class=terminal_task \
  trainer.policy.model.path=$MODEL \
  trainer.critic.model.path=$MODEL \
  trainer.strategy=fsdp2 \
  trainer.algorithm.advantage_estimator=gae \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=4 \
  trainer.placement.critic_num_gpus_per_node=4 \
  trainer.placement.ref_num_gpus_per_node=4 \
  trainer.flash_attn=false \
  trainer.remove_microbatch_padding=false \
  trainer.policy.use_torch_compile=false \
  trainer.gradient_checkpointing=true \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.critic_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=4096 \
  trainer.epochs=2 \
  trainer.update_epochs_per_batch=2 \
  trainer.ckpt_interval=100 \
  trainer.eval_interval=20 \
  trainer.eval_batch_size=10 \
  trainer.max_ckpts_to_keep=1 \
  trainer.logger=console \
  "trainer.project_name=terminal-task-rl" \
  "trainer.run_name=ppo-single-node-3b" \
  "trainer.ckpt_path=$CKPT_DIR" \
  "trainer.export_path=$EXPORT_DIR" \
  trainer.resume_mode=$RESUME_MODE \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=4 \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.35 \
  generator.n_samples_per_prompt=4 \
  generator.max_turns=8 \
  "environment.skyrl_gym.max_env_workers=10" \
  "generator.sampling_params.max_generate_length=2048" \
  "generator.sampling_params.temperature=0.6" \
  2>&1 | tee "$LOG_FILE"

echo "Training complete. Checkpoints in $CKPT_DIR, exports in $EXPORT_DIR."
