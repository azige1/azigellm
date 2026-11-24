#!/usr/bin/env bash
set -euo pipefail

# 消融与参数扫描脚本：基础 GRPO、逐项累加、omega 扫描（0.25/0.5/1.0）与规范化粒度消融
SFT_CKPT="${1:-checkpoints/project2-qwen3-8b-sft}"
STAGE1_CKPT="${2:-checkpoints/stage1_function/global_step_120/actor/huggingface}"

# 行 2：基础 GRPO（直接仓库级、0/1 reward、关篡改扫描、关分段共享、关 step 级 advantage、关容器池）
python -m long_horizon_rl.main --config-name stage2_repo \
  actor_rollout_ref.model.path="${SFT_CKPT}" \
  trainer.experiment_name="ablation_base_grpo_binary" \
  coding_rl.reward.mode=binary \
  coding_rl.reward.enable_tamper_scan=false \
  coding_rl.compaction.enable=false \
  coding_rl.step_advantage.enable=false \
  coding_rl.sandbox.enable=false

# 行 3：+ 两阶段 curriculum（从阶段一 checkpoint 起步）
python -m long_horizon_rl.main --config-name stage2_repo \
  actor_rollout_ref.model.path="${STAGE1_CKPT}" \
  trainer.experiment_name="ablation_plus_curriculum" \
  coding_rl.reward.mode=binary \
  coding_rl.reward.enable_tamper_scan=false \
  coding_rl.compaction.enable=false \
  coding_rl.step_advantage.enable=false \
  coding_rl.sandbox.enable=false

# 行 4：+ 隐藏测试通过比例 reward 与篡改检测
python -m long_horizon_rl.main --config-name stage2_repo \
  actor_rollout_ref.model.path="${STAGE1_CKPT}" \
  trainer.experiment_name="ablation_plus_pass_ratio_tamper" \
  coding_rl.reward.mode=pass_ratio \
  coding_rl.reward.enable_tamper_scan=true \
  coding_rl.compaction.enable=false \
  coding_rl.step_advantage.enable=false \
  coding_rl.sandbox.enable=false

# 行 5：+ compaction 分段共享与摘要 token 训练
python -m long_horizon_rl.main --config-name stage2_repo \
  actor_rollout_ref.model.path="${STAGE1_CKPT}" \
  trainer.experiment_name="ablation_plus_compaction" \
  coding_rl.compaction.enable=true \
  coding_rl.step_advantage.enable=false \
  coding_rl.sandbox.enable=false

# 行 6：+ step 级 advantage（默认 omega=0.5）
python -m long_horizon_rl.main --config-name stage2_repo \
  actor_rollout_ref.model.path="${STAGE1_CKPT}" \
  trainer.experiment_name="ablation_plus_step_adv" \
  coding_rl.step_advantage.omega=0.5 \
  coding_rl.sandbox.enable=false

# 行 7：+ 沙箱容器池预热与快照复位（即默认完整配置）
python -m long_horizon_rl.main --config-name stage2_repo \
  actor_rollout_ref.model.path="${STAGE1_CKPT}" \
  trainer.experiment_name="ablation_full_default"

# omega 扫描（0.25 / 1.0）
for OMEGA in 0.25 1.0; do
  python -m long_horizon_rl.main --config-name stage2_repo \
    actor_rollout_ref.model.path="${STAGE1_CKPT}" \
    trainer.experiment_name="ablation_omega_${OMEGA}" \
    coding_rl.step_advantage.omega="${OMEGA}"
done

# 规范化粒度消融：仅保留异常类型
python -m long_horizon_rl.main --config-name stage2_repo \
  actor_rollout_ref.model.path="${STAGE1_CKPT}" \
  trainer.experiment_name="ablation_anchor_exception_only" \
  coding_rl.step_advantage.granularity=exception_only
