from pathlib import Path

import pytest


def test_sync_trainer_and_sampled_estimator_exist():
    pytest.importorskip("verl")
    from verl.trainer.distillation.losses import get_distillation_loss_settings
    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
    from multi_teacher_distillation import losses
    from multi_teacher_distillation.trainer import DistillationTrainer

    assert issubclass(DistillationTrainer, PPOTrainerSync)
    assert get_distillation_loss_settings("sampled_reverse_kl").use_estimator


def test_agent_loop_output_zero_reward_skips_reward_worker():
    pytest.importorskip("verl")
    from multi_teacher_distillation.protocol import build_loop_output

    output = build_loop_output([1], [2, 3], [1, 0], [-0.1, 0.0], {"task_type": "table"}, 0.0)
    assert output.response_mask == [1, 0]
    assert output.reward_score == 0.0


def test_distillation_config_has_required_policy_gradient_contract():
    root = Path(__file__).resolve().parents[1]
    config = (root / "experiments/configs/distill_8xa800.yaml").read_text(encoding="utf-8")
    assert "use_policy_gradient: true" in config
    assert "policy_loss_mode: vanilla" in config
    assert "calculate_log_probs: true" in config
    assert "bypass_mode: true" in config
    assert "loss_mode: sampled_reverse_kl" in config
    assert "external: true" in config
    assert "total_training_steps: 48" in config
    assert "train_batch_size: 64" in config
    assert "anchor_from_reference: true" in config
