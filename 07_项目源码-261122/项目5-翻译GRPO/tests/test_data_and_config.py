"""Unit tests for data preparation, YAML config parsing, and 5.5 checkpoint selection."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from translation_grpo.checkpoint_eval import (
    CheckpointEvalRecord,
    compute_sign_test,
    detect_overoptimization_step,
    evaluate_checkpoint_vs_baseline,
    run_consistency_diagnostic,
    select_best_checkpoint,
)
from translation_grpo.data_prep import prepare_rl_datasets
from translation_grpo.judge_client import AsyncOpenAIJudgeClient, JudgeClientConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_prepare_rl_datasets_balances_and_aligns_sample_id(tmp_path):
    zh_path = PROJECT_ROOT / "data" / "sample_zh2en.jsonl"
    en_path = PROJECT_ROOT / "data" / "sample_en2zh.jsonl"
    out_dir = tmp_path / "processed"

    summary = prepare_rl_datasets([zh_path, en_path], out_dir, val_ratio=0.2, seed=42)
    assert summary["total_balanced"] == 20
    train_info = summary["splits"]["train"]
    val_info = summary["splits"]["val"]
    assert train_info["zh2en"] == train_info["en2zh"]
    assert val_info["zh2en"] == val_info["en2zh"]

    val_lines = [json.loads(line) for line in Path(val_info["jsonl"]).read_text(encoding="utf-8").splitlines() if line]
    for row in val_lines:
        assert "prompt" in row and isinstance(row["prompt"], list)
        assert row["extra_info"]["sample_id"]
        assert row["extra_info"]["reference"] == row["reward_model"]["ground_truth"]
        assert row["extra_info"]["source_len"] > 0


def test_configs_parse_and_match_specs():
    train_cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "grpo_8xa800_qwen3_8b.yaml").read_text(encoding="utf-8"))
    assert train_cfg["trainer"]["n_gpus_per_node"] == 6
    assert train_cfg["trainer"]["total_training_steps"] == 200
    assert train_cfg["trainer"]["save_freq"] == 20
    assert train_cfg["trainer"]["use_v1"] is True
    assert train_cfg["trainer"]["v1"]["trainer_mode"] == "sync"
    assert train_cfg["data"]["train_batch_size"] == 32
    assert train_cfg["actor_rollout_ref"]["rollout"]["n"] == 8
    assert train_cfg["actor_rollout_ref"]["model"]["lora_rank"] == 64
    assert train_cfg["actor_rollout_ref"]["model"]["lora_alpha"] == 128
    assert train_cfg["actor_rollout_ref"]["actor"]["clip_ratio_low"] == pytest.approx(0.2)
    assert train_cfg["actor_rollout_ref"]["actor"]["clip_ratio_high"] == pytest.approx(0.28)
    assert train_cfg["actor_rollout_ref"]["actor"]["use_kl_loss"] is False
    assert train_cfg["algorithm"]["use_kl_in_reward"] is False
    assert train_cfg["algorithm"]["adv_estimator"] == "grpo"

    sidecar = yaml.safe_load((PROJECT_ROOT / "configs" / "translation_grpo.yaml").read_text(encoding="utf-8"))
    assert sidecar["reward"]["terminology"] == pytest.approx(1.5)
    assert sidecar["reward"]["accuracy"] == pytest.approx(1.2)
    assert sidecar["advantage"]["anchor_lambda"] == pytest.approx(0.3)

    ref_cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "grpo_ref_30b_a3b.yaml").read_text(encoding="utf-8"))
    assert ref_cfg["data"]["train_batch_size"] == 64
    assert ref_cfg["trainer"]["n_gpus_per_node"] == 8

    eval_cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "judge_eval.yaml").read_text(encoding="utf-8"))
    assert eval_cfg["external_strong_judge"]["model"] == "gpt-5"
    assert eval_cfg["protocol"]["eval_decoding_temperature"] == pytest.approx(0.7)


def test_consistency_diagnostic_and_checkpoint_selection():
    z, p_val = compute_sign_test(wins=412, losses=290)
    assert z > 4.5
    assert p_val < 0.01

    records = [
        CheckpointEvalRecord(100, "ckpt_100", -1.5, 1.05, 380, 310, 100, 0.5507, 2.66, 0.008),
        CheckpointEvalRecord(120, "ckpt_120", -1.3, 1.01, 412, 290, 88, 0.5869, 4.60, 4e-6),
        CheckpointEvalRecord(140, "ckpt_140", -1.1, 1.02, 405, 295, 90, 0.5786, 4.15, 3e-5),
        CheckpointEvalRecord(160, "ckpt_160", -0.9, 1.04, 395, 305, 90, 0.5643, 3.40, 7e-4),
    ]
    assert select_best_checkpoint(records).step == 120
    assert detect_overoptimization_step(records, consecutive_divergences=2) == 160

    async def _online_transport(payload):
        msg = payload["messages"][-1]["content"]
        if "cand_bad" in msg:
            return json.dumps({"errors": [{"span": "x", "dim": "fluency", "severity": "major"}]})
        return json.dumps({"errors": []})

    async def _strong_transport(payload):
        msg = payload["messages"][-1]["content"]
        if "Translation A:" in msg:
            slot_a = msg.split("Translation A:\n")[1].split("\n\nTranslation B:")[0].strip()
            return json.dumps({"winner": "A" if "cand_good" in slot_a else "B"})
        if "cand_bad" in msg:
            return json.dumps({"errors": [{"span": "x", "dim": "terminology", "severity": "major"}]})
        return json.dumps({"errors": []})

    online_c = AsyncOpenAIJudgeClient(JudgeClientConfig(enable_cache=False), transport=_online_transport)
    strong_c = AsyncOpenAIJudgeClient(JudgeClientConfig(enable_cache=False), transport=_strong_transport)

    groups = [
        {"source": "源句一", "direction": "zh2en", "candidates": ["cand_good", "cand_bad"], "has_terminology": True},
        {"source": "源句二", "direction": "zh2en", "candidates": ["cand_good", "cand_bad"], "has_terminology": False},
    ]
    diag = asyncio.run(run_consistency_diagnostic(online_c, strong_c, groups))
    assert diag.penalty_pearson_r == pytest.approx(1.0)
    assert diag.top1_disagreement_rate == pytest.approx(0.0)

    ckpt_rec = asyncio.run(
        evaluate_checkpoint_vs_baseline(
            strong_c,
            step=120,
            checkpoint_path="ckpt_120",
            eval_samples=[{"source": "源句一", "direction": "zh2en"}],
            sft_baseline_outputs=["cand_bad"],
            checkpoint_outputs=["cand_good"],
            online_reward_mean=-1.2,
        )
    )
    assert ckpt_rec.pairwise_wins == 1
    assert ckpt_rec.strong_mqm_penalty_mean == pytest.approx(0.0)
