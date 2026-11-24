"""Training entry point and Ray TaskRunner for mathematical reasoning RL (verl 0.9.1 V1)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from pprint import pprint
from typing import Optional

import ray
from omegaconf import DictConfig, OmegaConf, open_dict

from math_reasoning_rl.config import EntropyBandConfig, RewardRoutingConfig


def build_runtime_env_with_hooks(config: DictConfig, base_runtime_env: dict) -> dict:
    """Attach worker_process_setup_hook and 6.1/6.2 env vars into Ray runtime_env."""
    env_vars = dict(base_runtime_env.get("env_vars", {}))

    math_rl = config.get("math_rl", {}) or {}
    band_cfg_raw = math_rl.get("entropy_band", {}) or {}
    band_cfg = EntropyBandConfig(
        low=float(band_cfg_raw.get("low", 0.45)),
        high=float(band_cfg_raw.get("high", 0.65)),
        coef=float(band_cfg_raw.get("coef", 0.005)),
        enabled=bool(band_cfg_raw.get("enabled", True)),
    )
    env_vars.update(band_cfg.to_env())

    reward_cfg_raw = math_rl.get("reward_routing", {}) or {}
    reward_cfg = RewardRoutingConfig(
        repetition_ngram_n=int(reward_cfg_raw.get("repetition_ngram_n", 4)),
        repetition_threshold=float(reward_cfg_raw.get("repetition_threshold", 0.3)),
        require_think_tags=bool(reward_cfg_raw.get("require_think_tags", True)),
        judge_base_url=str(reward_cfg_raw.get("judge_base_url", "http://127.0.0.1:8001/v1")),
        judge_api_key=str(reward_cfg_raw.get("judge_api_key", "EMPTY")),
        judge_model=str(reward_cfg_raw.get("judge_model", "Qwen/Qwen3-32B")),
        judge_timeout_s=float(reward_cfg_raw.get("judge_timeout_s", 60.0)),
        judge_max_retries=int(reward_cfg_raw.get("judge_max_retries", 3)),
    )
    env_vars.update(
        {
            "MATH_RL_REP_NGRAM_N": str(reward_cfg.repetition_ngram_n),
            "MATH_RL_REP_THRESHOLD": str(reward_cfg.repetition_threshold),
            "MATH_RL_REQUIRE_THINK": "1" if reward_cfg.require_think_tags else "0",
            "MATH_RL_JUDGE_BASE_URL": reward_cfg.judge_base_url,
            "MATH_RL_JUDGE_API_KEY": reward_cfg.judge_api_key,
            "MATH_RL_JUDGE_MODEL": reward_cfg.judge_model,
            "MATH_RL_JUDGE_TIMEOUT_S": str(reward_cfg.judge_timeout_s),
            "MATH_RL_JUDGE_MAX_RETRIES": str(reward_cfg.judge_max_retries),
        }
    )

    updated = dict(base_runtime_env)
    updated["env_vars"] = env_vars
    updated["worker_process_setup_hook"] = "math_reasoning_rl.runtime_hooks.setup_worker_patches"
    return updated


def load_merged_config(config_path: str | Path, overrides: Optional[list[str]] = None) -> DictConfig:
    """Compose verl base ppo_trainer config and merge project YAML + CLI overrides."""
    from hydra import compose, initialize_config_module

    with initialize_config_module(version_base=None, config_module="verl.trainer.config"):
        base_cfg = compose(config_name="ppo_trainer")

    override_yaml = OmegaConf.load(str(config_path))
    OmegaConf.set_struct(base_cfg, False)
    merged = OmegaConf.merge(base_cfg, override_yaml)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, cli_cfg)

    with open_dict(merged):
        # Ensure sampler_kwargs mirrors train_batch_size and filter_groups settings.
        sampler_kwargs = merged.trainer.v1.sampler.get("sampler_kwargs", {}) or {}
        sampler_kwargs["train_batch_size"] = int(merged.data.train_batch_size)
        fg = merged.algorithm.get("filter_groups", {}) or {}
        if fg.get("enable", False):
            sampler_kwargs["filter_groups_metric"] = str(fg.get("metric", "reward"))
            sampler_kwargs["max_inflight_gen_batches"] = int(fg.get("max_inflight_gen_batches", 2))
        else:
            sampler_kwargs["filter_groups_metric"] = None
        ov_cfg = merged.get("math_rl", {}).get("overflow_reuse", {}) or {}
        if ov_cfg:
            sampler_kwargs["reuse_enabled"] = bool(ov_cfg.get("enabled", True))
            sampler_kwargs["max_reuse_steps"] = int(ov_cfg.get("max_reuse_steps", 1))
            sampler_kwargs["max_reuse_ratio"] = float(ov_cfg.get("max_reuse_ratio", 0.5))
            sampler_kwargs["is_ratio_clip"] = float(ov_cfg.get("is_ratio_clip", 2.0))
        merged.trainer.v1.sampler.sampler_kwargs = sampler_kwargs

    return merged


@ray.remote
class MathReasoningTaskRunner:
    """Remote V1 task runner mirroring ``verl.trainer.main_ppo.TaskRunnerV1``."""

    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ
        from verl.utils.import_utils import load_class_from_fqn

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            agent_loop_manager_cls = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            agent_loop_manager_cls = AgentLoopManagerTQ

        self.agent_loop_manager = agent_loop_manager_cls.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config: DictConfig) -> None:
        import transfer_queue as tq

        from math_reasoning_rl.runtime_hooks import setup_worker_patches
        from math_reasoning_rl.trainer import MathReasoningPPOTrainer
        from verl.utils.logging_utils import configure_verl_logging

        configure_verl_logging()
        setup_worker_patches()

        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config

        tq.init(config.transfer_queue)
        succeeded = False
        try:
            self.trainer = MathReasoningPPOTrainer(config=config)
            self.trainer.init()
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(self.trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


def run_training(config: DictConfig) -> None:
    """Validate config, initialize Ray with setup hook, and launch MathReasoningTaskRunner."""
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    from verl.utils.device import auto_set_device

    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env(config)
        ray_init_kwargs = OmegaConf.to_container(config.ray_kwargs.get("ray_init", {}), resolve=True) or {}
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {}) or {}
        if config.transfer_queue.enable:
            default_runtime_env.setdefault("env_vars", {})["TRANSFER_QUEUE_ENABLE"] = "1"
        merged_runtime_env = OmegaConf.to_container(
            OmegaConf.merge(default_runtime_env, runtime_env_kwargs),
            resolve=True,
        )
        ray_init_kwargs["runtime_env"] = build_runtime_env_with_hooks(config, merged_runtime_env)
        ray.init(**ray_init_kwargs)

    runner = MathReasoningTaskRunner.remote()
    ray.get(runner.run.remote(config))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Launch math reasoning RL training on verl 0.9.1 V1.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "configs" / "dapo_8xa800_stage1_8k.yaml"),
        help="Path to project YAML configuration.",
    )
    args, overrides = parser.parse_known_args(argv)
    os.environ.setdefault("TRANSFER_QUEUE_ENABLE", "1")
    config = load_merged_config(args.config, overrides=overrides)
    run_training(config)


if __name__ == "__main__":
    main()
