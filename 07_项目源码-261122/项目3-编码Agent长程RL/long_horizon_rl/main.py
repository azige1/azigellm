"""训练启动入口：加载配置、注入 worker_process_setup_hook 并启动 CodingTrainingRunner。"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

from long_horizon_rl.trainer import CodingTrainingRunner


def _attach_setup_hook(config: DictConfig) -> None:
    """将自定义 loss 补丁函数注册到 Ray runtime_env.worker_process_setup_hook。"""
    with open_dict(config):
        if "ray_kwargs" not in config:
            config.ray_kwargs = {}
        if "ray_init" not in config.ray_kwargs:
            config.ray_kwargs.ray_init = {}
        if "runtime_env" not in config.ray_kwargs.ray_init or config.ray_kwargs.ray_init.runtime_env is None:
            config.ray_kwargs.ray_init.runtime_env = {}
        config.ray_kwargs.ray_init.runtime_env["worker_process_setup_hook"] = "long_horizon_rl.setup_hook.apply_patches"


def launch_training(config: DictConfig) -> None:
    """校验配置并以 V1 CodingTrainingRunner 启动多卡 RL 训练。"""
    auto_set_device(config)
    _attach_setup_hook(config)

    # 强制关闭 bypass_mode：3.3 节压缩改写历史后必须由 actor 重算 old_log_prob
    if config.algorithm.get("rollout_correction", None) is not None:
        with open_dict(config):
            config.algorithm.rollout_correction.bypass_mode = False

    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    run_ppo(config, task_runner_class=CodingTrainingRunner)


@hydra.main(config_path="../configs", config_name="stage2_repo", version_base=None)
def main(config: DictConfig) -> None:
    # 将相对路径的 agent_loop_config_path 规范化为绝对路径，避免 Ray worker cwd 差异
    agent_cfg_path = OmegaConf.select(config, "actor_rollout_ref.rollout.agent.agent_loop_config_path")
    if agent_cfg_path and not os.path.isabs(str(agent_cfg_path)):
        resolved = Path(__file__).resolve().parent.parent / str(agent_cfg_path)
        if resolved.exists():
            with open_dict(config):
                config.actor_rollout_ref.rollout.agent.agent_loop_config_path = str(resolved)
    launch_training(config)


if __name__ == "__main__":
    main()
