"""RL 训练入口：把终端任务环境注册进 skyrl_gym 并启动 PPO 训练。

用法：
    ray start --head
    python -m rl.main --config-dir rl/confs --config-name base
"""
import sys

import ray
from skyrl_gym.envs import register
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    register(
        id="terminal_task",
        entry_point="rl.env:TerminalTaskEnv",
    )
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
