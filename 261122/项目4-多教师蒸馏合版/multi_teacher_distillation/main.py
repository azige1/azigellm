from pprint import pprint

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.trainer.ppo.v1 import AgentLoopManagerTQ
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.logging_utils import configure_verl_logging

from . import losses  # noqa: F401
from .patches import install_runtime_patches
from .trainer import DistillationTrainer


@ray.remote
class TrainingRunner:
    def __init__(self) -> None:
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self) -> None:
        self.agent_loop_manager = AgentLoopManagerTQ.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config: DictConfig) -> None:
        configure_verl_logging()
        import transfer_queue as tq

        install_runtime_patches()
        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config
        tq.init(config.transfer_queue)
        succeeded = False
        try:
            self.trainer = DistillationTrainer(config=config)
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


@hydra.main(config_path="../experiments/configs", config_name="distill_8xa800", version_base=None)
def main(config: DictConfig) -> None:
    auto_set_device(config)
    use_reference = (
        need_reference_policy(config)
        or bool(config.get("teacher_scoring", {}).get("anchor_from_reference", True))
        or bool(config.get("reliability_weighting", {}).get("enabled", True))
    )
    validate_config(config=config, use_reference_policy=use_reference, use_critic=need_critic(config))
    run_ppo(config, task_runner_class=TrainingRunner)


if __name__ == "__main__":
    main()
