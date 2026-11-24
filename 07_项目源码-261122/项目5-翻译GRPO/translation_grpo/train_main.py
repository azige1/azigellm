"""Training entrypoint for Translation GRPO using verl 0.9.1 V1 TaskRunner pattern."""

from __future__ import annotations

from pprint import pprint

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.logging_utils import configure_verl_logging

from .trainer import TranslationGRPOTrainer


@ray.remote
class TranslationTaskRunner:
    """Ray remote TaskRunner instantiating TranslationGRPOTrainer without registering a new trainer_mode."""

    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        """Initialize V1 AgentLoopManagerTQ using trainer LLM/teacher/reward handles."""
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ

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

    def run(self, config: DictConfig):
        """Initialize TransferQueue, run TranslationGRPOTrainer, and tear down cleanly."""
        configure_verl_logging()
        import transfer_queue as tq

        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config

        tq.init(config.transfer_queue)
        succeeded = False
        try:
            self.trainer = TranslationGRPOTrainer(config=config)
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


@hydra.main(config_path="../configs", config_name="grpo_8xa800_qwen3_8b", version_base=None)
def main(config: DictConfig):
    """Hydra main entry point for Translation GRPO."""
    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    run_ppo(config, task_runner_class=TranslationTaskRunner)


if __name__ == "__main__":
    main()
