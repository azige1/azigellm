import inspect
import textwrap

_APPLIED = False


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError("版本锁定补丁目标缺失或不唯一")
    return source.replace(old, new)


def _patch_external_teachers() -> None:
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    init_source = textwrap.dedent(inspect.getsource(AgentLoopWorker.__init__))
    init_source = _replace_once(
        init_source,
        "    if self.distillation_enabled:\n",
        '    if self.distillation_enabled and not config.teacher_scoring.get("external", False):\n',
    )
    init_namespace = AgentLoopWorker.__init__.__globals__.copy()
    exec(compile(init_source, inspect.getsourcefile(AgentLoopWorker.__init__), "exec"), init_namespace)
    AgentLoopWorker.__init__ = init_namespace["__init__"]

    score_source = textwrap.dedent(inspect.getsource(AgentLoopWorker._compute_teacher_logprobs))
    score_source = _replace_once(
        score_source,
        "    if self.distillation_enabled and not validate:\n",
        '    if self.distillation_enabled and not validate and not self.config.teacher_scoring.get("external", False):\n',
    )
    score_namespace = AgentLoopWorker._compute_teacher_logprobs.__globals__.copy()
    exec(compile(score_source, inspect.getsourcefile(AgentLoopWorker._compute_teacher_logprobs), "exec"), score_namespace)
    AgentLoopWorker._compute_teacher_logprobs = score_namespace["_compute_teacher_logprobs"]


def _patch_empty_distillation_metrics() -> None:
    from verl.trainer.distillation import losses as base_losses
    from verl.utils.metric import AggregationType, Metric

    original = base_losses.compute_distillation_loss_range

    def safe_range(distillation_losses, response_mask):
        valid = response_mask.bool().to_padded_tensor(False) if response_mask.is_nested else response_mask.bool()
        if valid.any():
            return original(distillation_losses, response_mask)
        return {
            "distillation/loss_min": Metric(AggregationType.MIN, 0.0),
            "distillation/loss_max": Metric(AggregationType.MAX, 0.0),
        }

    base_losses.compute_distillation_loss_range = safe_range


def install_runtime_patches() -> None:
    global _APPLIED
    if _APPLIED:
        return
    import transformers
    import verl
    import vllm

    if verl.__version__ != "0.9.1" or vllm.__version__ != "0.24.0" or transformers.__version__ != "5.9.0":
        raise RuntimeError("运行版本必须固定为 verl 0.9.1、vLLM 0.24.0、transformers 5.9.0")
    _patch_external_teachers()
    _patch_empty_distillation_metrics()
    _APPLIED = True
