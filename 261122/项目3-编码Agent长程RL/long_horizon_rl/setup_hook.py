"""Ray worker 进程启动钩子：在 verl 损失模块导入后替换 ppo_loss。

hook 执行时不直接导入 verl.workers：已导入的模块立即打补丁，尚未导入的模块
通过 meta_path 查找器在导入完成后打补丁。``engine_workers`` 在导入时按名绑定
``ppo_loss``，因此 ``losses.ppo_loss`` 与 ``engine_workers.ppo_loss`` 两处都要替换。
"""

from __future__ import annotations

import importlib.abc
import sys

_LOSSES_MODULE = "verl.workers.utils.losses"
_ENGINE_WORKERS_MODULE = "verl.workers.engine_workers"
_PATCHED_FLAG = "_coding_rl_loss_patched"


def _patch_module(module) -> None:
    if getattr(module, _PATCHED_FLAG, False):
        return
    from long_horizon_rl.losses import coding_ppo_loss

    module.ppo_loss = coding_ppo_loss
    setattr(module, _PATCHED_FLAG, True)


class _PatchAfterLoad(importlib.abc.Loader):
    """包装原 loader：模块执行完成后立即替换 ppo_loss。"""

    def __init__(self, wrapped_loader) -> None:
        self._wrapped = wrapped_loader

    def create_module(self, spec):
        return self._wrapped.create_module(spec)

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        _patch_module(module)


class _LossModuleFinder(importlib.abc.MetaPathFinder):
    """只拦截两个目标模块的导入，其余模块交给后续查找器。"""

    def __init__(self) -> None:
        self._searching = False

    def find_spec(self, fullname, path, target=None):
        if self._searching or fullname not in (_LOSSES_MODULE, _ENGINE_WORKERS_MODULE):
            return None
        self._searching = True
        try:
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = _PatchAfterLoad(spec.loader)
                    return spec
        finally:
            self._searching = False
        return None


def apply_patches() -> None:
    """worker_process_setup_hook 入口。"""
    for name in (_LOSSES_MODULE, _ENGINE_WORKERS_MODULE):
        if name in sys.modules:
            _patch_module(sys.modules[name])
    if not any(isinstance(finder, _LossModuleFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LossModuleFinder())
