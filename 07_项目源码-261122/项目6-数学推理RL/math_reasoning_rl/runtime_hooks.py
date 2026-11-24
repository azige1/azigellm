"""Ray worker_process_setup_hook that patches losses.ppo_loss and engine_workers.ppo_loss post-import."""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from typing import Any

from math_reasoning_rl.entropy_band_loss import entropy_band_ppo_loss, set_original_ppo_loss

_LOSSES_MODULE = "verl.workers.utils.losses"
_ENGINE_WORKERS_MODULE = "verl.workers.engine_workers"
_PATCHED_FLAG = "_math_rl_entropy_band_patched"


def _patch_losses_module(module: Any) -> None:
    """Replace ``losses.ppo_loss`` while preserving original implementation."""
    if getattr(module, _PATCHED_FLAG, False):
        return
    orig_fn = getattr(module, "ppo_loss", None)
    if orig_fn is not None and orig_fn is not entropy_band_ppo_loss:
        set_original_ppo_loss(orig_fn)
        setattr(module, "_orig_ppo_loss", orig_fn)
    module.ppo_loss = entropy_band_ppo_loss
    setattr(module, _PATCHED_FLAG, True)


def _patch_engine_workers_module(module: Any) -> None:
    """Replace ``engine_workers.ppo_loss`` (bound by name on import)."""
    if getattr(module, _PATCHED_FLAG, False):
        return
    orig_fn = getattr(module, "ppo_loss", None)
    if orig_fn is not None and orig_fn is not entropy_band_ppo_loss:
        set_original_ppo_loss(orig_fn)
    module.ppo_loss = entropy_band_ppo_loss
    setattr(module, _PATCHED_FLAG, True)


def apply_patch_if_loaded() -> None:
    """Patch target modules immediately if they are already imported in sys.modules."""
    if _LOSSES_MODULE in sys.modules:
        _patch_losses_module(sys.modules[_LOSSES_MODULE])
    if _ENGINE_WORKERS_MODULE in sys.modules:
        _patch_engine_workers_module(sys.modules[_ENGINE_WORKERS_MODULE])


class _PostImportLoaderWrapper(importlib.abc.Loader):
    """Loader wrapper that invokes post-load patching right after module execution."""

    def __init__(self, wrapped_loader: importlib.abc.Loader, target_name: str) -> None:
        self._wrapped = wrapped_loader
        self._target_name = target_name

    def create_module(self, spec):
        if hasattr(self._wrapped, "create_module"):
            return self._wrapped.create_module(spec)
        return None

    def exec_module(self, module):
        if hasattr(self._wrapped, "exec_module"):
            self._wrapped.exec_module(module)
        elif hasattr(self._wrapped, "load_module"):
            module = self._wrapped.load_module(self._target_name)
        if self._target_name == _LOSSES_MODULE:
            _patch_losses_module(module)
        elif self._target_name == _ENGINE_WORKERS_MODULE:
            _patch_engine_workers_module(module)


class _VerlLossPatchFinder(importlib.abc.MetaPathFinder):
    """MetaPathFinder intercepting imports of ``verl.workers.utils.losses`` and ``engine_workers``."""

    def __init__(self) -> None:
        self._in_find = False

    def find_spec(self, fullname, path, target=None):
        if self._in_find or fullname not in (_LOSSES_MODULE, _ENGINE_WORKERS_MODULE):
            return None
        self._in_find = True
        try:
            for finder in sys.meta_path:
                if finder is self:
                    continue
                if hasattr(finder, "find_spec"):
                    spec = finder.find_spec(fullname, path, target)
                    if spec is not None and spec.loader is not None:
                        spec.loader = _PostImportLoaderWrapper(spec.loader, fullname)
                        return spec
        finally:
            self._in_find = False
        return None


def setup_worker_patches() -> None:
    """Ray worker_process_setup_hook entry point.

    Does NOT import ``verl.workers`` at hook execution time; instead patches already-loaded
    modules and installs a meta_path post-import hook for ``losses.ppo_loss`` and
    ``engine_workers.ppo_loss``.
    """
    apply_patch_if_loaded()
    if not any(isinstance(f, _VerlLossPatchFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _VerlLossPatchFinder())
