"""任务注册表。"""

from . import cpp2rust, py2node

_TASKS = {
    py2node.SPEC.name: py2node.SPEC,
    cpp2rust.SPEC.name: cpp2rust.SPEC,
}


def get_task_spec(name):
    """按名字取任务定义，未知任务给出清晰报错。"""
    if name not in _TASKS:
        raise ValueError(f"未知任务 '{name}'，可选: {sorted(_TASKS)}")
    return _TASKS[name]


def task_names():
    return sorted(_TASKS)
