"""自包含的调用采集插件：pytest 插件与 examples 脚本模式共用。

本文件会被复制进执行沙箱或经 PYTHONPATH 注入被测仓库的虚拟环境，
不能 import codeharvest 包内的任何模块，只能依赖标准库（pytest 模式另需 pytest）。
采集规则：每个函数保留最近 3 条，按调用者去重；异常调用记录异常类型与消息后原样抛出。
"""

import ast
import datetime
import functools
import importlib
import inspect
import json
import os
import sys
from decimal import Decimal
from collections.abc import Iterable

MAX_RECORDS_PER_FUNCTION = 3
TRUNCATE_HEAD_TAIL = 90

CAPTORS = {}


def caller_info(frame):
    """取调用者的模块、函数名与行号；frame 为包装函数外一层的调用帧。"""
    if frame is None:
        return None
    return {
        "module": frame.f_globals.get("__name__"),
        "func_name": frame.f_code.co_name,
        "lineno": frame.f_lineno,
    }


class ValueSerializers:
    """把运行时值序列化成可写 JSON 的形式。"""

    @staticmethod
    def serialize_default(obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        try:
            return repr(obj)
        except Exception:
            pass
        if hasattr(obj, "__dict__"):
            return {k: ValueSerializers.serialize_default(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            return [ValueSerializers.serialize_default(item) for item in obj]
        return f"<unserializable: {type(obj).__name__}>"

    @staticmethod
    def serialize(obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.ismethod(obj):
            try:
                return inspect.getsource(obj)
            except (OSError, TypeError):
                return repr(obj)
        serialized = ValueSerializers.serialize_default(obj)
        if isinstance(serialized, str) and len(serialized) > 2 * TRUNCATE_HEAD_TAIL:
            return (
                serialized[:TRUNCATE_HEAD_TAIL]
                + "  ......  "
                + serialized[-TRUNCATE_HEAD_TAIL:]
            )
        if isinstance(serialized, list) and len(serialized) > 25:
            return f"<{type(obj).__name__}(len={len(serialized)})>"
        return serialized


class ArgumentCaptor:
    """挂在被测函数上记录入参（展开默认值）、返回值或异常、类型与调用者。"""

    def __init__(self, func_id: str):
        self.func_id = func_id
        self.records = []

    def instrument(self, func):
        captor = self

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 在包装函数内取调用帧，记录的是真正的调用者而不是包装层自身
            info = caller_info(sys._getframe(1))
            bound_arguments = None
            try:
                bound_arguments = inspect.signature(func).bind(*args, **kwargs)
                bound_arguments.apply_defaults()
            except (TypeError, ValueError):
                bound_arguments = None

            try:
                output = func(*args, **kwargs)
            except Exception as e:
                captor._record(func, bound_arguments, None, e, info)
                raise
            captor._record(func, bound_arguments, output, None, info)
            return output

        wrapper.__wrapped_by_argument_captor__ = True
        return wrapper

    def _record(self, func, bound_arguments, output, exception, info) -> None:
        inputs = dict(bound_arguments.arguments) if bound_arguments else {}

        if exception is not None:
            record = {
                "func_name": func.__name__,
                "func_id": self.func_id,
                "inputs": {k: ValueSerializers.serialize(v) for k, v in inputs.items()},
                "input_types": {
                    k: f"{type(v).__module__}.{type(v).__qualname__}" for k, v in inputs.items()
                },
                "exception_type": f"{type(exception).__module__}.{type(exception).__qualname__}",
                "exception_message": str(exception),
                "caller_info": info,
            }
        else:
            record = {
                "func_name": func.__name__,
                "func_id": self.func_id,
                "inputs": {k: ValueSerializers.serialize(v) for k, v in inputs.items()},
                "input_types": {
                    k: f"{type(v).__module__}.{type(v).__qualname__}" for k, v in inputs.items()
                },
                "output": ValueSerializers.serialize(output),
                "output_type": f"{type(output).__module__}.{type(output).__qualname__}",
                "caller_info": info,
            }

        # 按调用者去重，仅保留最近 MAX_RECORDS_PER_FUNCTION 条
        key = json.dumps(info, sort_keys=True) if info else f"anon:{len(self.records)}"
        self.records = [r for r in self.records if json.dumps(r["caller_info"], sort_keys=True) != key]
        self.records.append(record)
        self.records = self.records[-MAX_RECORDS_PER_FUNCTION:]


def _load_targets():
    """从环境变量指定的 JSON 文件读取目标清单。"""
    targets_file = os.environ.get("CODEHARVEST_CAPTURE_TARGETS", "")
    if not targets_file or not os.path.exists(targets_file):
        return []
    with open(targets_file, "r") as f:
        return json.load(f)


def install_hook(owner, name: str, captor: "ArgumentCaptor") -> None:
    """把 owner（模块或类）上的函数替换为采集包装版，保留 staticmethod / classmethod 包装。"""
    raw = inspect.getattr_static(owner, name)
    if isinstance(raw, staticmethod):
        func = raw.__func__
        if getattr(func, "__wrapped_by_argument_captor__", False):
            return
        setattr(owner, name, staticmethod(captor.instrument(func)))
    elif isinstance(raw, classmethod):
        func = raw.__func__
        if getattr(func, "__wrapped_by_argument_captor__", False):
            return
        setattr(owner, name, classmethod(captor.instrument(func)))
    else:
        if getattr(raw, "__wrapped_by_argument_captor__", False):
            return
        setattr(owner, name, captor.instrument(raw))


def _wrap_target(target: dict) -> None:
    """按目标清单把模块属性（或类属性）替换为采集包装版。"""
    module_name = target["module"]
    name = target["name"]
    class_name = target.get("class_name")
    func_id = target.get("func_id") or (f"{module_name}.{class_name + '.' if class_name else ''}{name}")

    module = importlib.import_module(module_name)
    captor = CAPTORS.setdefault(func_id, ArgumentCaptor(func_id))
    owner = getattr(module, class_name) if class_name else module
    install_hook(owner, name, captor)


def dump_records(output_path: str) -> None:
    """按函数 id 聚合写出采集记录。"""
    records = {func_id: captor.records for func_id, captor in CAPTORS.items() if captor.records}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=ValueSerializers.serialize_default)


def pytest_configure(config):
    """pytest 插件入口：测试收集前挂好采集钩子。"""
    for target in _load_targets():
        try:
            _wrap_target(target)
        except Exception as e:
            print(f"[standalone_capture] 跳过目标 {target}: {e!r}")


def pytest_unconfigure(config):
    """pytest 插件出口：按函数 id 写出采集结果。"""
    output_path = os.environ.get("CODEHARVEST_CAPTURE_OUTPUT", "")
    if output_path and CAPTORS:
        dump_records(output_path)


def run_script_mode(script_path: str, output_path: str) -> None:
    """脚本模式：对单个 examples/*.py 挂钩子后用 runpy 执行。"""
    import runpy

    for target in _load_targets():
        try:
            _wrap_target(target)
        except Exception as e:
            print(f"[standalone_capture] 跳过目标 {target}: {e!r}")

    try:
        runpy.run_path(script_path, run_name="__main__")
    except Exception as e:
        print(f"[standalone_capture] 脚本 {script_path} 执行出错: {e!r}")

    if CAPTORS:
        dump_records(output_path)


def _script_main(argv=None) -> int:
    """独立脚本入口：python standalone_capture.py --script x.py --output out.json"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, help="要执行的 examples 脚本路径")
    parser.add_argument("--output", required=True, help="采集结果输出路径")
    args = parser.parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(args.script))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    repo_root = os.environ.get("CODEHARVEST_CAPTURE_REPO_ROOT", "")
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    run_script_mode(args.script, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(_script_main())
