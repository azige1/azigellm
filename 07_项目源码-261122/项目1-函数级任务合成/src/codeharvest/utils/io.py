"""抽取结果 / 测试结果的 JSON 读写。"""

import json
from pathlib import Path

from codeharvest.core.functions import FunctionRecord
from codeharvest.core.methods import MethodRecord
from codeharvest.core.targets import FunctionTestTarget, MethodTestTarget
from codeharvest.core.codegen_task import FunctionCodegenTask, MethodCodegenTask

def read_json_data(file_path: str | Path) -> dict | list:
    with open(file_path, "r") as f:
        return json.load(f)

def read_callable_records(file_path: str | Path) -> list[FunctionRecord | MethodRecord]:
    data = read_json_data(file_path)
    functions = []
    for func_data in data:
        if func_data.get("function_id"):
            functions.append(FunctionRecord(**func_data))
        elif func_data.get("method_id"):
            functions.append(MethodRecord(**func_data))
        else:
            raise ValueError("Unknown input type")
    return functions

def read_test_targets(
    file_path: str | Path,
) -> list[FunctionTestTarget | MethodTestTarget]:
    data = read_json_data(file_path)
    functions = []
    for func_data in data:
        if func_data.get("function_id"):
            
            func_data["file"]["file_module"]["repo"]["repo_name"] = func_data["file"][
                "file_module"
            ]["repo"]["repo_id"]
            func_data["file"]["file_module"]["repo"]["repo_org"] = func_data["file"][
                "file_module"
            ]["repo"]["repo_id"]
            functions.append(FunctionTestTarget(**func_data))
        elif func_data.get("method_id"):
            func_data["file"]["file_module"]["repo"]["repo_name"] = func_data["file"][
                "file_module"
            ]["repo"]["repo_id"]
            func_data["file"]["file_module"]["repo"]["repo_org"] = func_data["file"][
                "file_module"
            ]["repo"]["repo_id"]
            functions.append(MethodTestTarget(**func_data))
        else:
            raise ValueError("Unknown input type")
    return functions

def dump_callable_records(
    functions: list[FunctionRecord | MethodRecord] | list[FunctionTestTarget], file_path: str | Path
) -> None:
    data = [func.model_dump() for func in functions]
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def dump_test_targets(
    functions: list[FunctionTestTarget | MethodTestTarget], file_path: str | Path
) -> None:
    data = [func.model_dump() for func in functions]
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def dump_codegen_tasks(
    codegen_problems: list[FunctionCodegenTask | MethodCodegenTask],
    file_path: str | Path,
) -> None:
    data = [prob.model_dump() for prob in codegen_problems]
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
