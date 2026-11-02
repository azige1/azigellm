"""沙箱内单测试执行脚本：coverage 分支模式运行一个 unittest 并输出 JSON 日志。

只能依赖标准库与 coverage；由 local_exec_service 以仓库虚拟环境的 python 启动。
用法: python run_one_test.py <sandbox_dir> <test_id> <repo_root>
"""

import ast
import importlib
import importlib.util
import json
import os
import sys
import traceback
import unittest
import warnings

TRACEBACK_TAIL_LINES = 15


def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _error_line(message: str, test_file: str):
    """从 traceback 中找测试文件内的出错行号。"""
    line_no = None
    for tb_line in reversed(message.split("\n")):
        tb_line = tb_line.strip()
        if tb_line.startswith('File "') and test_file in tb_line:
            try:
                line_no = int(tb_line.rsplit(",", 1)[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
    return line_no


def _collect_condition_sources(file_path: str, start_line: int, end_line: int) -> dict:
    """解析源文件，返回 [start,end] 内各条件行对应的条件表达式源码。"""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        tree = ast.parse(f.read())

    conditions = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While)):
            test = node.test
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            test = node.iter
        elif isinstance(node, ast.IfExp):
            test = node.test
        elif isinstance(node, ast.BoolOp):
            test = node
        else:
            continue
        if start_line <= node.lineno <= end_line:
            try:
                conditions.setdefault(node.lineno, ast.unparse(test))
            except Exception:
                pass
    return conditions


def _load_file_report(cov, fut_module_path: str, sandbox_dir: str) -> dict:
    """用 coverage 公开的 JSON 报告接口取 fut_module 的行与分支数据。"""
    report_path = os.path.join(sandbox_dir, "coverage_report.json")
    cov.json_report(outfile=report_path, include=[fut_module_path])
    with open(report_path, "r") as f:
        report = json.load(f)
    target = os.path.abspath(fut_module_path)
    for file_name, file_data in report.get("files", {}).items():
        if os.path.abspath(file_name) == target or file_name.endswith("fut_module.py"):
            return file_data
    return {}


def _build_coverage_log(file_data: dict, fut_info: dict, conditions: dict) -> dict:
    """只统计被测函数行范围的分支覆盖率日志，行号 1 起。

    分支以 coverage 报告中的 [起始行, 去向行] 对计数，起始行落在被测函数范围内的才统计。
    """
    start_line = fut_info["start_line"]
    end_line = fut_info["end_line"]

    def in_range(line: int) -> bool:
        return start_line <= line <= end_line

    executed_lines = [l for l in file_data.get("executed_lines", []) if in_range(l)]
    missing = [l for l in file_data.get("missing_lines", []) if in_range(l)]
    excluded = [l for l in file_data.get("excluded_lines", []) if in_range(l)]
    executable = sorted(set(executed_lines) | set(missing))

    executed_branches = [b for b in file_data.get("executed_branches", []) if in_range(b[0])]
    missing_branches = [b for b in file_data.get("missing_branches", []) if in_range(b[0])]
    covered_branches = len(executed_branches)
    total_branches = covered_branches + len(missing_branches)

    unevaluated = {}
    for source_line, destination in missing_branches:
        condition = conditions.get(source_line)
        if condition is None:
            continue
        entry = unevaluated.setdefault(source_line, [condition, []])
        # 去向为负数表示跳出函数（如提前 return）
        if destination > 0:
            entry[1].append(destination)
    for entry in unevaluated.values():
        entry[1] = sorted(set(entry[1]))

    line_cov = (
        100.0 * (len(executable) - len(missing)) / len(executable) if executable else 100.0
    )
    branch_cov = 100.0 * covered_branches / total_branches if total_branches else 100.0

    return {
        "branch_coverage_percentage": branch_cov,
        "covered_branches": covered_branches,
        "total_branches": total_branches,
        "line_coverage_percentage": line_cov,
        "unexecuted_lines": missing,
        "unevaluated_branches": unevaluated,
        "start_line": start_line,
        "end_line": end_line,
        "num_executable_lines": len(executable),
        "num_excluded_lines": len(excluded),
        "num_unexecuted_lines": len(missing),
        "num_executed_branches": covered_branches,
    }


def _iter_test_cases(suite):
    """展开测试套件中的全部测试用例。"""
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_test_cases(item)
        else:
            yield item


def _run_unittest(test_file: str):
    """加载并运行单个测试模块，返回 (result, errors, 全部用例名)。"""
    module_name = os.path.basename(test_file)[:-3]
    module = _load_module(module_name, test_file)

    suite = unittest.TestLoader().loadTestsFromModule(module)
    all_names = [str(test) for test in _iter_test_cases(suite)]
    with open(os.devnull, "w") as devnull:
        runner = unittest.TextTestRunner(stream=devnull, verbosity=0)
        result = runner.run(suite)

    errors = []
    for test, tb in result.failures:
        errors.append({"type": "failure", "test": str(test), "message": tb})
    for test, tb in result.errors:
        errors.append({"type": "error", "test": str(test), "message": tb})
    return result, errors, all_names


def main() -> int:
    sandbox_dir, test_id, repo_root = sys.argv[1], sys.argv[2], sys.argv[3]

    for path in (sandbox_dir, repo_root):
        if path not in sys.path:
            sys.path.insert(0, path)

    output = {
        "test_id": test_id,
        "run_test_log": None,
        "run_test_errors": [],
        "coverage_logs": [],
        "captured_arg_logs": {},
        "verdict": "test_defect",
        "env_noise": [],
    }

    with open(os.path.join(sandbox_dir, "fut_info.json"), "r") as f:
        fut_manifest = json.load(f)
    fut_infos = fut_manifest["functions"]

    fut_module_path = os.path.join(sandbox_dir, "fut_module.py")
    test_file = os.path.join(sandbox_dir, f"test_{test_id}.py")

    import coverage

    cov = coverage.Coverage(
        data_file=os.path.join(sandbox_dir, f".coverage_{test_id}"),
        branch=True,
        include=[os.path.abspath(fut_module_path)],
    )
    cov.start()

    env_noise = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fut_module = _load_module("fut_module", fut_module_path)
        except Exception:
            cov.stop()
            tb = traceback.format_exc()
            output["run_test_log"] = {
                "valid": False,
                "passed_names": [],
                "failed_names": [],
                "errored_names": [],
                "passed_count": 0,
                "failed_count": 0,
                "errored_count": 0,
            }
            output["run_test_errors"] = [
                {"type": "error", "test": "import_fut_module", "message": tb, "line": None}
            ]
            output["env_noise"] = [str(w.message) for w in caught]
            print("CODEHARVEST_RESULT " + json.dumps(output))
            return 0

    for warning in caught:
        if "SyntaxWarning" in str(warning.category):
            env_noise.append(str(warning.message))
    output["env_noise"] = env_noise

    # 挂参数采集钩子
    try:
        import standalone_capture

        for fut_info in fut_infos:
            try:
                captor = standalone_capture.CAPTORS.setdefault(
                    fut_info["id"], standalone_capture.ArgumentCaptor(fut_info["id"])
                )
                owner = (
                    getattr(fut_module, fut_info["class_name"])
                    if fut_info.get("class_name")
                    else fut_module
                )
                standalone_capture.install_hook(owner, fut_info["name"], captor)
            except AttributeError:
                continue
    except Exception:
        pass

    try:
        result, errors, all_names = _run_unittest(test_file)
    except Exception:
        cov.stop()
        tb = traceback.format_exc()
        output["run_test_log"] = {
            "valid": False,
            "passed_names": [],
            "failed_names": [],
            "errored_names": [],
            "passed_count": 0,
            "failed_count": 0,
            "errored_count": 0,
        }
        output["run_test_errors"] = [
            {"type": "error", "test": f"load_{test_id}", "message": tb, "line": None}
        ]
        print("CODEHARVEST_RESULT " + json.dumps(output))
        return 0

    cov.stop()
    cov.save()

    failed_names = [str(t) for t, _ in result.failures]
    errored_names = [str(t) for t, _ in result.errors]
    skipped_names = {str(t) for t, _ in result.skipped}
    not_passed = set(failed_names) | set(errored_names) | skipped_names
    passed_names = [name for name in all_names if name not in not_passed]

    for entry in errors:
        entry["line"] = _error_line(entry["message"], test_file)
        if "UncopyableArgumentError" in entry["message"]:
            entry["defect_reason"] = "uncopyable_argument"
        entry["message"] = "\n".join(entry["message"].split("\n")[-TRACEBACK_TAIL_LINES:])

    output["run_test_log"] = {
        "valid": result.wasSuccessful(),
        "passed_names": passed_names,
        "failed_names": failed_names,
        "errored_names": errored_names,
        "passed_count": len(passed_names),
        "failed_count": len(failed_names),
        "errored_count": len(errored_names),
    }
    output["run_test_errors"] = errors

    try:
        file_data = _load_file_report(cov, fut_module_path, sandbox_dir)
        for fut_info in fut_infos:
            conditions = _collect_condition_sources(
                fut_module_path, fut_info["start_line"], fut_info["end_line"]
            )
            output["coverage_logs"].append(_build_coverage_log(file_data, fut_info, conditions))
    except Exception:
        # 覆盖率统计失败不影响通过与否的判定，但要把原因带回，避免静默记为 0
        output["coverage_logs"] = []
        output["coverage_error"] = traceback.format_exc()[-1000:]

    try:
        import standalone_capture

        output["captured_arg_logs"] = {
            func_id: captor.records
            for func_id, captor in standalone_capture.CAPTORS.items()
            if captor.records
        }
    except Exception:
        pass

    output["verdict"] = "passed" if result.wasSuccessful() else "test_defect"
    print("CODEHARVEST_RESULT " + json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
