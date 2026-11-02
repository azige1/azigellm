"""本地执行服务：实现 RepoExecService，供本地直连与 Docker 内 rpyc 两种模式调用。

每个被测函数对应一次 setup → init → submit 生命周期：
init 在临时目录构建沙箱（fut_module.py + equivalence_helpers.py + 采集插件副本），
submit 逐测试起子进程运行 run_one_test.py 并聚合日志。
"""

import ast
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import rpyc

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTRUMENT_DIR = os.path.abspath(
    os.path.join(PACKAGE_DIR, "..", "analysis", "instrument")
)

# codegen 模式下候选工作区需要隔离删除的条目
CODEGEN_SCRUB_NAMES = (
    "tests",
    "pyproject.toml",
    "setup.cfg",
    "pytest.ini",
    "tox.ini",
    "conftest.py",
)

DEFAULT_TEST_TIMEOUT = 60

# 相对导入语句首行的「from .x import」部分
RELATIVE_IMPORT_HEAD = re.compile(r"from\s+\.+[\w.]*\s+import")


class RepoExecService(rpyc.Service):
    """执行被测函数测试的服务；同名方法同时提供 exposed_ 别名以支持 rpyc。"""

    def __init__(self):
        self.repo_id = None
        self.repo_path = None
        self.file_path = None
        self.funclass_names = []
        self.generated_tests = {}
        self.fut_infos = []
        self.sandbox_dir = None
        self.repo_copy_dir = None
        self.codegen_mode = False
        self.test_timeout = DEFAULT_TEST_TIMEOUT
        self.env_noise = []

    # ---------- setup ----------

    def setup_repo(self, repo_json: str) -> dict:
        try:
            repo_data = json.loads(repo_json)
            self.repo_id = repo_data.get("repo_id")
            self.repo_path = repo_data.get("repo_path")
            # Docker 模式下 repo_path 是仓库名，仓库位于容器内 /repos 之下
            if self.repo_path and not os.path.isabs(self.repo_path):
                self.repo_path = os.path.join("/repos", self.repo_path)
            if self.repo_id is None and self.repo_path:
                self.repo_id = os.path.basename(os.path.normpath(self.repo_path))
            if not self.repo_path or not os.path.isdir(self.repo_path):
                return {"output": "", "error": f"repo_path 不存在: {self.repo_path}"}
            self.test_timeout = int(repo_data.get("test_timeout", DEFAULT_TEST_TIMEOUT))
            return {"output": "", "error": ""}
        except Exception as e:
            return {"output": "", "error": repr(e)}

    def setup_function(self, fut_json: str) -> dict:
        try:
            fut_data = json.loads(fut_json)
            self.funclass_names = fut_data["funclass_names"]
            self.file_path = fut_data["file_path"]
            return {"output": "", "error": ""}
        except Exception as e:
            return {"output": "", "error": repr(e)}

    def setup_test(self, test_json: str) -> dict:
        try:
            test_data = json.loads(test_json)
            self.generated_tests = test_data["generated_tests"]
            return {"output": "", "error": ""}
        except Exception as e:
            return {"output": "", "error": repr(e)}

    # ---------- init ----------

    def init(self) -> dict:
        try:
            if not self.repo_path or not self.file_path:
                return {"output": "", "error": "setup_repo/setup_function 未先调用"}
            source_file = os.path.join(self.repo_path, self.file_path)
            if not os.path.exists(source_file):
                return {"output": "", "error": f"源文件不存在: {source_file}"}

            with open(source_file, "r", encoding="utf-8", errors="replace") as f:
                original_source = f.read()

            self.fut_infos = self._build_fut_infos(original_source)
            if not self.fut_infos:
                return {"output": "", "error": f"未在 {self.file_path} 中找到被测函数"}

            self.sandbox_dir = tempfile.mkdtemp(prefix="codeharvest_sandbox_")
            # 自等价校验直接使用原仓库；只有 codegen 判分时才复制并清理候选工作区
            self.repo_copy_dir = self.repo_path
            self.codegen_mode = False

            fut_source = self._build_fut_module_source(original_source)
            self._write_sandbox_file("fut_module.py", fut_source)
            self._copy_support_file(
                os.path.join(PACKAGE_DIR, "equivalence_helpers.py"), "equivalence_helpers.py"
            )
            self._copy_support_file(
                os.path.join(INSTRUMENT_DIR, "standalone_capture.py"), "standalone_capture.py"
            )

            for test_id, test_source in self.generated_tests.items():
                self._write_sandbox_file(f"test_{test_id}.py", test_source)

            manifest = {
                "repo_root": self.repo_copy_dir,
                "functions": self.fut_infos,
            }
            self._write_sandbox_file("fut_info.json", json.dumps(manifest, indent=2))
            self.env_noise = []
            return {"output": self.sandbox_dir, "error": "", "env_noise": []}
        except Exception as e:
            return {"output": "", "error": repr(e)}

    def _write_sandbox_file(self, name: str, content: str) -> None:
        with open(os.path.join(self.sandbox_dir, name), "w", encoding="utf-8") as f:
            f.write(content)

    def _copy_support_file(self, src: str, name: str) -> None:
        shutil.copyfile(src, os.path.join(self.sandbox_dir, name))

    def _copy_repo_workspace(self) -> None:
        """复制仓库到沙箱，只删除仓库根目录下的 tests/ 与配置类文件（不读候选源码）。"""
        self.repo_copy_dir = os.path.join(self.sandbox_dir, "repo")
        if os.path.exists(self.repo_copy_dir):
            shutil.rmtree(self.repo_copy_dir)
        shutil.copytree(
            self.repo_path,
            self.repo_copy_dir,
            ignore=shutil.ignore_patterns("__pycache__", ".git", ".venv"),
        )
        for name in CODEGEN_SCRUB_NAMES:
            target = os.path.join(self.repo_copy_dir, name)
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            elif os.path.exists(target):
                os.remove(target)
        manifest_path = os.path.join(self.sandbox_dir, "fut_info.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["repo_root"] = self.repo_copy_dir
            self._write_sandbox_file("fut_info.json", json.dumps(manifest, indent=2))

    def _build_fut_infos(self, original_source: str) -> list:
        """解析原文件，定位每个被测函数的 1 起行号范围。"""
        tree = ast.parse(original_source)
        infos = []
        for full_name in self.funclass_names:
            if "." in full_name:
                class_name, method_name = full_name.rsplit(".", 1)
                node = self._find_method(tree, class_name, method_name)
                func_id = f"{self.file_path}:{full_name}"
                if node is not None:
                    infos.append(
                        {
                            "id": func_id,
                            "name": method_name,
                            "class_name": class_name,
                            "kind": "method",
                            "start_line": node.lineno,
                            "end_line": node.end_lineno,
                        }
                    )
            else:
                node = self._find_function(tree, full_name)
                func_id = f"{self.file_path}:{full_name}"
                if node is not None:
                    infos.append(
                        {
                            "id": func_id,
                            "name": full_name,
                            "class_name": None,
                            "kind": "function",
                            "start_line": node.lineno,
                            "end_line": node.end_lineno,
                        }
                    )
        return infos

    @staticmethod
    def _find_function(tree: ast.Module, name: str):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return node
        return None

    @staticmethod
    def _find_method(tree: ast.Module, class_name: str, method_name: str):
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == method_name:
                        return sub
        return None

    def _build_fut_module_source(self, original_source: str) -> str:
        """原文件源码原样前置（行号与原文件一致），末尾追加 reference_ 副本。"""
        tree = ast.parse(original_source)
        reference_nodes = []
        for full_name in self.funclass_names:
            if "." in full_name:
                class_name, _ = full_name.rsplit(".", 1)
                for node in tree.body:
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        ref_class = copy.deepcopy(node)
                        ref_class.name = f"reference_{class_name}"
                        reference_nodes.append(ref_class)
                        break
            else:
                for node in tree.body:
                    if (
                        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == full_name
                    ):
                        ref_func = copy.deepcopy(node)
                        ref_func.name = f"reference_{full_name}"
                        reference_nodes.append(ref_func)
                        break

        # 中性化相对导入：单行内替换文本，保持行号不变
        source_lines = original_source.split("\n")
        self._rewrite_relative_imports(tree, source_lines)
        front = "\n".join(source_lines)

        reference_module = ast.Module(body=reference_nodes, type_ignores=[])
        reference_source = ast.unparse(reference_module)
        return f"{front}\n\n\n{reference_source}\n"

    def _rewrite_relative_imports(self, tree: ast.Module, source_lines: list) -> None:
        """把 from .x import y 就地改写为绝对导入，只改首行的「from 模块」部分，行数不变。

        多行写法 from .x import (a,\\n b) 同样适用：括号与后续行保持原样。
        """
        repo_root = self.repo_path
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.level > 0):
                continue
            abs_module = self._absolute_module_name(node, repo_root)
            if abs_module is None:
                continue
            line_index = node.lineno - 1
            line = source_lines[line_index]
            head = line[node.col_offset :]
            match = RELATIVE_IMPORT_HEAD.match(head)
            if match is None:
                continue
            source_lines[line_index] = (
                line[: node.col_offset] + f"from {abs_module} import" + head[match.end() :]
            )

    def _absolute_module_name(self, node: ast.ImportFrom, repo_root: str) -> str | None:
        """按文件所在包位置计算相对导入的绝对模块名。"""
        source_file = os.path.join(self.repo_path, self.file_path)
        current_dir = os.path.dirname(source_file)
        package_parts = []
        while os.path.exists(os.path.join(current_dir, "__init__.py")):
            package_parts.insert(0, os.path.basename(current_dir))
            current_dir = os.path.dirname(current_dir)
        # 包根目录可以是仓库根，也可以是仓库下的 src/ 一类目录（该目录需在 PYTHONPATH 上）
        if not os.path.abspath(current_dir).startswith(os.path.abspath(repo_root)):
            return None
        base = package_parts[: max(0, len(package_parts) - (node.level - 1))]
        parts = base + ([node.module] if node.module else [])
        return ".".join(parts) if parts else None

    # ---------- codegen ----------

    def setup_codegen_mode(self) -> dict:
        try:
            self.codegen_mode = True
            self._copy_repo_workspace()
            return {"output": "", "error": ""}
        except Exception as e:
            return {"output": "", "error": repr(e)}

    @staticmethod
    def _candidate_method(candidates: dict, class_name: str, method_name: str):
        """候选可以给出整个类，也可以只给出该方法的 def。"""
        candidate = candidates.get(class_name)
        if isinstance(candidate, ast.ClassDef):
            for sub in candidate.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == method_name:
                    return sub
        candidate = candidates.get(method_name)
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return candidate
        return None

    def execute(self, code: str) -> dict:
        """用候选代码替换 fut_module 中的被测函数或被测方法（参考副本保持原样）。

        候选中的 import 语句插到模块开头；方法只替换类中的该方法，类的其余成员保留。
        """
        try:
            candidate_tree = ast.parse(code)
            candidates = {
                node.name: node
                for node in candidate_tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            candidate_imports = [
                node for node in candidate_tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            fut_module_path = os.path.join(self.sandbox_dir, "fut_module.py")
            with open(fut_module_path, "r", encoding="utf-8") as f:
                module_tree = ast.parse(f.read())

            function_targets = {}
            method_targets = {}
            for fut_info in self.fut_infos:
                if fut_info["class_name"]:
                    method = self._candidate_method(
                        candidates, fut_info["class_name"], fut_info["name"]
                    )
                    if method is None:
                        return {"output": "", "error": f"候选代码缺少被测方法: {fut_info['class_name']}.{fut_info['name']}"}
                    method_targets.setdefault(fut_info["class_name"], {})[fut_info["name"]] = method
                else:
                    candidate = candidates.get(fut_info["name"])
                    if not isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        return {"output": "", "error": f"候选代码缺少被测函数: {fut_info['name']}"}
                    function_targets[fut_info["name"]] = candidate

            replaced = []
            for node in module_tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in function_targets:
                    node.args = function_targets[node.name].args
                    node.body = function_targets[node.name].body
                    node.decorator_list = function_targets[node.name].decorator_list
                    node.returns = function_targets[node.name].returns
                    replaced.append(node.name)
                elif isinstance(node, ast.ClassDef) and node.name in method_targets:
                    for index, sub in enumerate(node.body):
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name in method_targets[node.name]:
                            node.body[index] = copy.deepcopy(method_targets[node.name][sub.name])
                            replaced.append(f"{node.name}.{sub.name}")

            module_tree.body = candidate_imports + module_tree.body
            with open(fut_module_path, "w", encoding="utf-8") as f:
                f.write(ast.unparse(module_tree) + "\n")
            return {"output": f"replaced: {sorted(replaced)}", "error": ""}
        except Exception as e:
            return {"output": "", "error": repr(e)}

    # ---------- submit ----------

    def _repo_python(self) -> str:
        """优先使用仓库虚拟环境的 python（build --local 所建），退回当前解释器。"""
        from codeharvest.extraction.local_setup import repo_venv_python

        return repo_venv_python(self.repo_path) or sys.executable

    def submit(self) -> dict:
        if not self.sandbox_dir:
            return {"output": "", "error": "init 未先调用", "logs": ""}

        run_tests_logs = {}
        run_tests_errors = {}
        coverage_logs = []
        captured_arg_logs = {}
        all_env_noise = []
        verdict = "passed"
        stderr_chunks = []

        python = self._repo_python()
        runner = os.path.join(PACKAGE_DIR, "run_one_test.py")

        for test_id in self.generated_tests:
            try:
                proc = subprocess.run(
                    [python, runner, self.sandbox_dir, str(test_id), self.repo_copy_dir],
                    cwd=self.sandbox_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.test_timeout,
                )
            except subprocess.TimeoutExpired:
                run_tests_logs[test_id] = self._invalid_log("timeout")
                run_tests_errors[test_id] = [
                    {
                        "type": "error",
                        "test": f"test_{test_id}",
                        "message": f"执行超时（{self.test_timeout}s）",
                        "line": None,
                    }
                ]
                verdict = "test_defect"
                continue

            if proc.stderr:
                stderr_chunks.append(proc.stderr)

            result = self._parse_result_line(proc.stdout)
            if result is None:
                run_tests_logs[test_id] = self._invalid_log("no_result")
                run_tests_errors[test_id] = [
                    {
                        "type": "error",
                        "test": f"test_{test_id}",
                        "message": f"子进程无结果输出: {proc.stderr[-500:]}",
                        "line": None,
                    }
                ]
                verdict = "test_defect"
                continue

            run_tests_logs[test_id] = result["run_test_log"]
            run_tests_errors[test_id] = result["run_test_errors"]
            if result["coverage_logs"]:
                coverage_logs = result["coverage_logs"]
            all_env_noise.extend(result.get("env_noise", []))
            for func_id, records in result.get("captured_arg_logs", {}).items():
                captured_arg_logs.setdefault(func_id, [])
                for record in records:
                    key = json.dumps(record.get("caller_info"), sort_keys=True)
                    captured_arg_logs[func_id] = [
                        r
                        for r in captured_arg_logs[func_id]
                        if json.dumps(r.get("caller_info"), sort_keys=True) != key
                    ]
                    captured_arg_logs[func_id].append(record)
                captured_arg_logs[func_id] = captured_arg_logs[func_id][-3:]
            if result.get("verdict") != "passed":
                verdict = "test_defect"

        logs = {
            "run_tests_logs": run_tests_logs,
            "run_tests_errors": run_tests_errors,
            "coverage_logs": coverage_logs,
            "captured_arg_logs": captured_arg_logs,
            "verdict": verdict,
            "env_noise": all_env_noise,
        }
        error = ""
        if verdict != "passed":
            error = "\n".join(
                e["message"]
                for errors in run_tests_errors.values()
                for e in errors[:1]
            )[:2000]
        return {
            "output": "",
            "error": error,
            "logs": json.dumps(logs),
        }

    @staticmethod
    def _invalid_log(reason: str) -> dict:
        return {
            "valid": False,
            "passed_names": [],
            "failed_names": [],
            "errored_names": [reason],
            "passed_count": 0,
            "failed_count": 0,
            "errored_count": 1,
        }

    @staticmethod
    def _parse_result_line(stdout: str):
        for line in reversed(stdout.split("\n")):
            line = line.strip()
            if line.startswith("CODEHARVEST_RESULT "):
                try:
                    return json.loads(line[len("CODEHARVEST_RESULT ") :])
                except json.JSONDecodeError:
                    return None
        return None

    def stop_server(self) -> None:
        if self.sandbox_dir and os.path.isdir(self.sandbox_dir):
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
            self.sandbox_dir = None

    # exposed_ 别名：rpyc 经网络调用走这些名字，本地直连走无前缀方法
    exposed_setup_repo = setup_repo
    exposed_setup_function = setup_function
    exposed_setup_test = setup_test
    exposed_init = init
    exposed_submit = submit
    exposed_setup_codegen_mode = setup_codegen_mode
    exposed_execute = execute
    exposed_stop_server = stop_server
