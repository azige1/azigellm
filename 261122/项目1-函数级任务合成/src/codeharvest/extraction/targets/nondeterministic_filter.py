"""不可对照函数过滤：剔除运行时行为不确定、无法与参考实现等价对照的函数。"""

import ast

# 命中即剔除的「模块.属性」调用链前缀
NONDETERMINISTIC_CALL_PREFIXES = (
    "time.time",
    "time.monotonic",
    "time.perf_counter",
    "datetime.now",
    "datetime.datetime.now",
    "datetime.utcnow",
    "datetime.date.today",
    "uuid.uuid4",
    "uuid.uuid1",
    "os.urandom",
    "secrets.token",
)

# 网络相关库前缀
NETWORK_CALL_PREFIXES = (
    "requests.",
    "urllib.",
    "httpx.",
    "socket.",
    "http.client.",
    "aiohttp.",
    "ftplib.",
    "smtplib.",
)

MAX_CALLEE_DEPTH = 3


def _call_chain(node: ast.Call) -> str:
    """把调用节点还原成 「模块.属性」 形式的链。"""
    func = node.func
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def _collect_calls(root: ast.AST) -> list[str]:
    return [_call_chain(node) for node in ast.walk(root) if isinstance(node, ast.Call)]


def _function_has_nondeterminism(function_node: ast.AST, body_text: str) -> bool:
    calls = _collect_calls(function_node)
    for call in calls:
        for prefix in NONDETERMINISTIC_CALL_PREFIXES:
            if call == prefix or call.startswith(prefix + "(") or call.startswith(prefix):
                return True
        for prefix in NETWORK_CALL_PREFIXES:
            if call.startswith(prefix):
                return True

    # random.* 未设种子视为不确定
    uses_random = any(call.startswith("random.") for call in calls)
    if uses_random and "random.seed" not in body_text:
        return True
    return False


class NondeterministicFilter:
    """静态判定函数（含同文件内至多 MAX_CALLEE_DEPTH 层被调函数）是否不确定。"""

    @staticmethod
    def filter_functions(
        function_asts: list[ast.FunctionDef],
        module_ast: ast.Module,
    ) -> list[ast.FunctionDef]:
        module_functions = {
            node.name: node
            for node in module_ast.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        return [
            function_ast
            for function_ast in function_asts
            if not NondeterministicFilter.is_nondeterministic(
                function_ast, module_functions
            )
        ]

    @staticmethod
    def is_nondeterministic(
        function_node: ast.AST,
        module_functions: dict[str, ast.FunctionDef],
        depth: int = 0,
        visited: set[str] | None = None,
    ) -> bool:
        body_text = ast.unparse(function_node)
        if _function_has_nondeterminism(function_node, body_text):
            return True
        if depth >= MAX_CALLEE_DEPTH:
            return False

        visited = visited if visited is not None else set()
        for call in _collect_calls(function_node):
            callee_name = call.split(".")[0].split("(")[0]
            if callee_name in visited:
                continue
            callee = module_functions.get(callee_name)
            if callee is None:
                continue
            visited.add(callee_name)
            if NondeterministicFilter.is_nondeterministic(
                callee, module_functions, depth + 1, visited
            ):
                return True
        return False
