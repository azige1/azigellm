"""副作用静态判定：命中任一规则的函数不挂采集钩子。"""

import ast

# 写模式的 open 标志
READ_MODES = {"r", "rb", "r+", "rb+", "U", "rU"}

# 副作用调用链前缀
SIDE_EFFECT_CALL_PREFIXES = (
    "os.remove",
    "os.rename",
    "os.rmdir",
    "os.makedirs",
    "os.mkdir",
    "os.replace",
    "shutil.",
    "subprocess.",
    "socket.",
)

# 网络库前缀
NETWORK_PREFIXES = (
    "requests.",
    "urllib.",
    "httpx.",
    "socket.",
)

# 名字前缀即判定有副作用
SIDE_EFFECT_NAME_PREFIXES = (
    "write",
    "save",
    "delete",
    "remove",
    "send",
    "upload",
    "post",
    "put",
)

MAX_CALLEE_DEPTH = 2


def _call_chain(node: ast.Call) -> str:
    func = node.func
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def _open_has_write_mode(node: ast.Call) -> bool:
    """按 mode 参数判定 open 是否写文件；未给 mode 时为只读，mode 不是字面量时保守视为写。"""
    mode_node = node.args[1] if len(node.args) >= 2 else None
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode_node = keyword.value
    if mode_node is None:
        return False
    if not isinstance(mode_node, ast.Constant):
        return True
    return str(mode_node.value) not in READ_MODES


def _has_global_statement(root: ast.AST) -> bool:
    for node in ast.walk(root):
        if isinstance(node, ast.Global):
            return True
    return False


def _module_level_names(module_ast: ast.Module | None) -> set[str]:
    """模块顶层赋值绑定的名字（可变全局状态的候选）。"""
    names = set()
    if module_ast is None:
        return names
    for node in module_ast.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
    return names


def _local_names(function_node: ast.AST) -> set[str]:
    """函数参数与函数体内以名字形式绑定的局部变量。"""
    names = set()
    args = getattr(function_node, "args", None)
    if args is not None:
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            names.add(arg.arg)
        if args.vararg:
            names.add(args.vararg.arg)
        if args.kwarg:
            names.add(args.kwarg.arg)
    for node in ast.walk(function_node):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def _base_name(target: ast.AST) -> str | None:
    """取下标或属性写入目标最左侧的名字，如 CACHE[k].x -> CACHE。"""
    while isinstance(target, (ast.Subscript, ast.Attribute)):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


def _assigns_module_level_mutable(root: ast.AST, module_ast: ast.Module | None = None) -> bool:
    """函数体内对模块级可变对象的下标或属性写入（如 CACHE[k] = v）视为有副作用。

    局部变量赋值与对参数、self 的写入不算；对模块级名字的直接重新绑定需要 global 声明，
    由 _has_global_statement 单独判定。
    """
    module_names = _module_level_names(module_ast)
    if not module_names:
        return False
    local_names = _local_names(root)
    for node in ast.walk(root):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = node.targets
        for target in targets:
            if isinstance(target, (ast.Subscript, ast.Attribute)):
                base = _base_name(target)
                if base in module_names and base not in local_names:
                    return True
    return False


class SideEffectFilter:
    """判定函数体（含同文件内至多 MAX_CALLEE_DEPTH 层被调函数）是否有副作用。"""

    @staticmethod
    def has_side_effects(
        function_node: ast.AST,
        module_functions: dict[str, ast.FunctionDef] | None = None,
        module_ast: ast.Module | None = None,
        depth: int = 0,
        visited: set[str] | None = None,
    ) -> bool:
        function_name = getattr(function_node, "name", "")
        if function_name.lower().startswith(SIDE_EFFECT_NAME_PREFIXES):
            return True

        calls = [
            _call_chain(node) for node in ast.walk(function_node) if isinstance(node, ast.Call)
        ]
        for chain in calls:
            if chain == "open" or chain.endswith(".open"):
                call_nodes = [
                    n for n in ast.walk(function_node) if isinstance(n, ast.Call) and _call_chain(n) == chain
                ]
                if any(_open_has_write_mode(n) for n in call_nodes):
                    return True
            for prefix in SIDE_EFFECT_CALL_PREFIXES + NETWORK_PREFIXES:
                if chain.startswith(prefix):
                    return True

        if _has_global_statement(function_node):
            return True
        if _assigns_module_level_mutable(function_node, module_ast):
            return True

        if depth >= MAX_CALLEE_DEPTH or not module_functions:
            return False

        visited = visited if visited is not None else set()
        for chain in calls:
            callee_name = chain.split(".")[0].split("(")[0]
            if callee_name in visited:
                continue
            callee = module_functions.get(callee_name)
            if callee is None:
                continue
            visited.add(callee_name)
            if SideEffectFilter.has_side_effects(
                callee, module_functions, module_ast, depth + 1, visited
            ):
                return True
        return False

    @staticmethod
    def module_function_table(module_ast: ast.Module) -> dict[str, ast.FunctionDef]:
        return {
            node.name: node
            for node in module_ast.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
