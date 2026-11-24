"""切片上下文符号完整性检查：找出被引用但未定义、未导入、非内置的名字。"""

import ast
import builtins
import symtable


def _module_defined_names(code: str) -> set[str]:
    """收集模块级定义的名字：赋值、函数 / 类定义、import 绑定、全局声明。"""
    defined = set(dir(builtins))
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return defined

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for name in _target_names(target):
                    defined.add(name)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            for name in _target_names(node.target):
                defined.add(name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                defined.add(alias.asname or alias.name)
    return defined


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for elt in target.elts:
            names.extend(_target_names(elt))
        return names
    return []


def _walk_scopes(table: symtable.SymbolTable, entry_names: set[str], missing: set[str]):
    """递归符号表，收集入口函数引用的全局 / 自由名字。"""
    for symbol in table.get_symbols():
        if symbol.is_global() or symbol.is_free():
            if not symbol.is_imported() and symbol.get_name() not in dir(builtins):
                missing.add(symbol.get_name())

    for child in table.get_children():
        _walk_scopes(child, entry_names, missing)


def find_missing_symbols(code: str, entry_names: list[str]) -> list[str]:
    """返回切片代码中入口函数引用但未解析的名字列表（升序）。

    code 为按文件拼接的切片上下文源码；entry_names 为被测函数 / 方法名。
    """
    try:
        top = symtable.symtable(code, "<slice>", "exec")
    except SyntaxError:
        return []

    defined = _module_defined_names(code)
    missing: set[str] = set()

    entry_set = set(entry_names)
    for child in top.get_children():
        if child.get_name() in entry_set:
            _walk_scopes(child, entry_set, missing)

    # 类方法的入口：遍历类作用域内的方法
    for child in top.get_children():
        if child.get_type() == "class":
            for grandchild in child.get_children():
                if grandchild.get_name() in entry_set:
                    _walk_scopes(grandchild, entry_set, missing)

    return sorted(name for name in missing if name not in defined)
