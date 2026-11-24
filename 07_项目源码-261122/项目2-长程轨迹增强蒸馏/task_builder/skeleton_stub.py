"""骨架提交生成：用 AST 把函数与方法体替换为 docstring + raise NotImplementedError。

类属性、常量、import、``__all__`` 与类型别名保持不变；
只有 import 的 ``__init__.py`` 天然无需改动。
"""

import ast
from pathlib import Path


def _stub_body(tree: ast.AST) -> ast.AST:
    """把树中所有函数/方法的函数体替换为 docstring + NotImplementedError。"""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            new_body: list[ast.stmt] = []
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                new_body.append(node.body[0])
            new_body.append(
                ast.Raise(
                    exc=ast.Call(func=ast.Name(id="NotImplementedError", ctx=ast.Load()), args=[], keywords=[]),
                    cause=None,
                )
            )
            node.body = new_body
    return tree


def stub_source(source: str) -> str:
    """返回骨架化后的源码文本。"""
    tree = ast.parse(source)
    tree = _stub_body(tree)
    return ast.unparse(tree) + "\n"


def build_skeleton_commit(repo_dir: Path, src_dir: str) -> None:
    """把源码目录下的每个 .py 文件原地骨架化（删除测试目录由调用方负责）。"""
    src = repo_dir / src_dir
    for py_file in sorted(src.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        if "def " not in source and "async def " not in source:
            continue
        py_file.write_text(stub_source(source), encoding="utf-8")


__all__ = []
