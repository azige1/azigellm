"""AST 解析与定义查找工具。"""

import ast
from typing import Optional

from codeharvest.analysis.ast_tools.parents import attach_parent_links

def parse_ast(code: str, add_parents: bool = True) -> ast.Module:
    astree = ast.parse(code)
    if add_parents:
        astree = attach_parent_links(astree)
    return astree

def parse_ast_file(file_path: str, add_parents: bool = True) -> ast.Module:
    with open(file_path, "r") as f:
        code = f.read()
    return parse_ast(code, add_parents=add_parents)

def locate_def_in_ast(
    astree: ast.AST, name: str, def_type: Optional[type | tuple[type, ...]] = None
) -> Optional[ast.AST]:

    definition_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    if def_type is not None:
        definition_types = def_type if isinstance(def_type, tuple) else (def_type,)

    for node in ast.walk(astree):
        if isinstance(
            node,
            definition_types,
        ):
            node_name = getattr(node, "name", None)
            if node_name == name:
                return node

    return None

def locate_function_in_ast(
    astree: ast.AST, name: str
) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    func_def_types = (ast.FunctionDef, ast.AsyncFunctionDef)
    found_node = locate_def_in_ast(astree, name, func_def_types)
    return found_node if isinstance(found_node, func_def_types) else None

def collect_body_imports(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> list[ast.Import | ast.ImportFrom]:
    all_imports: list[ast.Import | ast.ImportFrom] = []
    for stmt in node.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            all_imports.append(stmt)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            all_imports += collect_body_imports(stmt)
    return all_imports
