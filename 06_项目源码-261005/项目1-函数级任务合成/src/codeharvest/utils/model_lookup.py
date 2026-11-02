"""按标识符或路径在数据模型中查找模块 / 类型。"""

import os
import ast
from typing import Any
import typing_extensions

from codeharvest.core.identifiers import SymbolId
from codeharvest.core.code_module import ModuleRecord
from codeharvest.core.repository import RepoRecord
from codeharvest.core.callgraph import CodeElementKind
from codeharvest.analysis.ast_tools.finder import parse_ast_file
from codeharvest.workspace import WORKSPACE_DIR

IncEx: typing_extensions.TypeAlias = (
    "set[int] | set[str] | dict[int, Any] | dict[str, Any] | None"
)

def find_module_by_identifier(identifier: SymbolId, repo: RepoRecord) -> ModuleRecord:
    func_or_class_module = make_callable_module(identifier, repo)
    meth_module = make_method_module(identifier, repo)

    if func_or_class_module.exists():
        return func_or_class_module
    elif meth_module.exists():
        return meth_module
    else:
        raise ValueError(f"Could not find module for: {identifier}")

def find_module_by_path(local_path: str, repo: RepoRecord) -> ModuleRecord:
    relative_path = local_path.split(f"{repo.local_repo_path}/")[1]
    module_id = SymbolId.from_relative_path(relative_path)
    module = ModuleRecord(module_id=module_id, repo=repo)
    return module

def resolve_element_kind(identifier: SymbolId, repo: RepoRecord) -> CodeElementKind:

    if identifier.identifier.startswith("<builtin>"):
        return CodeElementKind.BUILTIN

    func_or_class_module = make_callable_module(identifier, repo)
    meth_module = make_method_module(identifier, repo)

    if func_or_class_module.exists():
        file_ast = parse_ast_file(func_or_class_module.local_path)
        code_elem_name = identifier.identifier.split(".")[-1]

        for node in file_ast.body:  
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == code_elem_name
            ):
                return CodeElementKind.FUNCTION
            elif isinstance(node, ast.ClassDef) and node.name == code_elem_name:
                return CodeElementKind.CLASS

        return CodeElementKind.OTHER

    elif meth_module.exists():
        file_ast = parse_ast_file(meth_module.local_path)
        code_elem_name = identifier.identifier.split(".")[-1]
        code_elem_parent = identifier.identifier.split(".")[-2]

        for node in file_ast.body:  
            if isinstance(node, ast.ClassDef) and node.name == code_elem_parent:
                for n in node.body:
                    if (
                        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and n.name == code_elem_name
                    ):
                        return CodeElementKind.METHOD

        return CodeElementKind.OTHER

    else:
        first_part = identifier.identifier.split(".")[0]
        paths_in_repo = []
        for root, dirs, files in os.walk(f"{WORKSPACE_DIR}/{repo.local_repo_path}/"):
            for file in files:
                if file.endswith(".py"):
                    paths_in_repo.append(os.path.join(root, file))

        if not any(first_part in path for path in paths_in_repo):
            return CodeElementKind.API

        return CodeElementKind.OTHER

def make_callable_module(identifier: SymbolId, repo: RepoRecord) -> ModuleRecord:
    mod_from_func_class = lambda item: ".".join(item.split(".")[:-1])
    func_module_id = SymbolId(identifier=mod_from_func_class(identifier.identifier))
    func_module = ModuleRecord(module_id=func_module_id, repo=repo)

    return func_module

def make_method_module(identifier: SymbolId, repo: RepoRecord) -> ModuleRecord:
    mod_from_method = lambda item: ".".join(item.split(".")[:-2])
    meth_module_id = SymbolId(identifier=mod_from_method(identifier.identifier))
    meth_module = ModuleRecord(module_id=meth_module_id, repo=repo)

    return meth_module

def extend_excluded_fields(exclude: IncEx, fields_to_exclude: set[str]) -> IncEx:
    if exclude is None:
        exclude = fields_to_exclude
    elif isinstance(exclude, set):
        exclude |= fields_to_exclude  
    elif isinstance(exclude, dict):
        for field in fields_to_exclude:
            exclude[field] = ...  
    else:
        raise ValueError("exclude must be None, a set, or a dict")
    return exclude
