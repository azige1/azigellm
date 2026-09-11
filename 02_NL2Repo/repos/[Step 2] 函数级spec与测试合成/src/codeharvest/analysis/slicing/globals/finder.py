"""查找语句所依赖的全局符号入口。"""

import ast
import builtins

from codeharvest.analysis.slicing.globals.annotations import (
    annotation_global_names,
)
from codeharvest.analysis.slicing.globals.bytecode import (
    definition_global_names,
)

def wrap_stmt_in_function(node: ast.stmt) -> ast.AsyncFunctionDef:
    node_unparse = ast.unparse(node)
    wrapped_func_code = "async def fake_func():\n"
    for line in node_unparse.split("\n"):
        wrapped_func_code += "    " + line + "\n"
    wrapped_func_node = ast.parse(wrapped_func_code).body[0]
    return wrapped_func_node  

def find_referenced_globals(astnode: ast.stmt, unique: bool = True) -> list[str]:

    fake_function_node = wrap_stmt_in_function(astnode)

    all_type_annotation_globals = annotation_global_names(astnode)

    all_globals = definition_global_names(fake_function_node)

    all_globals.extend(all_type_annotation_globals)

    
    all_globals = [g for g in all_globals if g not in dir(builtins)]

    
    all_globals = [
        g for g in all_globals if not (g.startswith("__") and g.endswith("__"))
    ]

    if unique:
        return list(set(all_globals))

    return all_globals
