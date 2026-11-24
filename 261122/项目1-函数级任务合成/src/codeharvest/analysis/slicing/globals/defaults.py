"""分析默认参数表达式引用的全局名。"""

import ast

from codeharvest.analysis.slicing.globals.expressions import expr_global_names

def gather_arg_defaults(
    astnode: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> list[ast.expr]:
    all_default_asts: list[ast.expr] = []
    for node in ast.walk(astnode):
        if isinstance(node, ast.arguments):
            all_default_asts.extend(node.defaults)
            all_default_asts.extend([expr for expr in node.kw_defaults if expr])
    return all_default_asts

def defaults_to_global_names(default_asts: list[ast.expr]) -> list[str]:

    global_vars = []
    for default_ast in default_asts:
        global_vars.extend(expr_global_names(default_ast))
    return global_vars

def defaultarg_global_names(
    astnode: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> list[str]:
    default_asts = gather_arg_defaults(astnode)
    return defaults_to_global_names(default_asts)
