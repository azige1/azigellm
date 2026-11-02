"""分析装饰器表达式引用的全局名。"""

import ast

from codeharvest.analysis.slicing.globals.expressions import expr_global_names

def decorator_global_names(
    astnode: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> list[str]:
    all_decorator_globals = []
    for node in ast.walk(astnode):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                unparsed_decorator = ast.unparse(decorator)
                if ".setter" in unparsed_decorator:
                    continue
                if ".getter" in unparsed_decorator:
                    continue
                if ".deleter" in unparsed_decorator:
                    continue
                if ".wraps" in unparsed_decorator:
                    continue
                all_decorator_globals.extend(expr_global_names(decorator))
    return all_decorator_globals
