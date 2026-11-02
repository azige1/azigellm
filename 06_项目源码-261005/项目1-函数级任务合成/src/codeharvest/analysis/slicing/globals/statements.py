"""分析各类语句引用的全局名。"""

import ast

from codeharvest.analysis.slicing.globals.expressions import expr_global_names
from codeharvest.analysis.slicing.globals.bytecode import (
    definition_global_names,
)
from codeharvest.analysis.slicing.globals.annotations import (
    annotation_global_names,
)
from codeharvest.analysis.slicing.globals.defaults import (
    defaultarg_global_names,
)
from codeharvest.analysis.slicing.globals.decorators import (
    decorator_global_names,
)

def wrap_stmt_in_function(node: ast.stmt):
    node_unparse = ast.unparse(node)
    wrapped_func_code = "def fake_func():\n"
    for line in node_unparse.split("\n"):
        wrapped_func_code += "    " + line + "\n"
    wrapped_func_node = ast.parse(wrapped_func_code).body[0]
    return wrapped_func_node

class StmtGlobalsVisitor(ast.NodeVisitor):
    def visit(self, node) -> list[str]:
        if node is None:
            return []
        type_annotation_globals = annotation_global_names(node)
        visit_super = super().visit(node)
        return visit_super + type_annotation_globals

    def visit_FunctionDef(self, node: ast.FunctionDef) -> list[str]:
        return self.definition_handler(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> list[str]:
        return self.definition_handler(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> list[str]:
        globals = self.definition_handler(node)
        for base in node.bases:
            globals += expr_global_names(base)
        return globals

    def visit_Return(self, node: ast.Return) -> list[str]:
        return self.visit(node.value) if node.value else []

    def visit_Delete(self, node: ast.Delete) -> list[str]:
        globals: list[str] = []
        for target in node.targets:
            globals += expr_global_names(target)
        return globals

    def visit_Assign(self, node: ast.Assign) -> list[str]:
        globals = []
        for target in node.targets:
            globals += expr_global_names(target)
        globals += expr_global_names(node.value)
        return globals

    
    
    
    

    def visit_AugAssign(self, node: ast.AugAssign) -> list[str]:
        globals = expr_global_names(node.target)
        globals += expr_global_names(node.value)
        return globals

    def visit_AnnAssign(self, node: ast.AnnAssign) -> list[str]:
        globals = expr_global_names(node.annotation)
        if node.value:
            globals += expr_global_names(node.value)
        return globals

    def visit_For(self, node: ast.For) -> list[str]:
        return self.for_handler(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> list[str]:
        return self.for_handler(node)

    def visit_While(self, node: ast.While) -> list[str]:
        globals = expr_global_names(node.test)
        for body_stmt in node.body:
            globals += stmt_global_names(body_stmt)
        for if_condition in node.orelse:
            globals += stmt_global_names(if_condition)
        return globals

    def visit_If(self, node: ast.If) -> list[str]:
        globals = expr_global_names(node.test)
        for body_stmt in node.body:
            globals += stmt_global_names(body_stmt)
        for if_condition in node.orelse:
            globals += stmt_global_names(if_condition)
        return globals

    def visit_With(self, node: ast.With) -> list[str]:
        return self.with_handler(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> list[str]:
        return self.with_handler(node)

    def visit_Match(self, node: ast.Match) -> list[str]:
        globals: list[str] = []
        for case in node.cases:
            globals += pattern_global_names(case.pattern)
            if case.guard:
                globals += expr_global_names(case.guard)
            for stmt in case.body:
                globals += stmt_global_names(stmt)
        return globals

    def visit_Raise(self, node: ast.Raise) -> list[str]:
        globals = []
        if node.exc:
            globals += expr_global_names(node.exc)
        if node.cause:
            globals += expr_global_names(node.cause)
        return globals

    def visit_Try(self, node: ast.Try) -> list[str]:
        return self.try_handler(node)

    def visit_TryStar(self, node: ast.TryStar) -> list[str]:
        return self.try_handler(node)

    def visit_Assert(self, node: ast.Assert) -> list[str]:
        globals = expr_global_names(node.test)
        if node.msg:
            globals += expr_global_names(node.msg)
        return globals

    def visit_Import(self, node: ast.Import) -> list[str]:
        return []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> list[str]:
        return []

    def visit_Global(self, node: ast.Global) -> list[str]:
        return node.names

    def visit_Nonlocal(self, node: ast.Nonlocal) -> list[str]:
        return node.names

    def visit_Expr(self, node: ast.Expr) -> list[str]:
        return expr_global_names(node.value)

    def visit_Pass(self, node: ast.Pass) -> list[str]:
        return []

    def visit_Break(self, node: ast.Break) -> list[str]:
        return []

    def visit_Continue(self, node: ast.Continue) -> list[str]:
        return []

    def definition_handler(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> list[str]:
        all_globals: list[str] = []
        
        all_globals += defaultarg_global_names(node)
        
        all_globals += decorator_global_names(node)
        
        all_globals += definition_global_names(node)
        return all_globals

    def for_handler(self, node: ast.For | ast.AsyncFor) -> list[str]:
        return stmt_global_names(wrap_stmt_in_function(node))

    def with_handler(self, node: ast.With | ast.AsyncWith) -> list[str]:
        return stmt_global_names(wrap_stmt_in_function(node))

    def try_handler(self, node: ast.Try | ast.TryStar) -> list[str]:
        return stmt_global_names(wrap_stmt_in_function(node))

def stmt_global_names(ast_node: ast.stmt) -> list[str]:
    visitor = StmtGlobalsVisitor()
    return visitor.visit(ast_node)

def pattern_global_names(pattern: ast.pattern) -> list[str]:
    
    
    
    
    

    
    

    
    

    if isinstance(pattern, ast.MatchValue):
        return expr_global_names(pattern.value)
    elif isinstance(pattern, ast.MatchSingleton):
        return []
    elif isinstance(pattern, ast.MatchSequence):
        globals: list[str] = []
        for p in pattern.patterns:
            globals += pattern_global_names(p)
        return globals
    elif isinstance(pattern, ast.MatchMapping):
        globals: list[str] = []
        for key in pattern.keys:
            globals += expr_global_names(key)
        for p in pattern.patterns:
            globals += pattern_global_names(p)
        if pattern.rest:
            globals.append(pattern.rest)
        return globals
    elif isinstance(pattern, ast.MatchClass):
        globals = expr_global_names(pattern.cls)
        for p in pattern.patterns:
            globals += pattern_global_names(p)
        for kwd_attr in pattern.kwd_attrs:
            globals.append(kwd_attr)
        for kwd_pattern in pattern.kwd_patterns:
            globals += pattern_global_names(kwd_pattern)
        return globals
    elif isinstance(pattern, ast.MatchStar):
        return [pattern.name] if pattern.name else []
    elif isinstance(pattern, ast.MatchAs):
        globals: list[str] = []
        if pattern.pattern:
            globals += pattern_global_names(pattern.pattern)
        if pattern.name:
            globals.append(pattern.name)
        return globals
    elif isinstance(pattern, ast.MatchOr):
        globals: list[str] = []
        for p in pattern.patterns:
            globals += pattern_global_names(p)
        return globals
    return []

def excepthandler_global_names(handler: ast.ExceptHandler) -> list[str]:
    globals: list[str] = []
    non_globals: list[str] = []
    if handler.name:
        non_globals.append(handler.name)

    for stmt in handler.body:
        globals += stmt_global_names(stmt)
    globals = [g for g in globals if g not in non_globals]

    if handler.type:
        globals += expr_global_names(handler.type)

    return globals
