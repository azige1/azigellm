"""分析类型注解引用的全局名。"""

import ast

from codeharvest.analysis.slicing.globals.expressions import (
    ExprGlobalsVisitor,
)

def gather_type_annotations(astnode: ast.AST) -> list[ast.AST]:
    type_annotations: list[ast.AST] = []

    for node in ast.walk(astnode):
        if isinstance(node, ast.AnnAssign):
            type_annotations.append(node.annotation)
        if isinstance(node, ast.arg):
            if node.annotation:
                type_annotations.append(node.annotation)
        if isinstance(node, ast.FunctionDef):
            if node.returns:
                type_annotations.append(node.returns)
        if isinstance(node, ast.ClassDef):
            if node.bases:
                type_annotations.extend(node.bases)
    return type_annotations

class AnnotationGlobalsVisitor(ExprGlobalsVisitor):
    def visit_Constant(self, node) -> list[str]:
        if isinstance(node.value, str):
            try:
                node_parsed = ast.parse(node.value).body
            except SyntaxError:
                return []
            assert len(node_parsed) == 1 and isinstance(node_parsed[0], ast.Expr)
            node_parsed = node_parsed[0].value
            if isinstance(node_parsed, ast.Name):
                return [node.value]
            return []
        if node.value is None:
            return ["None"]
        if node.value == Ellipsis:
            return []
        
        return []

    def visit_Subscript(self, node) -> list[str]:
        globals = self.visit(node.value)
        if ast.unparse(node.value) != "Literal":
            globals.extend(self.visit(node.slice))
        return globals

def annotations_to_global_names(type_annotations: list[ast.AST]) -> list[str]:

    global_vars = []
    for type_ann in type_annotations:
        global_vars.extend(AnnotationGlobalsVisitor().visit(type_ann))
    return global_vars

def annotation_global_names(
    astnode: ast.AST,
) -> list[str]:
    type_annotations = gather_type_annotations(astnode)
    return annotations_to_global_names(type_annotations)
