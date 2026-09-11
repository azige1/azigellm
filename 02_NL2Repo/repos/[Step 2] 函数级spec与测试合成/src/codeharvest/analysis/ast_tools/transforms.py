"""一组 AST 变换器：别名替换、删除函数 / 类 / 方法等。"""

import ast

class SyntaxTreeTransformer(ast.NodeTransformer):

    def __init__(self, tree: ast.Module):
        self.tree = tree

    def transform(self):
        return super().visit(self.tree)

class AliasNameReplacer(SyntaxTreeTransformer):

    def __init__(self, tree: ast.Module, names: list[str]):
        super().__init__(tree)
        self.aliases = self.get_all_aliases(names)

    def get_all_aliases(self, names: list[str]) -> dict:
        aliases = {}
        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:

                    
                    if alias.asname and alias.asname not in names:

                        
                        if alias.name in names:
                            aliases[alias.asname] = alias.name

        return aliases

    def visit_Name(self, node):
        if node.id in self.aliases:
            return ast.copy_location(
                ast.Name(id=self.aliases[node.id], ctx=node.ctx), node
            )
        return node

class FunctionRemover(SyntaxTreeTransformer):

    def __init__(self, tree: ast.Module, function_name: str):
        super().__init__(tree)
        self.function_name = function_name
        self.removed_node = None

    def visit_FunctionDef(self, node):
        if node.name == self.function_name:
            
            self.removed_node = node
            return None
        return node

class ClassRemover(SyntaxTreeTransformer):

    def __init__(self, tree: ast.Module, class_name: str):
        super().__init__(tree)
        self.class_name = class_name
        self.removed_node = None

    def visit_ClassDef(self, node):
        if node.name == self.class_name:
            self.removed_node = node
            return None
        return node

class MethodToClassEndMover(SyntaxTreeTransformer):

    def __init__(self, tree: ast.Module, class_name: str, method_name: str):
        super().__init__(tree)
        self.class_name = class_name
        self.method_name = method_name

    def visit_ClassDef(self, node):
        if node.name == self.class_name:
            method_to_move = None
            for method in node.body:
                if (
                    isinstance(method, ast.FunctionDef)
                    and method.name == self.method_name
                ):
                    method_to_move = method
                    break
            if method_to_move:
                node.body.remove(method_to_move)
                node.body.append(method_to_move)
        return node

class LastNodeRemover(SyntaxTreeTransformer):

    def __init__(self, tree: ast.Module):
        super().__init__(tree)

    def visit_Module(self, node):

        if isinstance(node.body[-1], ast.ClassDef):
            if len(node.body[-1].body) == 1:
                node.body.pop()
            else:
                node.body[-1].body.pop()
        else:
            node.body.pop()

        return node

class MethodsRemover(SyntaxTreeTransformer):

    def __init__(self, tree: ast.Module, method_names: list[str]):
        self.method_names = method_names
        super().__init__(tree)

    def visit_ClassDef(self, node):
        node.body = [
            method
            for method in node.body
            if not (
                isinstance(method, ast.FunctionDef) and method.name in self.method_names
            )
        ]
        return node
