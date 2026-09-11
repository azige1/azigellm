"""类方法抽取器。"""

import ast

from codeharvest.analysis.ast_tools import parse_ast_file
from codeharvest.extraction.config import RepoJobConfig
from codeharvest.extraction.targets.base import FileCallableExtractor

class FileMethodCollector(FileCallableExtractor):

    @staticmethod
    def extract_methods_from_ast(
        astree: ast.Module, repo_args: RepoJobConfig
    ) -> list[ast.FunctionDef]:
        method_asts = FileMethodCollector.get_methods_from_ast(astree)

        if repo_args.disable_all_filters:
            return method_asts

        
        if not repo_args.disable_dunder_methods:
            method_asts = FileMethodCollector.filter_dunder_methods(method_asts)

        
        if not repo_args.disable_no_docstring:
            method_asts = FileMethodCollector.filter_keep_docstring(method_asts)

        
        if not repo_args.disable_signature_filters:
            method_asts = FileMethodCollector.filter_literal_returns(method_asts)

        
        if not repo_args.disable_keyword_filters:
            method_asts = FileMethodCollector.filter_docstring_keywords(method_asts)
            method_asts = FileMethodCollector.filter_func_body_keywords(method_asts)
            method_asts = FileMethodCollector.filter_bad_function_names(method_asts)

        
        if not repo_args.disable_wrapper_filters:
            method_asts = FileMethodCollector.filter_with_decorator(
                method_asts, allowed_decorators=["staticmethod", "classmethod"]
            )
            method_asts = FileMethodCollector.filter_wrapper_methods(method_asts)

        return method_asts

    @staticmethod
    def get_methods_from_ast(astree: ast.Module) -> list[ast.FunctionDef]:
        methods: list[ast.FunctionDef] = []
        for node in astree.body:
            if isinstance(node, ast.ClassDef):
                for subnode in node.body:
                    if isinstance(subnode, ast.FunctionDef):
                        methods.append(subnode)
        return methods

    @staticmethod
    def filter_wrapper_methods(
        method_asts: list[ast.FunctionDef],
    ) -> list[ast.FunctionDef]:
        return [
            method_ast
            for method_ast in method_asts
            if not FileMethodCollector.is_wrapper_method(method_ast)
        ]

    @staticmethod
    def is_wrapper_method(method_ast: ast.FunctionDef) -> bool:
        if len(method_ast.body) == 1:
            return True
        if len(method_ast.body) == 2:
            if "self." in ast.unparse(method_ast.body[1]):
                return True
        if len(method_ast.body) == 3:
            if "self." in ast.unparse(method_ast.body[1]):
                if "return" in ast.unparse(method_ast.body[2]):
                    return True
        return False
