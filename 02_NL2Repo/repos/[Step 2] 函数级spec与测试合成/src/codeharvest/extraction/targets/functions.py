"""顶层函数抽取器。"""

import ast

from codeharvest.extraction.config import RepoJobConfig
from codeharvest.extraction.targets.base import FileCallableExtractor

class FileFunctionCollector(FileCallableExtractor):

    @staticmethod
    def extract_functions_from_ast(
        astree: ast.Module, repo_args: RepoJobConfig
    ) -> list[ast.FunctionDef]:
        function_asts = FileFunctionCollector.get_functions_from_ast(astree)

        if repo_args.disable_all_filters:
            return function_asts

        
        if not repo_args.disable_dunder_methods:
            function_asts = FileFunctionCollector.filter_dunder_methods(function_asts)

        
        if not repo_args.disable_no_docstring:
            function_asts = FileFunctionCollector.filter_keep_docstring(function_asts)

        
        if not repo_args.disable_signature_filters:
            function_asts = FileFunctionCollector.filter_nonzero_arguments(
                function_asts
            )
            function_asts = FileFunctionCollector.filter_nonzero_returns(function_asts)
            function_asts = FileFunctionCollector.filter_literal_returns(function_asts)

        
        if not repo_args.disable_keyword_filters:
            function_asts = FileFunctionCollector.filter_docstring_keywords(
                function_asts
            )
            function_asts = FileFunctionCollector.filter_func_body_keywords(
                function_asts
            )
            function_asts = FileFunctionCollector.filter_bad_function_names(
                function_asts
            )

        
        if not repo_args.disable_wrapper_filters:
            function_asts = FileFunctionCollector.filter_with_decorator(function_asts)
            function_asts = FileFunctionCollector.filter_onlt_one_stmt(function_asts)

        return function_asts

    @staticmethod
    def get_functions_from_ast(astree: ast.Module) -> list[ast.FunctionDef]:
        functions: list[ast.FunctionDef] = []
        for node in astree.body:
            if isinstance(node, ast.FunctionDef):
                functions.append(node)
        return functions

    @staticmethod
    def filter_nonzero_arguments(
        function_asts: list[ast.FunctionDef],
    ) -> list[ast.FunctionDef]:
        return [
            function_ast
            for function_ast in function_asts
            if FileFunctionCollector.get_num_args(function_ast) > 0
        ]

    @staticmethod
    def get_num_args(function_ast: ast.FunctionDef) -> int:
        return (
            len(function_ast.args.args)
            + len(function_ast.args.kwonlyargs)
            + len(function_ast.args.posonlyargs)
            + (0 if function_ast.args.kwarg is None else 1)
            + (0 if function_ast.args.vararg is None else 1)
        )

    @staticmethod
    def filter_nonzero_returns(
        function_asts: list[ast.FunctionDef],
    ) -> list[ast.FunctionDef]:
        return [
            function_ast
            for function_ast in function_asts
            if any(isinstance(node, ast.Return) for node in function_ast.body)
        ]

    @staticmethod
    def filter_onlt_one_stmt(
        function_asts: list[ast.FunctionDef],
    ) -> list[ast.FunctionDef]:
        
        return [
            function_ast for function_ast in function_asts if len(function_ast.body) > 2
        ]
