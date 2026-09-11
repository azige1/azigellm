"""依赖切片器主流程。"""

import ast
from typing import Type

from codeharvest.core import RepoRecord, FileRecord, FunctionRecord, ClassRecord
from codeharvest.analysis.callgraph import CallGraphQuery
from codeharvest.analysis.slicing.graph import StatementDependencyGraph
from codeharvest.analysis.slicing.statements import StatementNode, StatementTable
from codeharvest.analysis.slicing.handlers import (
    StatementHandler,
    DefinitionStatementHandler,
    ImportStatementHandler,
)

HandlersMapping: dict[Type[ast.AST], Type[StatementHandler]] = {
    ast.ClassDef: DefinitionStatementHandler,
    ast.FunctionDef: DefinitionStatementHandler,
    ast.AsyncFunctionDef: DefinitionStatementHandler,
    ast.Import: ImportStatementHandler,
    ast.ImportFrom: ImportStatementHandler,
}

class CodeDependencySlicer:

    def __init__(
        self,
        repo: RepoRecord,
        ast_stmt_list: list[StatementNode],
        file_ast_cache: dict[str, StatementTable],
        depth: int = -1,
        slice_imports: bool = True,
    ):
        self.repo = repo
        self.ast_stmt_list = ast_stmt_list
        self.file_ast_cache = file_ast_cache
        try:
            self.callgraph_explorer = CallGraphQuery(self.repo)
        except Exception as e:
            self.callgraph_explorer = None

        self.recursion_stack: list[StatementNode] = []
        self.visited_set: set[StatementNode] = set()

        self.dependency_graph = StatementDependencyGraph(self.ast_stmt_list)

        self.depth = depth

        self.slice_imports = slice_imports

    @classmethod
    def from_function_models(
        cls,
        function_models: FunctionRecord | list[FunctionRecord],
        depth: int = -1,
        slice_imports: bool = True,
    ):
        function_models = (
            function_models if isinstance(function_models, list) else [function_models]
        )
        assert (
            len(set([f.repo for f in function_models])) == 1
        ), f"{[f.repo for f in function_models]} are not the same repos"

        repo = function_models[0].repo

        file_ast_cache: dict[str, StatementTable] = {}

        ast_stmt_list: list[StatementNode] = []
        for function in function_models:
            assert function.function_name is not None
            if function.file_path in file_ast_cache:
                ast_stmts = file_ast_cache[function.file_path]
            else:
                ast_stmts = StatementTable(function.file)
                file_ast_cache[function.file_path] = ast_stmts

            resolved_function = ast_stmts.find_function_stmt_with_name(
                function.function_name
            )
            assert resolved_function is not None
            ast_stmt_list.append(resolved_function)

        return cls(repo, ast_stmt_list, file_ast_cache, depth, slice_imports)

    @classmethod
    def from_class_models(
        cls,
        class_models: ClassRecord | list[ClassRecord],
        depth: int = -1,
        slice_imports: bool = True,
    ):
        class_models = (
            class_models if isinstance(class_models, list) else [class_models]
        )
        assert (
            len(set([c.repo for c in class_models])) == 1
        ), f"{[c.repo for c in class_models]} are not the same repos"

        repo = class_models[0].repo

        file_ast_cache: dict[str, StatementTable] = {}

        ast_stmt_list: list[StatementNode] = []
        for class_model in class_models:
            assert class_model.class_name is not None
            if class_model.file_path in file_ast_cache:
                ast_stmts = file_ast_cache[class_model.file_path]
            else:
                ast_stmts = StatementTable(class_model.file)
                file_ast_cache[class_model.file_path] = ast_stmts

            resolved_class = ast_stmts.find_class_stmt_with_name(class_model.class_name)
            assert resolved_class is not None
            ast_stmt_list.append(resolved_class)

        return cls(repo, ast_stmt_list, file_ast_cache, depth, slice_imports)

    @classmethod
    def from_funclass_models(
        cls,
        funclass_models: list[FunctionRecord | ClassRecord],
        depth: int = -1,
        slice_imports: bool = True,
    ):
        funclass_models = (
            funclass_models if isinstance(funclass_models, list) else [funclass_models]
        )
        assert (
            len(set([f.repo for f in funclass_models])) == 1
        ), f"{[f.repo for f in funclass_models]} are not the same repos"

        repo = funclass_models[0].repo

        file_ast_cache: dict[str, StatementTable] = {}

        ast_stmt_list: list[StatementNode] = []
        for funclass_model in funclass_models:
            if isinstance(funclass_model, FunctionRecord):
                assert funclass_model.function_name is not None
                if funclass_model.file_path in file_ast_cache:
                    ast_stmts = file_ast_cache[funclass_model.file_path]
                else:
                    ast_stmts = StatementTable(funclass_model.file)
                    file_ast_cache[funclass_model.file_path] = ast_stmts

                resolved_function = ast_stmts.find_function_stmt_with_name(
                    funclass_model.function_name
                )
                assert (
                    resolved_function is not None
                ), f"{funclass_model.function_name} {funclass_model.file_path}"
                ast_stmt_list.append(resolved_function)
            elif isinstance(funclass_model, ClassRecord):
                assert funclass_model.class_name is not None
                if funclass_model.file_path in file_ast_cache:
                    ast_stmts = file_ast_cache[funclass_model.file_path]
                else:
                    ast_stmts = StatementTable(funclass_model.file)
                    file_ast_cache[funclass_model.file_path] = ast_stmts

                resolved_class = ast_stmts.find_class_stmt_with_name(
                    funclass_model.class_name
                )
                assert (
                    resolved_class is not None
                ), f"{funclass_model.class_name} {funclass_model.file_path}"
                ast_stmt_list.append(resolved_class)

        return cls(repo, ast_stmt_list, file_ast_cache, depth, slice_imports)

    def run(self):
        for ast_stmt in self.ast_stmt_list:
            self.visit(
                ast_stmt, self.file_ast_cache[ast_stmt.file_path], depth=self.depth
            )

    def visit(
        self,
        stmt: StatementNode,
        all_stmts: StatementTable,
        search_key: str = "",
        depth: int = -1,
    ):
        if depth == 0:
            return

        if stmt in self.visited_set or stmt in self.recursion_stack:
            return

        

        for ast_type, handler in HandlersMapping.items():
            if isinstance(stmt.stmt, ast_type):
                if (not self.slice_imports) and issubclass(handler, ImportStatementHandler):
                    return
                handler_instance = handler(stmt, all_stmts, search_key, self, depth)
                handler_instance.handle()
                return
        StatementHandler(stmt, all_stmts, search_key, self, depth).handle()
        return

    def get_file_ast_stmts(self, file_path: str) -> StatementTable:
        if file_path in self.file_ast_cache:
            return self.file_ast_cache[file_path]

        file_obj = FileRecord.from_file_path(file_path, self.repo)
        ast_stmts = StatementTable(file_obj)
        self.file_ast_cache[file_path] = ast_stmts
        return ast_stmts
