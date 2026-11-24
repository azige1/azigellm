"""类 / 函数定义语句的处理器。"""

import ast
import os
import logging
from typing import TYPE_CHECKING

from codeharvest.logging_utils import slicer_logger
from codeharvest.core import FunctionRecord, ClassRecord, MethodRecord
from codeharvest.analysis.ast_tools.finder import collect_body_imports
from codeharvest.analysis.imports.resolver import ImportPathResolver
from codeharvest.analysis.slicing.statements import StatementTable
from codeharvest.analysis.slicing.handlers.base import StatementHandler

if TYPE_CHECKING:
    from codeharvest.analysis.slicing.slicer import CodeDependencySlicer

class DefinitionStatementHandler(StatementHandler):

    def _add_globals(self):
        for symbol in self.global_access_symbols:
            past_statement = self.ast_statements.resolve_last_stmt(symbol)
            if past_statement is None:
                all_wildcard_imports = self.ast_statements.resolve_wildcard_imports()
                if len(all_wildcard_imports) == 0:
                    slicer_logger.log(
                        logging.INFO,
                        f"Cannot resolve symbol {symbol} @ {self.ast_statement}",
                    )
                    continue
                for wildcard_import in all_wildcard_imports:
                    self._add_past_statement(wildcard_import, symbol)
                continue

            self._add_past_statement(past_statement, symbol)

    def _handle_callees(self):
        if self.slicer.callgraph_explorer is None:
            return
        callees = self.slicer.callgraph_explorer.find_file_callees(
            self.ast_statement.file, self.ast_statement.stmt.name
        )

        for callee in callees:
            if isinstance(callee, FunctionRecord):
                ast_stmts = self.slicer.get_file_ast_stmts(callee.file_path)
                assert callee.function_name is not None
                callee_aststmt = ast_stmts.find_function_stmt_with_name(
                    callee.function_name
                )

                if callee_aststmt is None:
                    continue

                self._add_past_statement(
                    callee_aststmt, callee.function_name, ast_stmts
                )
            elif isinstance(callee, ClassRecord):
                ast_stmts = self.slicer.get_file_ast_stmts(callee.file_path)
                assert callee.class_name is not None
                callee_aststmt = ast_stmts.find_class_stmt_with_name(callee.class_name)

                if callee_aststmt is None:
                    continue

                self._add_past_statement(callee_aststmt, callee.class_name, ast_stmts)
            elif isinstance(callee, MethodRecord):
                callee = callee.parent_class
                assert callee.class_name is not None
                ast_stmts = self.slicer.get_file_ast_stmts(callee.file_path)
                callee_aststmt = ast_stmts.find_class_stmt_with_name(callee.class_name)

                if callee_aststmt is None:
                    continue

                self._add_past_statement(callee_aststmt, callee.class_name, ast_stmts)
            else:
                raise ValueError(f"Unknown callee type {callee}")

    def _handle_internal_imports(self):
        internal_imports = collect_body_imports(self.ast_statement.stmt)
        new_imports: list[ast.Import | ast.ImportFrom] = []
        for internal_import in internal_imports:
            new_imports.extend(
                StatementTable.transform_imports(
                    self.ast_statement.file_path, internal_import
                )
            )

        for new_import in new_imports:
            fake_ast_stmt = self.ast_statements.create_fake_import_aststmt(new_import)
            if not self._is_repo_import(new_import):
                # 第三方包不展开，只保留 import 语句本身
                self.slicer.dependency_graph.add_edge(
                    self.ast_statement, fake_ast_stmt, "-1"
                )
                continue
            self.slicer.visit(
                fake_ast_stmt, self.ast_statements, "-1", depth=self.depth - 1
            )

    def _is_repo_import(self, import_node: ast.Import | ast.ImportFrom) -> bool:
        """判断 import 是否指向仓库内的文件（第三方包返回 False）。"""
        import_file = ImportPathResolver.resolve_module_path(
            self.ast_statement.file_path, import_node
        )
        return bool(import_file) and os.path.exists(import_file)

    def _handle_dynamic_imports(self):
        """动态引用（importlib / __import__ / getattr(模块, 变量)）时纳入整个模块。"""
        module_names = self._collect_dynamic_module_names()
        for module_name in module_names:
            import_node = ast.Import(names=[ast.alias(name=module_name)])
            import_file = ImportPathResolver.resolve_module_path(
                self.ast_statement.file_path, import_node
            )
            if not import_file or not os.path.exists(import_file):
                continue
            if import_file == self.ast_statement.file_path:
                continue
            try:
                ast_stmts = self.slicer.get_file_ast_stmts(import_file)
            except Exception:
                continue
            for stmt_node in ast_stmts.statements_list:
                self.slicer.visit(
                    stmt_node, ast_stmts, "-1", depth=self.depth - 1
                )

    def _collect_dynamic_module_names(self) -> list[str]:
        """收集语句中动态引用到的模块名。"""
        module_names = []
        root = self.ast_statement.stmt
        for node in ast.walk(root):
            if not isinstance(node, ast.Call):
                continue
            chain = self._call_chain(node)
            if chain in ("importlib.import_module", "__import__"):
                if node.args and isinstance(node.args[0], ast.Constant):
                    module_names.append(str(node.args[0].value))
            elif chain == "getattr" and len(node.args) >= 1:
                value = node.args[0]
                if isinstance(value, ast.Name):
                    binding = self._import_binding_for(value.id)
                    if binding is not None:
                        module_names.append(binding)
        return module_names

    @staticmethod
    def _call_chain(node: ast.Call) -> str:
        func = node.func
        parts = []
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name):
            parts.append(func.id)
        return ".".join(reversed(parts))

    def _import_binding_for(self, name: str) -> str | None:
        """找名字对应的 import 绑定（如 import a.b as name → a.b）。"""
        try:
            table = self.ast_statements
            stmt_node = table.resolve_last_stmt(name)
        except Exception:
            return None
        if stmt_node is None:
            return None
        stmt = stmt_node.stmt
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.asname == name:
                    return alias.name
        if isinstance(stmt, ast.ImportFrom) and stmt.module:
            for alias in stmt.names:
                if alias.asname == name or alias.name == name:
                    return f"{stmt.module}.{alias.name}"
        return None

    def _handle(self):
        stmt = self.ast_statement.stmt
        assert isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))

        self._handle_callees()
        self._handle_internal_imports()
        self._handle_dynamic_imports()

        return
