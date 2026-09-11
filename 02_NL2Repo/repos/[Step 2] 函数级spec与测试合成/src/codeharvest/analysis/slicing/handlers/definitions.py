"""类 / 函数定义语句的处理器。"""

import ast
import logging
from typing import TYPE_CHECKING

from codeharvest.logging_utils import slicer_logger
from codeharvest.core import FunctionRecord, ClassRecord, MethodRecord
from codeharvest.analysis.ast_tools.finder import collect_body_imports
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
            self.slicer.visit(
                fake_ast_stmt, self.ast_statements, "-1", depth=self.depth - 1
            )

    def _handle(self):
        stmt = self.ast_statement.stmt
        assert isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))

        self._handle_callees()
        self._handle_internal_imports()

        return
