"""import 语句的处理器。"""

import os
import ast
from typing import TYPE_CHECKING

from codeharvest.analysis.imports import ImportPathResolver
from codeharvest.analysis.slicing.statements import StatementTable
from codeharvest.analysis.slicing.handlers.base import StatementHandler

if TYPE_CHECKING:
    from codeharvest.analysis.slicing.slicer import CodeDependencySlicer

class ImportStatementHandler(StatementHandler):
    def _handle(self):
        import_file = ImportPathResolver.resolve_module_path(
            self.ast_statements.file_path, self.ast_statement.stmt  
        )
        if not os.path.exists(import_file):
            
            return
        if import_file == self.ast_statement.file_path:
            return
        ast_stmts = self.slicer.get_file_ast_stmts(import_file)

        past_statement = ast_stmts.resolve_last_stmt(self.search_key)

        if past_statement is None:
            return

        self._add_past_statement(past_statement, self.search_key, ast_stmts)
