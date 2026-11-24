"""语句处理器基类。"""

import logging
from typing import TYPE_CHECKING

from codeharvest.logging_utils import slicer_logger
from codeharvest.analysis.slicing.globals import find_referenced_globals
from codeharvest.analysis.slicing.statements import StatementNode, StatementTable

if TYPE_CHECKING:
    from codeharvest.analysis.slicing.slicer import CodeDependencySlicer

class StatementHandler:
    def __init__(
        self,
        ast_statement: StatementNode,
        ast_statements: StatementTable,
        search_key: str,
        slicer: "CodeDependencySlicer",
        depth: int = -1,
    ):
        self.ast_statement = ast_statement
        self.ast_statements = ast_statements
        self.index = self.ast_statement.idx
        self.search_key = search_key
        self.slicer = slicer
        self.depth = depth

    def add_self_to_recursion_stack(self):
        self.slicer.recursion_stack.append(self.ast_statement)

    def pop_recursion_stack(self):
        self.slicer.recursion_stack.pop()

    def exists_in_recursion_stack(self):
        return self.ast_statement in self.slicer.recursion_stack

    def add_to_visited(self):
        self.slicer.visited_set.add(self.ast_statement)

    def exists_in_visited(self):
        return self.ast_statement in self.slicer.visited_set

    def _preprocess(self):
        if self.exists_in_recursion_stack():
            return True

        if self.exists_in_visited():
            return True

        
        self.add_self_to_recursion_stack()

        return False

    def _postprocess(self):
        self.pop_recursion_stack()
        self.add_to_visited()

    def _add_past_statement(
        self,
        past_statement: StatementNode,
        symbol: str,
        ast_statements: StatementTable | None = None,
    ):
        
        self.slicer.dependency_graph.add_edge(
            self.ast_statement,
            past_statement,
            symbol,
        )

        
        if ast_statements is None:
            self.slicer.visit(
                past_statement, self.ast_statements, symbol, depth=self.depth - 1
            )
        else:
            self.slicer.visit(
                past_statement, ast_statements, symbol, depth=self.depth - 1
            )

    def _add_globals(self):
        
        for symbol in self.global_access_symbols:
            
            
            past_statement = self.ast_statements.resolve_last_stmt(symbol, self.index)
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
        return

    def _handle(self):
        return

    def handle(self):
        
        exit = self._preprocess()
        if exit:
            return

        
        self.global_access_symbols = find_referenced_globals(
            self.ast_statement.stmt, unique=True
        )

        
        self._add_globals()

        
        self._handle()

        
        self._postprocess()
