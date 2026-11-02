"""完整仓库上下文构建器。"""

import os
import ast

from codeharvest.synthesis.context.builder import ContextBuilder
from codeharvest.synthesis.context.formatting import ContextRenderer, ContextOutputFormat
from codeharvest.analysis.callgraph.query import CallGraphQuery
from codeharvest.analysis.imports.resolver import ImportPathResolver
from codeharvest.core import FunctionRecord, MethodRecord

class FullRepoContextBuilder(ContextBuilder):

    def __init__(
        self,
        func_meth: FunctionRecord | MethodRecord,
        max_context_size: int | None = None,
        format: ContextOutputFormat = ContextOutputFormat.MARKDOWN_FILES,
        filter_calls: bool = True,
    ):
        super().__init__(func_meth, max_context_size, format)
        self.context_type = "full"
        self.filter_calls = filter_calls
        self.cg_explorer = CallGraphQuery(self.func_meth.repo)
        self.construct_context()

    def construct_context(self):

        if self.filter_calls:
            context_files = self.callee_files()
        else:
            context_files = self.imported_files()

        for file_path in context_files:
            if file_path == self.func_meth.file_path:
                continue

            
            with open(file_path, "r") as f:
                rel_path = self.full_to_rel_path(file_path)
                code = f.read()
                self.context += ContextRenderer.format(code, rel_path, self.format)
                self.file2code[rel_path] = code

        self.context += self.processed_fut_file()

        
        if self.max_context_size and self.context_size > self.max_context_size:
            self.truncate_context()

    

    def callee_files(self) -> set[str]:
        caller_id = self.func_meth.id
        callees: list = self.cg_explorer.callees_for_identifier(
            caller_id=caller_id
        )
        return {callee.file_path for callee in callees}

    def imported_files(self) -> set[str]:
        with open(self.func_meth.file_path, "r") as f:
            tree = ast.parse(f.read())

        imported_files = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                file_path = ImportPathResolver.resolve_module_path(
                    self.func_meth.file_path, node
                )

                if os.path.exists(file_path):
                    imported_files.add(file_path)

        return imported_files

    def processed_fut_file(self) -> str:
        with open(self.func_meth.file_path, "r") as f:
            code = f.read()

        assert self.func_meth.name is not None, "FunctionRecord name not available"

        code, func_code = self._remove_func_class_from_file(code, self.func_meth)
        code = code + "\n\n" + func_code
        rel_path = self.full_to_rel_path(self.func_meth.file_path)
        fut_file_context = ContextRenderer.format(code.strip(), rel_path, self.format)
        self.file2code[rel_path] = code

        return fut_file_context
