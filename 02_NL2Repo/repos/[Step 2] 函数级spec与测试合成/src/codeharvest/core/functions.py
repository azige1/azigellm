"""函数记录模型：从仓库中抽取出的顶层函数。"""

import ast
from typing import Optional, Any
import typing_extensions
from pydantic import BaseModel, BaseConfig, Field, validator

from codeharvest.core.repository import RepoRecord
from codeharvest.core.source_file import FileRecord
from codeharvest.core.code_module import ModuleRecord
from codeharvest.core.identifiers import SymbolId
from codeharvest.core.prompt_context import PromptContext
from codeharvest.utils.model_lookup import find_module_by_identifier, find_module_by_path
from codeharvest.analysis.callgraph.query import CallGraphQuery

class FunctionRecord(BaseModel):
    function_id: SymbolId
    file: FileRecord
    function_code: str
    function_name: Optional[str] = None
    function_complexity: Optional[str] = None
    context: Optional[PromptContext] = None

    @property
    def file_path(self) -> str:
        return self.file.file_path

    @property
    def repo(self) -> RepoRecord:
        return self.file.repo

    @property
    def _repo_name(self) -> str:
        return self.file._repo_name

    @property
    def repo_id(self) -> str:
        return self.file.repo_id

    @property
    def module(self) -> ModuleRecord:
        return self.file.file_module

    @property
    def name(self) -> None | str:
        return self.function_name

    @property
    def id(self) -> str:
        return self.function_id.identifier

    @property
    def code(self) -> str:
        return self.function_code

    @property
    def code_ast(self) -> ast.FunctionDef:
        return ast.parse(self.code).body[0]  

    @property
    def callees(self) -> list:
        cg_explorer = CallGraphQuery(self.repo)
        callees = cg_explorer.callees_for_identifier(self.id)
        return callees

    @property
    def callee_count(self) -> int:
        cg_explorer = CallGraphQuery(self.repo)
        return cg_explorer.count_callees(self.id)

    @property
    def num_code_lines(self) -> int:
        num_lines = 0
        for idx, body_stmt in enumerate(self.code_ast.body):
            if (
                idx == 0
                and isinstance(body_stmt, ast.Expr)
                and isinstance(body_stmt.value, ast.Constant)
                and isinstance(body_stmt.value.value, str)
            ):
                continue
            num_lines += ast.unparse(body_stmt).count("\n") + 1
        return num_lines

    

    @validator("function_name", pre=True, always=True)
    def populate_function_name(cls, v, values):
        function_id: SymbolId = values.get("function_id")
        if v is None:
            return function_id.identifier.split(".")[-1]
        return v

    def add_context(self, context: PromptContext):
        self.context = context

    @classmethod
    def from_id_and_repo(cls, function_id: SymbolId, repo: RepoRecord) -> "FunctionRecord":
        module: ModuleRecord = find_module_by_identifier(function_id, repo)

        return cls(
            function_id=function_id,
            file=FileRecord(file_module=module),
            function_code="",
        )

    @classmethod
    def from_name_file_repo(
        cls, function_name: str, local_file_path: str, repo: RepoRecord
    ) -> "FunctionRecord":
        pass
        module: ModuleRecord = find_module_by_path(local_file_path, repo)
        module_identifier = module.module_id.identifier
        function_id = SymbolId(identifier=module_identifier + "." + function_name)
        return cls(
            function_id=function_id,
            file=FileRecord(file_module=module),
            function_code="",
        )
