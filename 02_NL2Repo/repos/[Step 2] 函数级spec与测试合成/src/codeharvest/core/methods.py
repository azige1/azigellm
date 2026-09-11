"""方法记录模型：从仓库中抽取出的类方法。"""

import ast
from typing import Optional

from pydantic import BaseModel, validator

from codeharvest.core.source_file import FileRecord
from codeharvest.core.repository import RepoRecord
from codeharvest.core.classes import ClassRecord
from codeharvest.core.code_module import ModuleRecord
from codeharvest.core.identifiers import SymbolId
from codeharvest.utils.model_lookup import find_module_by_identifier
from codeharvest.core.prompt_context import PromptContext
from codeharvest.analysis.callgraph.query import CallGraphQuery

class MethodRecord(BaseModel):
    method_id: SymbolId
    file: FileRecord
    method_code: str
    method_name: Optional[str] = None
    parent_class_id: Optional[SymbolId] = None
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
        return self.method_name

    @property
    def id(self) -> str:
        return self.method_id.identifier

    @property
    def code(self) -> str:
        return self.method_code

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
        callee_count = cg_explorer.count_callees(self.id)
        return callee_count

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

    

    @validator("method_name", pre=True, always=True)
    def set_method_name(cls, v, values):
        if v is not None:
            return v

        method_id = values.get("method_id")
        method_name = method_id.identifier.split(".")[-1]
        return method_name

    @validator("parent_class_id", pre=True, always=True)
    def set_parent_class_id(cls, v, values):
        if v is not None:
            return v

        method_id = values.get("method_id")
        parent_class_id = SymbolId(
            identifier=".".join(method_id.identifier.split(".")[:-1])
        )
        return parent_class_id

    def add_context(self, context: PromptContext):
        self.context = context

    @classmethod
    def from_id_and_repo(cls, method_id: SymbolId, repo: RepoRecord) -> "MethodRecord":
        module: ModuleRecord = find_module_by_identifier(method_id, repo)

        return cls(
            method_id=method_id,
            file=FileRecord(file_module=module),
            method_code="",
        )

    @property
    def parent_class(self) -> ClassRecord:
        assert self.parent_class_id is not None
        return ClassRecord.from_id_and_repo(
            class_id=self.parent_class_id,
            repo=self.repo,
        )

    @property
    def class_name(self) -> str | None:
        return self.parent_class.class_name
