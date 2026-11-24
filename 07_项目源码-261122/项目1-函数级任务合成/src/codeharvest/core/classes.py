"""类记录模型：从仓库中抽取出的类定义。"""

import ast
from pydantic import BaseModel, validator
from typing import Optional

from codeharvest.core.source_file import FileRecord
from codeharvest.core.identifiers import SymbolId
from codeharvest.core.repository import RepoRecord
from codeharvest.core.code_module import ModuleRecord
from codeharvest.utils.model_lookup import find_module_by_identifier, find_module_by_path
from codeharvest.analysis.callgraph.query import CallGraphQuery

class ClassRecord(BaseModel):
    class_id: SymbolId
    file: FileRecord
    class_name: Optional[str] = None

    _method_ids: Optional[list[SymbolId]] = None

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
        return self.class_name

    @property
    def id(self) -> str:
        return self.class_id.identifier

    @property
    def callees(self) -> list:
        cg_explorer = CallGraphQuery(self.repo)
        callees = cg_explorer.callees_for_identifier(self.id)
        return callees

    @property
    def method_ids(self) -> list[SymbolId]:
        
        

        if self._method_ids is not None:
            return self._method_ids

        if self.repo.callgraph is None:
            raise ValueError("Callgraph not found in the repo")

        if self.repo.callgraph.id2type is None:
            raise ValueError("Callgraph id2type not found in the repo")

        cgraph_ids = [k.identifier for k in self.repo.callgraph.id2type.keys()]
        class_id_str = self.class_id.identifier
        prefix = class_id_str + "."

        method_ids = [
            SymbolId(identifier=id_str)
            for id_str in cgraph_ids
            if id_str.startswith(prefix) and id_str.count(".") == prefix.count(".")
        ]

        self._method_ids = method_ids

        return self._method_ids

    

    @validator("class_name", pre=True, always=True)
    def set_class_name(cls, v, values):
        if v is not None:
            return v

        class_id = values.get("class_id")
        class_name = class_id.identifier.split(".")[-1]
        return class_name

    @classmethod
    def from_id_and_repo(cls, class_id: SymbolId, repo: RepoRecord) -> "ClassRecord":
        module: ModuleRecord = find_module_by_identifier(class_id, repo)

        return cls(
            class_id=class_id,
            file=FileRecord(file_module=module),
        )

    @classmethod
    def from_name_file_repo(
        cls, class_name: str, local_file_path: str, repo: RepoRecord
    ) -> "ClassRecord":
        module: ModuleRecord = find_module_by_path(local_file_path, repo)
        module_identifier = module.module_id.identifier
        class_id = SymbolId(identifier=module_identifier + "." + class_name)
        return cls(
            class_id=class_id,
            file=FileRecord(file_module=module),
        )
