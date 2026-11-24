"""模块记录模型：描述文件 / 类 / 函数等不同粒度的模块单元。"""

import os
from enum import Enum
from pydantic import BaseModel

from codeharvest.core.repository import RepoRecord
from codeharvest.core.identifiers import SymbolId

class ModuleKind(str, Enum):
    FILE = "file"
    PACKAGE = "package"

class ModuleRecord(BaseModel):
    module_id: SymbolId
    module_type: ModuleKind = ModuleKind.FILE
    repo: RepoRecord

    

    @property
    def local_path(self) -> str:
        parts = self.module_id.identifier.split(".")
        path = self.repo.repo_path
        segment = ""

        for i, part in enumerate(parts):
            
            segment = f"{segment}.{part}" if segment else part
            current_path = os.path.join(path, segment)

            
            if os.path.exists(current_path):
                path = current_path
                segment = ""

            elif i == len(parts) - 1:
                
                if self.module_type != ModuleKind.PACKAGE:
                    py_path = f"{current_path}.py"
                    if os.path.exists(py_path):
                        return py_path

                
                return os.path.join(path, segment)

        return path

    @property
    def relative_module_path(self) -> str:
        if self.module_type == ModuleKind.PACKAGE:
            return self.module_id.identifier.replace(".", "/")
        else:
            return f"{self.module_id.identifier.replace('.', '/')}.py"

    @property
    def _repo_name(self) -> str:
        return self.repo.repo_name

    @property
    def repo_id(self) -> str:
        return self.repo.repo_id

    

    def exists(self) -> bool:
        return os.path.exists(self.local_path)

    @classmethod
    def from_file_path(cls, file_path: str, repo: RepoRecord | None) -> "ModuleRecord":
        if repo is None:
            repo = RepoRecord.from_file_path(file_path)
        module_id = SymbolId(
            identifier=file_path.replace(repo.repo_path, "")[1:]
            .replace(".py", "")
            .replace("/", ".")
        )
        return cls(module_id=module_id, repo=repo)
