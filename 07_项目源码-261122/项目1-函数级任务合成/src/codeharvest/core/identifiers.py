"""符号标识符模型：用点分路径唯一标记仓库内的代码元素。"""

from pydantic import BaseModel

class SymbolId(BaseModel):
    identifier: str

    @classmethod
    def from_relative_path(cls, relative_path: str) -> "SymbolId":
        if relative_path.endswith(".py"):
            relative_path = relative_path[:-3]
        return cls(identifier=relative_path.replace("/", "."))

    @classmethod
    def from_absolute_path_repo_path(
        cls, absolute_path: str, repo_path: str
    ) -> "SymbolId":
        relative_path = absolute_path.replace(repo_path, "")
        return cls.from_relative_path(relative_path)

    def __str__(self):
        return self.identifier

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        if isinstance(other, SymbolId):
            return self.identifier == other.identifier
        return False
