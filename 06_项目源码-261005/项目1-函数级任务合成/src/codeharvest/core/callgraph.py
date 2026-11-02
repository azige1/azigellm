"""调用图模型：包装调用关系字典与元素类型标注。"""

import json
from typing import Optional
from pydantic import BaseModel
from enum import Enum, auto

from codeharvest.core.identifiers import SymbolId

class CodeElementKind(Enum):

    BUILTIN = auto()
    API = auto()
    FUNCTION = auto()
    METHOD = auto()
    CLASS = auto()
    OTHER = auto()

class CallGraphModel(BaseModel):
    graph: dict[SymbolId, list[SymbolId]]
    id2type: Optional[dict[SymbolId, CodeElementKind]] = None

    def get(self, key, default=None):
        if default is None:
            default = []
        return self.graph.get(key, default)

    def get_type(self, key):
        if self.id2type is None:
            return CodeElementKind.OTHER
        return self.id2type.get(key, CodeElementKind.OTHER)

    def keys(self):
        return self.graph.keys()

    def values(self):
        return self.graph.values()

    def items(self):
        return self.graph.items()

    def __contains__(self, key):
        return key in self.graph

    def __getitem__(self, key):
        return self.graph[key]

    def __len__(self):
        return len(self.graph)

    def __delitem__(self, key):
        del self.graph[key]

    def __setitem__(self, key, value):
        self.graph[key] = value

    

    @classmethod
    def from_dict(
        cls, graph: dict[str, list[str]], types: dict[str, str]
    ) -> "CallGraphModel":

        cgraph = {
            SymbolId(identifier=k): [SymbolId(identifier=v) for v in vs]
            for k, vs in graph.items()
        }

        id2type = {SymbolId(identifier=k): CodeElementKind[v] for k, v in types.items()}

        return cls(graph=cgraph, id2type=id2type)

    @classmethod
    def from_json(cls, file_path: str) -> "CallGraphModel":
        with open(file_path, "r") as f:
            data = json.load(f)

            if "graph" not in data:
                return cls.from_dict(graph=data, types={})

            graph = data.get("graph", {})
            id2type = data.get("id2type", {})
            return cls.from_dict(graph=graph, types=id2type)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            k.identifier: [v.identifier for v in vs] for k, vs in self.graph.items()
        }
