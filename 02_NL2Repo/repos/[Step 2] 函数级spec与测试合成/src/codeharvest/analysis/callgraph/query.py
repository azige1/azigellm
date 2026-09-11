"""调用图查询：按标识符或文件 + 函数名找 callee。"""

import typing_extensions

from codeharvest.core.identifiers import SymbolId
from codeharvest.core.repository import RepoRecord

from codeharvest.core.callgraph import CodeElementKind
from codeharvest.core.source_file import FileRecord
from codeharvest.core.callgraph import CallGraphModel

from codeharvest.logging_utils import logger

class CallGraphQuery:
    def __init__(self, repo: RepoRecord):
        self.repo = repo
        self.callgraph: CallGraphModel = repo.callgraph

    def merge_callee_lists(self, callers: list[SymbolId]) -> list[SymbolId]:
        merged_callgraph = set()
        for caller_id in callers:
            callees_ids = self.callgraph.get(caller_id, [])
            merged_callgraph.update(callees_ids)

        return list(merged_callgraph)

    def callees_for_identifier(self, caller_id: str) -> list:
        from codeharvest.core import FunctionRecord, MethodRecord, ClassRecord

        caller_identifier = SymbolId(identifier=caller_id)
        caller_type = self.callgraph.get_type(caller_identifier)
        callees_ids = self.callgraph.get(caller_identifier, [])

        if caller_type == CodeElementKind.CLASS:
            class_ = ClassRecord.from_id_and_repo(caller_identifier, self.repo)
            class_methods_ids = class_.method_ids
            callees_ids = self.merge_callee_lists([caller_identifier] + class_methods_ids)

        callees: list = []
        for cid in callees_ids:
            callee_type = self.callgraph.get_type(cid)
            try:
                if callee_type == CodeElementKind.METHOD:
                    callee = MethodRecord.from_id_and_repo(cid, self.repo)

                elif callee_type == CodeElementKind.CLASS:
                    callee = ClassRecord.from_id_and_repo(cid, self.repo)

                else:
                    callee = FunctionRecord.from_id_and_repo(cid, self.repo)

                callees.append(callee)
            except ValueError as e:
                
                
                pass

        return callees

    def count_callees(self, caller_id: str) -> int:
        caller_identifier = SymbolId(identifier=caller_id)
        caller_type = self.callgraph.get_type(caller_identifier)
        callees_ids = self.callgraph.get(caller_identifier, [])
        return len(callees_ids)

    def find_file_callees(self, file: FileRecord, function_name: str) -> list:
        module_id = file.file_module.module_id
        function_id = f"{module_id}.{function_name}"
        return self.callees_for_identifier(function_id)
