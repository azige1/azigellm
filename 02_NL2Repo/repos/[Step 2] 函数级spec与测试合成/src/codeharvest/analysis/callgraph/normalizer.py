"""调用图清洗：规范化 callee 标识、剔除不可解析条目。"""

import os
import ast
import json
from typing import TYPE_CHECKING

from codeharvest.core.callgraph import CallGraphModel
from codeharvest.core.repository import RepoRecord
from codeharvest.core.identifiers import SymbolId
from codeharvest.analysis.modules.inspection import ModuleInspector
from codeharvest.analysis.imports.resolver import ImportPathResolver
from codeharvest.utils.model_lookup import find_module_by_identifier, resolve_element_kind

from codeharvest.workspace import REPOS_DIR
from codeharvest.logging_utils import logger

if TYPE_CHECKING:
    from codeharvest.core.code_module import ModuleRecord
    from codeharvest.utils.model_lookup import CodeElementKind

class CallGraphNormalizer:

    @staticmethod
    def normalize_callee_identifiers(repo: RepoRecord):
        cgraph = repo.callgraph
        stats = {
            "retained": set(),
            "normalized": set(),
            "unresolved_import": set(),
            "unnormalized": set(),
        }

        for caller, callees in cgraph.items():
            caller_file = find_module_by_identifier(caller, repo).local_path

            pkg_root = ModuleInspector.find_package_root(caller_file)
            pkg_start = os.path.dirname(pkg_root)
            if pkg_start == str(REPOS_DIR):
                pkg_start = pkg_root

            
            unresolved_callees = CallGraphNormalizer.find_unresolvable_ids(callees, repo)
            stats["retained"].update(set(callees) - set(unresolved_callees))

            for callee in unresolved_callees:
                callee_parts = callee.identifier.split(".")
                module_parts, function_name = callee_parts[:-1], callee_parts[-1]

                
                temp_node = ast.Import(
                    names=[ast.alias(name=".".join(module_parts), asname=None)]
                )
                callee_file = ImportPathResolver.resolve_module_path(caller_file, temp_node)

                
                if os.path.exists(callee_file):
                    relative_module_path = os.path.relpath(callee_file, start=pkg_start)
                    module_notation = relative_module_path.replace(os.sep, ".")[:-3]

                    
                    try:
                        callee.identifier = f"{module_notation}.{function_name}"
                        find_module_by_identifier(callee, repo)
                        stats["normalized"].add(callee)
                    except ValueError as e:
                        stats["unnormalized"].add(callee)
                        pass

                else:
                    stats["unresolved_import"].add(callee)

        return stats

    @staticmethod
    def drop_empty_callees(repo: RepoRecord):
        callers_to_check = list(repo.callgraph.keys())

        for caller in callers_to_check:
            callees = repo.callgraph[caller]
            if len(callees) == 0:
                del repo.callgraph[caller]

    @staticmethod
    def drop_unresolvable_callers(repo: RepoRecord):

        
        unresolvable_callers = CallGraphNormalizer.find_unresolvable_ids(
            list(repo.callgraph.keys()), repo
        )

        for caller in unresolvable_callers:
            del repo.callgraph[caller]

    @staticmethod
    def build_id_kind_map(repo: RepoRecord) -> dict[SymbolId, "CodeElementKind"]:
        cgraph = repo.callgraph
        id2type_map: dict[SymbolId, "CodeElementKind"] = {}

        unique_ids = set()
        for caller, callees in cgraph.items():
            unique_ids.add(caller)
            unique_ids.update(callees)

        for uid in unique_ids:
            id2type_map[uid] = resolve_element_kind(uid, repo)

        return id2type_map

    

    @staticmethod
    def find_unresolvable_ids(
        identifiers: list[SymbolId], repo: RepoRecord
    ) -> list[SymbolId]:
        unresolvable_identifiers = []
        for identifier in identifiers:
            try:
                module: ModuleRecord = find_module_by_identifier(identifier, repo)
            except ValueError:
                unresolvable_identifiers.append(identifier)

        return unresolvable_identifiers
