"""把 import 语句解析成仓库内的实际文件路径。"""

import ast
import os

from codeharvest.analysis.modules.inspection import ModuleInspector

class ImportPathResolver:

    @staticmethod
    def resolve_module_path(file_path: str, node: ast.ImportFrom | ast.Import) -> str:
        if isinstance(node, ast.ImportFrom):
            module_parts = (node.module or "").split(".")
            level = node.level
        elif isinstance(node, ast.Import):
            if not node.names:
                raise ValueError("No module name found in import statement.")
            module_parts = node.names[0].name.split(".")
            level = 0
        else:
            raise ValueError("Unsupported import node type.")

        base_dir = os.path.dirname(file_path)
        root_dir = ModuleInspector.find_package_root(file_path)
        resolved_path = None

        if level > 0:
            resolved_path = ImportPathResolver.resolve_relative_module(
                base_dir, module_parts, level
            )
        else:
            resolved_path = ImportPathResolver.resolve_absolute_module(
                base_dir, root_dir, module_parts
            )

        if os.path.isdir(os.path.abspath(resolved_path)):
            package_path = os.path.join(resolved_path, "__init__.py")
            return os.path.abspath(package_path)
        else:
            return os.path.abspath(resolved_path + ".py")

    

    @staticmethod
    def resolve_absolute_module(
        base_dir: str, root_dir: str, module_parts: list[str]
    ) -> str:

        def potential_path_exists(path: str) -> bool:
            return os.path.exists(path) or os.path.exists(path + ".py")

        
        resolved_path = os.path.join(root_dir, *module_parts)
        if potential_path_exists(resolved_path):
            return resolved_path

        
        resolved_path = os.path.join(base_dir, *module_parts)
        if potential_path_exists(resolved_path):
            return resolved_path

        
        current_dir = base_dir
        while current_dir.startswith(root_dir):
            potential_path = os.path.join(current_dir, *module_parts)

            if potential_path_exists(potential_path):
                return potential_path

            current_dir = os.path.dirname(current_dir)

        
        return os.path.join(root_dir, *module_parts)

    @staticmethod
    def resolve_relative_module(
        base_dir: str, module_parts: list[str], level: int
    ) -> str:
        return os.path.join(base_dir, *[".."] * (level - 1), *module_parts)
