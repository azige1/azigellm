"""改写 import：相对转绝对、通配符展开。"""

import ast
import os
import shutil
import importlib

from codeharvest.analysis.modules.inspection import ModuleInspector
from codeharvest.analysis.imports.resolver import ImportPathResolver

class ImportRewriter:

    @staticmethod
    def make_absolute_import(file_path: str, node: ast.ImportFrom) -> None:
        module_parts = (node.module or "").split(".")
        package_name = ModuleInspector.find_package_name(file_path)
        abs_parts = package_name.split(".") + module_parts[node.level - 1 :]
        node.module = ".".join(abs_parts).rstrip(".")
        node.level = 0

    @staticmethod
    def expand_wildcard_import(file_path: str, node: ast.ImportFrom) -> None:
        module_file = ImportPathResolver.resolve_module_path(file_path, node)
        if os.path.exists(module_file):
            all_members = ModuleInspector.list_member_names(module_file)
            node.names = [ast.alias(name=member, asname=None) for member in all_members]

        
        else:
            try:
                module = importlib.import_module(node.module)  
                all_members = [name for name in dir(module) if not name.startswith("_")]
                if hasattr(module, "__all__"):
                    all_members = [
                        name for name in all_members if name in module.__all__
                    ]
                node.names = [
                    ast.alias(name=member, asname=None) for member in all_members
                ]
            except ImportError:
                pass

    @staticmethod
    def rewrite_import_node(file_path: str, node: ast.ImportFrom) -> None:
        if isinstance(node, ast.ImportFrom):
            if node.level > 0:
                ImportRewriter.make_absolute_import(file_path, node)

            if node.names[0].name == "*":
                ImportRewriter.expand_wildcard_import(file_path, node)

    @staticmethod
    def rewrite_file_imports(file_path: str) -> None:
        with open(file_path, "r") as file:
            try:
                tree = ast.parse(file.read())
            except SyntaxError:
                raise SyntaxError(f"Syntax error in file: {file_path}")

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                ImportRewriter.rewrite_import_node(file_path, node)

        source_code = ast.unparse(tree)
        ast.parse(source_code)
        with open(file_path, "w") as file:
            file.write(source_code)

    @staticmethod
    def rewrite_repo_imports(repo_path: str) -> str:
        temp_path = repo_path + "_temp"
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        temp_path = shutil.copytree(repo_path, temp_path)

        for root, _, files in os.walk(temp_path):
            for file in files:
                if file.endswith(".py"):
                    try:
                        ImportRewriter.rewrite_file_imports(os.path.join(root, file))
                    except SyntaxError as e:
                        print(f"Error in file: {os.path.join(root, file)}")
                        print(e)
                        raise e
        return temp_path
