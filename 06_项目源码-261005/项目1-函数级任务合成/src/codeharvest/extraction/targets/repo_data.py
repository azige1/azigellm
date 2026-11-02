"""按仓库汇总抽取结果的主逻辑。"""

import ast

from codeharvest.analysis.ast_tools import parse_ast_file
from codeharvest.core import SymbolId, RepoRecord, FileRecord, FunctionRecord, MethodRecord
from codeharvest.extraction.config import RepoJobConfig
from codeharvest.extraction.targets.methods import FileMethodCollector
from codeharvest.extraction.targets.functions import FileFunctionCollector

MAX_LINES_FUNCTION = 25
MAX_LINES_METHOD = 20

def extract_repo_targets(args) -> tuple[list[FunctionRecord], list[MethodRecord]]:
    repo, repo_args = args
    functions: list[FunctionRecord] = []
    methods: list[MethodRecord] = []
    no_filter = repo_args.disable_lines_filter or repo_args.disable_all_filters

    for file_path in repo.list_repo_files():
        if not file_path.endswith(".py"):
            continue

        
        file_path_split = file_path.split("/")
        if any([part.startswith(".") for part in file_path_split]):
            continue
        try:
            astree = parse_ast_file(file_path)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            continue

        function_asts = FileFunctionCollector.extract_functions_from_ast(
            astree, repo_args
        )
        file_obj = FileRecord.from_file_path(file_path, repo)
        for function_ast in function_asts:
            function_name = function_ast.name
            func_obj = FunctionRecord(
                function_id=SymbolId(
                    identifier=f"{file_obj.file_id}.{function_name}"
                ),
                file=file_obj,
                function_code=ast.unparse(function_ast),
                function_name=function_ast.name,
                function_complexity=None,
                context=None,
            )

            if no_filter or func_obj.num_code_lines < MAX_LINES_FUNCTION:
                try:
                    if func_obj.callee_count:
                        functions.append(func_obj)
                except Exception as e:  
                    functions.append(func_obj)

        method_asts = FileMethodCollector.extract_methods_from_ast(astree, repo_args)
        for method_ast in method_asts:
            method_name = method_ast.name
            parent_class_ast = method_ast.parent  
            parent_class_name = parent_class_ast.name  
            method_obj = MethodRecord(
                method_id=SymbolId(
                    identifier=f"{file_obj.file_id}.{parent_class_name}.{method_name}"
                ),
                file=file_obj,
                method_code=ast.unparse(method_ast),
                method_name=method_ast.name,
                parent_class_id=SymbolId(
                    identifier=f"{file_obj.file_id}.{parent_class_name}"
                ),
                context=None,
            )
            if no_filter or method_obj.num_code_lines < MAX_LINES_METHOD:
                try:
                    if method_obj.callee_count:
                        methods.append(method_obj)
                except Exception as e:  
                    methods.append(method_obj)

    return functions, methods

if __name__ == "__main__":
    repo = RepoRecord.from_file_path("./example_repo")
    f, m = extract_repo_targets(repo)
    
    
