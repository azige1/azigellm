"""上下文构建基类：截断与裁剪策略。"""

import ast
import tiktoken
from typing import Optional

from codeharvest.core.functions import FunctionRecord
from codeharvest.core.methods import MethodRecord
from codeharvest.core.classes import ClassRecord
from codeharvest.core.prompt_context import PromptContext
from codeharvest.synthesis.context.formatting import ContextRenderer, ContextOutputFormat
from codeharvest.analysis.ast_tools.transforms import (
    FunctionRemover,
    LastNodeRemover,
    ClassRemover,
    MethodToClassEndMover,
)

class ContextBuilder:

    def __init__(
        self,
        func_meth: FunctionRecord | MethodRecord,
        max_context_size: int | None = None,
        format: ContextOutputFormat = ContextOutputFormat.MARKDOWN_FILES,
    ):
        self.func_meth = func_meth
        self.repo_path = self.func_meth.repo.repo_path
        self.full_to_rel_path = lambda x: x.removeprefix(self.repo_path)[1:]
        self.max_context_size = max_context_size
        self.context_type = "naive"
        self.context = ""
        self.file2code = {}

        
        self.token_model = "gpt-3.5-turbo"
        self.tokenizer = tiktoken.encoding_for_model(self.token_model)

        self.format = format

    @property
    def context_size(self) -> int:
        return len(self.tokenizer.encode(self.context, disallowed_special=()))

    def get_context(self) -> PromptContext:
        context_info = {
            "context_type": self.context_type,
            "context": self.context,
        }
        return PromptContext(**context_info)

    def construct_context(self):
        fut_code = self.func_meth.code

        if isinstance(self.func_meth, MethodRecord):
            fut_code = self._keep_only_method_in_class(self.func_meth)

        self.context = ContextRenderer.format(
            fut_code,
            self.full_to_rel_path(self.func_meth.file_path),
            self.format,
        )

    def truncate_context(self):

        assert self.file2code is not {}, "file2code map empty"

        self.truncate_external_context()

        assert self.max_context_size is not None
        if self.context_size > self.max_context_size:
            self.truncate_file_context()

    def truncate_external_context(self):

        file2code_map = self.file2code.copy()
        fut_rel_path = self.full_to_rel_path(self.func_meth.file_path)
        fut_code = file2code_map.pop(fut_rel_path)

        
        if isinstance(self.func_meth, MethodRecord):
            fut_code = self._move_method_to_class_end(
                fut_code,
                self.func_meth.class_name,  
                self.func_meth.name,  
            )

        assert self.max_context_size is not None
        
        while self.context_size > self.max_context_size and len(file2code_map) > 0:
            last_file = list(file2code_map.keys())[0]
            last_code = file2code_map[last_file]

            
            if ast.parse(last_code).body == []:
                file2code_map.pop(last_file)

            
            else:
                truncated_code = self._remove_last_ast_node(last_code)
                file2code_map[last_file] = f"{truncated_code}\n\n# ..."

            self.context = ""
            for rel_path, code in file2code_map.items():
                self.context += ContextRenderer.format(code, rel_path, self.format)

            self.context += ContextRenderer.format(fut_code, fut_rel_path, self.format)

    def truncate_file_context(self):

        fut_rel_path = self.full_to_rel_path(self.func_meth.file_path)
        code = self.file2code[fut_rel_path]

        assert self.func_meth.name is not None, "FunctionRecord name not available"

        
        code, func_class_code = self._remove_func_class_from_file(code, self.func_meth)

        
        if isinstance(self.func_meth, MethodRecord):
            func_class_code = self._move_method_to_class_end(
                func_class_code,
                self.func_meth.class_name,  
                self.func_meth.name,
            )

        assert self.max_context_size is not None
        
        while self.context_size > self.max_context_size and ast.parse(code).body != []:
            code = self._remove_last_ast_node(code)  

            
            if ast.parse(code).body == []:
                self.context = ""
                self.context = ContextRenderer.format(
                    func_class_code, fut_rel_path, self.format
                )
                break
            else:
                code += "\n\n# ..."

            formatted_code = self._append_code(code, func_class_code)
            self.context = ContextRenderer.format(
                formatted_code, fut_rel_path, self.format
            )

    

    @staticmethod
    def _remove_func_class_from_file(
        code: str, func_meth: FunctionRecord | MethodRecord
    ) -> tuple[str, str]:
        tree = ast.parse(code)

        if isinstance(func_meth, (MethodRecord, ClassRecord)):
            transformer = ClassRemover(tree, func_meth.class_name)  
        else:
            transformer = FunctionRemover(tree, func_meth.name)  

        cleaned_tree = transformer.transform()
        removed_node = transformer.removed_node

        if removed_node is None:
            raise ValueError(f"FunctionRecord {func_meth.name} not found in code")

        return ast.unparse(cleaned_tree), ast.unparse(removed_node)

    @staticmethod
    def _append_code(code: str, removed_code: str) -> str:
        return f"{code}\n\n{removed_code.strip()}\n"

    @staticmethod
    def _remove_last_ast_node(code: str) -> str:
        tree = ast.parse(code)
        cleaned_tree = LastNodeRemover(tree).transform()
        return ast.unparse(cleaned_tree)

    @staticmethod
    def _move_method_to_class_end(code: str, class_name: str, method_name: str) -> str:
        tree = ast.parse(code)
        transformer = MethodToClassEndMover(tree, class_name, method_name)
        cleaned_tree = transformer.transform()
        return ast.unparse(cleaned_tree)

    @staticmethod
    def _keep_only_method_in_class(method: MethodRecord) -> str:
        class_name = method.class_name
        method_name = method.name
        file_path = method.file.file_path

        with open(file_path, "r") as f:
            code = f.read()
            tree = ast.parse(code)

        class_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                class_node = node
                break

        if class_node is None:
            raise ValueError(f"ClassRecord {class_name} not found for method {method_name}")

        class_node.body = [
            method
            for method in class_node.body
            if isinstance(method, ast.FunctionDef) and method.name == method_name
        ]
        class_node.body.append(ast.Expr(ast.Str("# ...")))

        return ast.unparse(class_node)
