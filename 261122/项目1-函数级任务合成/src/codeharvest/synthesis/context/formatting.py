"""上下文输出格式化（markdown / 路径注释）。"""

from enum import Enum

class ContextOutputFormat(Enum):
    MARKDOWN_FILES = "markdown_files"
    PATH_COMMENT = "path_comment"

class ContextRenderer:
    _format_methods = {
        ContextOutputFormat.MARKDOWN_FILES: "_to_markdown_files",
        ContextOutputFormat.PATH_COMMENT: "_to_path_comment",
    }

    @classmethod
    def format(cls, code: str, file_path: str, format_type: ContextOutputFormat) -> str:
        method_name = cls._format_methods.get(format_type)

        if method_name is None:
            raise ValueError(f"Unsupported format: {format_type}")

        method = getattr(cls, method_name)
        return method(code, file_path)

    @staticmethod
    def _to_markdown_files(code: str, file_path: str) -> str:
        return f"```python\n# {file_path}\n\n{code}\n```\n"

    @staticmethod
    def _to_path_comment(code: str, file_path: str) -> str:
        return f"# {file_path}\n\n{code}\n"
