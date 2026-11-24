"""规格文档生成：README + 目录结构树 + 公开接口清单 + docs 文本，截断到 2.4 万字符。

2.4 万字符约 8 千 token（字符数 / 3）；学生上下文 32K，规格过长会挤占轨迹空间并提前触发压缩。
"""

from pathlib import Path
import ast

MAX_SPEC_CHARS = 24000


def collect_public_apis(repo_dir: Path, src_dir: str, test_dir: str = "tests") -> dict[str, list[str]]:
    """按模块收集公开函数、类与方法签名和 docstring 首段（跳过位于源码目录内的测试目录）。"""
    apis: dict[str, list[str]] = {}
    test_root = repo_dir / test_dir
    for py_file in sorted((repo_dir / src_dir).rglob("*.py")):
        if test_root in py_file.parents:
            continue
        module = py_file.relative_to(repo_dir / src_dir).with_suffix("").as_posix()
        if module.endswith("__init__"):
            module = module[: -len(".__init__")] or "__init__"
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        entries: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                entries.append(_format_function(node))
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                header = f"class {node.name}"
                if node.bases:
                    header += f"({', '.join(ast.unparse(b) for b in node.bases)})"
                entries.append(header)
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and not (
                        sub.name.startswith("_") and sub.name not in ("__init__",)
                    ):
                        entries.append("    " + _format_function(sub))
        if entries:
            apis[module] = entries
    return apis


def _format_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    sig = f"{prefix} {node.name}({ast.unparse(node.args)})"
    if node.returns is not None:
        sig += f" -> {ast.unparse(node.returns)}"
    doc = ast.get_docstring(node)
    if doc:
        first = doc.strip().split("\n\n")[0].replace("\n", " ")
        sig += f"\n    {first}"
    return sig


def render_directory_tree(repo_dir: Path, test_dir: str = "tests", max_entries: int = 200) -> str:
    """生成简化的目录结构树（跳过 .git、缓存与测试目录）。"""
    lines: list[str] = []
    skip_dirs = {".git", "__pycache__", ".venv", ".pytest_cache"}
    test_root = repo_dir / test_dir
    count = 0
    for path in sorted(repo_dir.rglob("*")):
        if any(part in skip_dirs for part in path.parts):
            continue
        if path == test_root or test_root in path.parents:
            continue
        count += 1
        if count > max_entries:
            lines.append("...")
            break
        depth = len(path.relative_to(repo_dir).parts) - 1
        lines.append("    " * depth + path.name + ("/" if path.is_dir() else ""))
    return "\n".join(lines)


def find_readme(repo_dir: Path) -> Path | None:
    """按常见文件名查找 README（不少 Python 库用 .rst）。"""
    for name in ("README.md", "README.rst", "README.txt", "README"):
        candidate = repo_dir / name
        if candidate.exists():
            return candidate
    return None


def make_spec_markdown(
    repo_dir: Path,
    src_dir: str,
    test_dir: str = "tests",
    docs_dir: str = "docs",
    max_chars: int = MAX_SPEC_CHARS,
) -> str:
    """组装 spec.md：依次为 README、目录结构、公开接口、文档，超过 max_chars 时从尾部截断。

    骨架里已保留全部签名与 docstring，文档排在最后，截断时优先舍弃。
    目录结构与公开接口都不包含测试目录，判分测试的文件名和用例名不进入规格。
    """
    parts: list[str] = ["# 任务规格", ""]
    readme = find_readme(repo_dir)
    if readme is not None:
        parts += ["## README", "", readme.read_text(encoding="utf-8", errors="ignore"), ""]
    parts += ["## 目录结构", "", "```", render_directory_tree(repo_dir, test_dir), "```", ""]
    parts += ["## 公开接口", ""]
    for module, entries in collect_public_apis(repo_dir, src_dir, test_dir).items():
        parts.append(f"### {module}")
        parts.append("```python\n" + "\n".join(entries) + "\n```")
        parts.append("")
    docs = repo_dir / docs_dir
    if docs.is_dir():
        doc_text = []
        for doc_file in sorted(docs.rglob("*")):
            if doc_file.suffix.lower() not in (".md", ".rst"):
                continue
            doc_text.append(
                f"## {doc_file.relative_to(repo_dir).as_posix()}\n\n"
                + doc_file.read_text(encoding="utf-8", errors="ignore")
            )
        if doc_text:
            parts += ["## 文档", ""] + doc_text + [""]
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[文档已截断]\n"
    return text


__all__ = []
