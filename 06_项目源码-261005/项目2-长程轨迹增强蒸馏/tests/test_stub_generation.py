"""骨架生成测试：保留签名与 docstring，函数体替换为 raise NotImplementedError。"""

import ast

from task_builder.skeleton_stub import stub_source


def test_function_stubbed_but_signature_kept():
    source = (
        'def add(a, b=1, *args, **kwargs):\n'
        '    """加法。"""\n'
        '    return a + b\n'
    )
    result = stub_source(source)
    tree = ast.parse(result)
    func = tree.body[0]
    assert func.name == "add"
    # 参数名、顺序、默认值、*args/**kwargs 保留
    args = func.args
    assert [a.arg for a in args.args] == ["a", "b"]
    assert args.defaults and ast.unparse(args.defaults[0]) == "1"
    assert args.vararg.arg == "args" and args.kwarg.arg == "kwargs"
    # docstring 保留，函数体只有 docstring + raise
    assert ast.get_docstring(func) == "加法。"
    raises = [n for n in func.body if isinstance(n, ast.Raise)]
    assert len(raises) == 1
    assert isinstance(raises[0].exc, ast.Call) and raises[0].exc.func.id == "NotImplementedError"
    assert "return" not in result


def test_class_and_constants_kept():
    source = (
        "CONSTANT = 42\n"
        "import os\n"
        "__all__ = ['K']\n"
        "class K:\n"
        "    attr = 1\n"
        '    """类文档。"""\n'
        "    def method(self, x):\n"
        "        return x\n"
    )
    result = stub_source(source)
    assert "CONSTANT = 42" in result
    assert "__all__" in result
    assert "attr = 1" in result
    assert ast.get_docstring(ast.parse(result).body[2]) == "类文档。"
    method = ast.parse(result).body[2].body[1]
    assert method.name == "method"
    assert any(isinstance(n, ast.Raise) for n in method.body)
    assert "return x" not in result


def test_init_with_only_imports_unchanged():
    source = "from .core import add\n__all__ = ['add']\n"
    assert stub_source(source).strip() == source.strip()
