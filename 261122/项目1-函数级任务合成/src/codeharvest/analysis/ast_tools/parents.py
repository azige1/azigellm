"""为 AST 节点补充父指针，便于自底向上遍历。"""

import ast
from typing import TypeVar

T = TypeVar("T", bound=ast.AST)

def attach_parent_links(tree: T) -> T:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            setattr(child, "parent", node)
    setattr(tree, "parent", None)
    return tree
