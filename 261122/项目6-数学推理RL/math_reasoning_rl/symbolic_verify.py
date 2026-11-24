"""Rule-based mathematical answer extraction and SymPy equivalence checking."""

from __future__ import annotations

import math
import re
from typing import Any, Optional

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_LATEX_PAREN_REPLACEMENTS = [
    (r"\left(", "("),
    (r"\right)", ")"),
    (r"\left[", "["),
    (r"\right]", "]"),
    (r"\left\{", "{"),
    (r"\right\}", "}"),
    (r"\left|", "|"),
    (r"\right|", "|"),
    (r"\left", ""),
    (r"\right", ""),
    (r"\dfrac", r"\frac"),
    (r"\tfrac", r"\frac"),
    (r"\cfrac", r"\frac"),
    (r"\cdot", "*"),
    (r"\times", "*"),
    (r"\div", "/"),
    (r"\pi", "pi"),
    (r"\infty", "oo"),
    # 长命令先替换，避免 \leq 被 \le 截成 "<=q"
    (r"\leq", "<="),
    (r"\le", "<="),
    (r"\geq", ">="),
    (r"\ge", ">="),
    (r"\neq", "!="),
    (r"\pm", "+-"),
    ("π", "pi"),
    ("×", "*"),
    ("÷", "/"),
    ("−", "-"),
    ("－", "-"),
]


def extract_last_boxed(text: str) -> Optional[str]:
    """Extract content inside the last balanced ``\\boxed{...}`` or ``\\fbox{...}``."""
    # 两种宏统一按出现位置取最后一个
    candidates = [(text.rfind(macro), macro) for macro in (r"\boxed{", r"\fbox{")]
    candidates = [item for item in candidates if item[0] >= 0]
    for idx, macro in sorted(candidates, reverse=True):
        if idx >= 0:
            start = idx + len(macro)
            depth = 1
            pos = start
            while pos < len(text) and depth > 0:
                if text[pos] == "{":
                    depth += 1
                elif text[pos] == "}":
                    depth -= 1
                pos += 1
            if depth == 0:
                return text[start : pos - 1].strip()
    return None


def extract_rule_answer(answer_text: str) -> Optional[str]:
    """Extract a single final mathematical expression from an answer segment."""
    if not answer_text or not answer_text.strip():
        return None

    boxed = extract_last_boxed(answer_text)
    if boxed is not None:
        return boxed

    for pattern in (
        r"(?:最终答案|答案|Answer|Final Answer)\s*[:：]\s*\$?([^\n$]+)\$?",
        r"\$\$([^$]+)\$\$",
        r"\$([^$]+)\$",
    ):
        matches = re.findall(pattern, answer_text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    lines = [line.strip() for line in answer_text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else None


def _replace_frac_tokens(expr: str) -> str:
    """Convert nested ``\\frac{a}{b}`` expressions into ``((a)/(b))``."""
    while r"\frac" in expr:
        idx = expr.find(r"\frac")
        pos = idx + len(r"\frac")
        while pos < len(expr) and expr[pos].isspace():
            pos += 1

        args: list[str] = []
        for _ in range(2):
            if pos >= len(expr):
                return expr
            if expr[pos] == "{":
                depth = 1
                start = pos + 1
                pos += 1
                while pos < len(expr) and depth > 0:
                    if expr[pos] == "{":
                        depth += 1
                    elif expr[pos] == "}":
                        depth -= 1
                    pos += 1
                if depth != 0:
                    return expr
                args.append(expr[start : pos - 1])
            else:
                args.append(expr[pos])
                pos += 1

        replacement = f"(({args[0]})/({args[1]}))"
        expr = expr[:idx] + replacement + expr[pos:]
    return expr


def _replace_sqrt_tokens(expr: str) -> str:
    """Convert ``\\sqrt{a}`` and ``\\sqrt[n]{a}`` into SymPy ``sqrt(...)`` / ``root(...)``."""
    expr = re.sub(r"\\sqrt\[([^\]]+)\]\{([^}]+)\}", r"((\2)**(1/(\1)))", expr)
    while r"\sqrt" in expr:
        idx = expr.find(r"\sqrt")
        pos = idx + len(r"\sqrt")
        while pos < len(expr) and expr[pos].isspace():
            pos += 1
        if pos >= len(expr):
            break
        if expr[pos] == "{":
            depth = 1
            start = pos + 1
            pos += 1
            while pos < len(expr) and depth > 0:
                if expr[pos] == "{":
                    depth += 1
                elif expr[pos] == "}":
                    depth -= 1
                pos += 1
            if depth != 0:
                break
            inner = expr[start : pos - 1]
        else:
            inner = expr[pos]
            pos += 1
        expr = expr[:idx] + f"sqrt({inner})" + expr[pos:]
    return expr


def normalize_latex_expression(raw: str) -> str:
    """Normalize LaTeX math text into a parser-friendly expression string."""
    s = raw.strip()
    s = s.strip("$").strip()
    s = re.sub(r"\\(?:text|mathrm|mathbf|textbf)\{([^}]*)\}", r"\1", s)
    s = s.replace(r"\%", "/100").replace("%", "/100")
    s = s.replace(r"^{\circ}", "").replace(r"^\circ", "").replace("°", "")
    for old, new in _LATEX_PAREN_REPLACEMENTS:
        s = s.replace(old, new)
    s = _replace_frac_tokens(s)
    s = _replace_sqrt_tokens(s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    s = re.sub(r"\s+", "", s)
    return s.rstrip(".,;。，；")


def parse_to_sympy(raw: str) -> Optional[sp.Basic]:
    """Parse a LaTeX or algebraic string into a SymPy expression or Relational."""
    norm = normalize_latex_expression(raw)
    if not norm:
        return None

    # Handle equations like x=2 by converting to Eq or parsing directly.
    if "=" in norm and "==" not in norm and "<=" not in norm and ">=" not in norm and "!=" not in norm:
        parts = norm.split("=")
        if len(parts) == 2:
            lhs = parse_to_sympy(parts[0])
            rhs = parse_to_sympy(parts[1])
            if lhs is not None and rhs is not None:
                return sp.Eq(lhs, rhs)

    local_dict = {
        "pi": sp.pi,
        "e": sp.E,
        "E": sp.E,
        "i": sp.I,
        "I": sp.I,
        "oo": sp.oo,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "log": sp.log,
        "ln": sp.log,
        "exp": sp.exp,
        "abs": sp.Abs,
    }
    try:
        return parse_expr(norm, local_dict=local_dict, transformations=_TRANSFORMATIONS, evaluate=True)
    except Exception:
        return None


def _strip_leading_assignment(expr_str: str) -> str:
    """Strip single-variable assignment prefix such as ``x = 2`` -> ``2``."""
    cleaned = expr_str.strip()
    match = re.match(r"^[a-zA-Z]\w*\s*=\s*(.+)$", cleaned)
    return match.group(1).strip() if match else cleaned


def _are_sympy_objects_equivalent(a: Any, b: Any, tol: float = 1e-7) -> bool:
    """Check mathematical equivalence between two parsed SymPy objects."""
    # parse_expr 可能返回元组、列表等 Python 容器：只接受逐元素等价的同类容器
    if not isinstance(a, sp.Basic) or not isinstance(b, sp.Basic):
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and type(a) is type(b):
            return len(a) == len(b) and all(
                _are_sympy_objects_equivalent(x, y, tol=tol) for x, y in zip(a, b, strict=True)
            )
        return False

    if isinstance(a, sp.Equality) and isinstance(b, sp.Equality):
        diff_a = sp.simplify(a.lhs - a.rhs)
        diff_b = sp.simplify(b.lhs - b.rhs)
        if diff_a == 0 or diff_b == 0:
            return diff_a == diff_b
        # 两个方程的残差相差一个非零常数倍即等价（如 2x=4 与 x=2）
        try:
            ratio = sp.simplify(diff_a / diff_b)
            if ratio.free_symbols == set() and ratio.is_finite and ratio != 0:
                return True
        except Exception:
            pass
        return False

    # 方程与表达式比较：只允许「单个变量 = 答案」的赋值写法省略左侧
    if isinstance(a, sp.Equality) and not isinstance(b, sp.Equality):
        return isinstance(a.lhs, sp.Symbol) and _are_sympy_objects_equivalent(a.rhs, b, tol=tol)
    if isinstance(b, sp.Equality) and not isinstance(a, sp.Equality):
        return isinstance(b.lhs, sp.Symbol) and _are_sympy_objects_equivalent(a, b.rhs, tol=tol)

    # 无穷与非有限值只做精确比较，不进入数值容差
    if a.is_finite is False or b.is_finite is False:
        return a == b

    try:
        if sp.simplify(a - b) == 0:
            return True
    except Exception:
        pass

    try:
        if sp.expand(a - b) == 0:
            return True
    except Exception:
        pass

    # Numerical check for constant expressions (e.g. nested radicals / trig constants).
    if not a.free_symbols and not b.free_symbols:
        try:
            va = complex(a.evalf(30))
            vb = complex(b.evalf(30))
            if not all(math.isfinite(v) for v in (va.real, va.imag, vb.real, vb.imag)):
                return False
            return abs(va - vb) <= tol * max(1.0, abs(va), abs(vb))
        except Exception:
            pass

    return False


def are_expressions_equivalent(pred: str, gold: str) -> bool:
    """Determine whether predicted expression is symbolically equivalent to gold."""
    if pred is None or gold is None:
        return False

    pred_norm = normalize_latex_expression(pred)
    gold_norm = normalize_latex_expression(gold)
    if not pred_norm or not gold_norm:
        return False
    if pred_norm == gold_norm:
        return True

    # Handle comma-separated multi-value answers (e.g. roots "1, -2").
    if "," in pred_norm and "," in gold_norm:
        pred_items = [x.strip() for x in pred_norm.split(",") if x.strip()]
        gold_items = [x.strip() for x in gold_norm.split(",") if x.strip()]
        if len(pred_items) == len(gold_items) and all(
            are_expressions_equivalent(p, g) for p, g in zip(pred_items, gold_items, strict=True)
        ):
            return True

    sym_pred = parse_to_sympy(pred)
    sym_gold = parse_to_sympy(gold)
    if sym_pred is not None and sym_gold is not None:
        if _are_sympy_objects_equivalent(sym_pred, sym_gold):
            return True

    # Fallback: compare after stripping single-variable LHS assignment (e.g. "x=2" vs "2").
    pred_rhs = _strip_leading_assignment(pred)
    gold_rhs = _strip_leading_assignment(gold)
    if pred_rhs != pred or gold_rhs != gold:
        sym_pred_rhs = parse_to_sympy(pred_rhs)
        sym_gold_rhs = parse_to_sympy(gold_rhs)
        if sym_pred_rhs is not None and sym_gold_rhs is not None:
            if _are_sympy_objects_equivalent(sym_pred_rhs, sym_gold_rhs):
                return True

    return False


def verify_rule_answer(answer_text: str, ground_truth: str) -> tuple[bool, Optional[str]]:
    """Extract the rule-routed answer from text and verify symbolic equivalence."""
    extracted = extract_rule_answer(answer_text)
    if extracted is None:
        return False, None
    return are_expressions_equivalent(extracted, ground_truth), extracted
