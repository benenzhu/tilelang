"""Simplify redundant parentheses in generated CUDA/HIP source code.

This module provides a default postproc pass that removes unnecessary parentheses
from ``[...]`` index expressions and ``*(type *)(expr)`` pointer dereference
expressions while preserving correct operator precedence.
"""

from __future__ import annotations

import re

# C++ operator precedence (higher number = binds tighter)
_PRECEDENCE = {
    "UNARY": 50,
    "*": 40, "/": 40, "%": 40,
    "+": 30, "-": 30,
    "<<": 20, ">>": 20,
    "&": 10,
    "^": 9,
    "|": 8,
}

# Regex: match a C-style type cast like (int), (float*), (uint32_t), (const int) etc.
_CAST_PATTERN = (
    r"\(\s*(?:const\s+)?"
    r"(?:int|float|double|char|short|long|unsigned|signed|void|[a-zA-Z0-9_]+_t)"
    r"(?:\s*\*)?\s*\)"
)
# Regular token pattern
_REGULAR_PATTERN = r">>|<<|[a-zA-Z_][a-zA-Z0-9_\.]*|\d+|[+*&/()\-~!%^|]"
_TOKEN_RE = re.compile(f"{_CAST_PATTERN}|{_REGULAR_PATTERN}")

# Match pointer cast pattern: (type *) where type can be multi-word like "unsigned int"
# or templated like uint4, bfloat16x8_vec, float32x4, etc.
_PTR_CAST_RE = re.compile(
    r"\(\s*(?:const\s+)?[a-zA-Z_]\w*(?:\s+[a-zA-Z_]\w*)*\s*\*\s*\)"
)


class _ASTNode:
    __slots__ = ("value", "left", "right", "is_unary", "is_op", "prec")

    def __init__(self, value: str, left=None, right=None, *, is_unary: bool = False):
        self.value = value
        self.left = left
        self.right = right
        self.is_unary = is_unary
        self.is_op = (value in _PRECEDENCE) or is_unary
        if is_unary:
            self.prec = _PRECEDENCE["UNARY"]
        else:
            self.prec = _PRECEDENCE.get(value, 100)


def _tokenize(expr: str) -> list[str]:
    return _TOKEN_RE.findall(expr)


def _build_ast(tokens: list[str]) -> _ASTNode | None:
    if not tokens:
        return None
    op_stack: list[str] = []
    node_stack: list[_ASTNode] = []

    def _pop_op():
        op = op_stack.pop()
        if op.startswith("U:") or op.startswith("CAST:"):
            right = node_stack.pop()
            node_stack.append(_ASTNode(op, None, right, is_unary=True))
        else:
            right = node_stack.pop()
            left = node_stack.pop()
            node_stack.append(_ASTNode(op, left, right))

    prev_is_op_or_lparen = True

    for token in tokens:
        if token == "(":
            op_stack.append(token)
            prev_is_op_or_lparen = True
        elif token == ")":
            while op_stack and op_stack[-1] != "(":
                _pop_op()
            if op_stack:
                op_stack.pop()
            prev_is_op_or_lparen = False
        elif token.startswith("(") and token.endswith(")"):
            # Type cast token
            op_stack.append(f"CAST:{token}")
            prev_is_op_or_lparen = True
        elif token in _PRECEDENCE or token in ("-", "+", "*", "&", "~", "!"):
            op = token
            is_unary_now = False
            if prev_is_op_or_lparen and token in ("-", "+", "*", "&"):
                op = f"U:{token}"
                is_unary_now = True
            elif token in ("~", "!"):
                op = f"U:{token}"
                is_unary_now = True

            curr_prec = _PRECEDENCE["UNARY"] if is_unary_now else _PRECEDENCE.get(op, 0)
            assoc = "R" if is_unary_now else "L"

            while op_stack and op_stack[-1] != "(":
                top_op = op_stack[-1]
                top_is_unary = top_op.startswith("U:") or top_op.startswith("CAST:")
                top_prec = _PRECEDENCE["UNARY"] if top_is_unary else _PRECEDENCE.get(top_op, 0)
                if (assoc == "L" and top_prec >= curr_prec) or (
                    assoc == "R" and top_prec > curr_prec
                ):
                    _pop_op()
                else:
                    break
            op_stack.append(op)
            prev_is_op_or_lparen = True
        else:
            node_stack.append(_ASTNode(token))
            prev_is_op_or_lparen = False

    while op_stack:
        _pop_op()

    return node_stack[0] if node_stack else None


def _generate_code(node: _ASTNode | None) -> str:
    if not node:
        return ""
    if not node.is_op:
        return node.value

    if node.is_unary:
        right_str = _generate_code(node.right)
        if node.right and node.right.is_op and node.right.prec < node.prec:
            right_str = f"({right_str})"
        if node.value.startswith("U:"):
            return f"{node.value[2:]}{right_str}"
        elif node.value.startswith("CAST:"):
            return f"{node.value[5:]} {right_str}"

    left_str = _generate_code(node.left)
    right_str = _generate_code(node.right)

    if node.left and node.left.is_op and node.left.prec < node.prec:
        left_str = f"({left_str})"

    if node.right and node.right.is_op:
        if node.right.prec < node.prec:
            right_str = f"({right_str})"
        elif node.right.prec == node.prec:
            right_str = f"({right_str})"

    return f"{left_str} {node.value} {right_str}"


def _simplify_expr(expr: str) -> str:
    """Simplify parentheses in a single expression string."""
    tokens = _tokenize(expr)
    ast = _build_ast(tokens)
    return _generate_code(ast)


def _find_matching_paren(code: str, start: int) -> int:
    """Return the index past the matching ')' for '(' at *start*, or -1."""
    depth = 1
    j = start + 1
    n = len(code)
    while j < n and depth > 0:
        if code[j] == "(":
            depth += 1
        elif code[j] == ")":
            depth -= 1
        j += 1
    return j if depth == 0 else -1


def _simplify_ptr_deref(code: str) -> str:
    """Simplify expressions inside ``*(type *)(expr)`` pointer dereference patterns."""
    result: list[str] = []
    i = 0
    n = len(code)
    while i < n:
        m = _PTR_CAST_RE.match(code, i)
        if m:
            cast_end = m.end()
            # Skip optional whitespace after the cast
            ws_end = cast_end
            while ws_end < n and code[ws_end] in (" ", "\t"):
                ws_end += 1
            if ws_end < n and code[ws_end] == "(":
                paren_end = _find_matching_paren(code, ws_end)
                if paren_end > 0:
                    inner = code[ws_end + 1 : paren_end - 1]
                    # Only simplify if the inner expression contains arithmetic
                    # (skip things like function calls, simple variables, etc.)
                    stripped = inner.strip()
                    if stripped and any(c in stripped for c in "+-&|^"):
                        try:
                            simplified = _simplify_expr(stripped)
                            result.append(code[i:cast_end])
                            result.append(code[cast_end:ws_end])
                            result.append(f"({simplified})")
                            i = paren_end
                            continue
                        except Exception:
                            pass
            # No match or simplification failed, emit the cast as-is
            result.append(m.group())
            i = cast_end
        else:
            result.append(code[i])
            i += 1
    return "".join(result)


def simplify_index_brackets(code: str) -> str:
    """Simplify redundant parentheses in generated source code.

    Handles two patterns:
    1. ``[expr]`` — array index expressions
    2. ``*(type *)(expr)`` — pointer dereference with cast expressions

    If parsing of any individual expression fails, the original text is kept
    unchanged.
    """
    code = code.replace("((int)threadIdx.x)", "threadIdx.x")
    code = code.replace("((int)blockIdx.x)", "blockIdx.x")
    code = code.replace("((int)blockIdx.y)", "blockIdx.y")

    # Pass 1: simplify [...]
    result: list[str] = []
    i = 0
    n = len(code)
    while i < n:
        if code[i] == "[":
            depth = 1
            start = i
            j = i + 1
            while j < n and depth > 0:
                if code[j] == "[":
                    depth += 1
                elif code[j] == "]":
                    depth -= 1
                j += 1
            inner = code[start + 1 : j - 1].strip()
            if inner:
                try:
                    simplified = _simplify_expr(inner)
                    result.append(f"[{simplified}]")
                except Exception:
                    result.append(code[start:j])
            else:
                result.append(code[start:j])
            i = j
        else:
            result.append(code[i])
            i += 1
    code = "".join(result)

    # Pass 2: simplify *(type *)(expr)
    code = _simplify_ptr_deref(code)

    return code
