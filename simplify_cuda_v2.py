#!/usr/bin/env python3
"""
CUDA 代码括号简化器 v2 - 基于 AST 的正确算法
"""

from __future__ import annotations
import re
import sys
from dataclasses import dataclass
from typing import Union

# ============== AST 节点定义 ==============


@dataclass
class Identifier:
    name: str


@dataclass
class Number:
    value: str


@dataclass
class BinaryOp:
    op: str
    left: Expr
    right: Expr


@dataclass
class UnaryOp:
    op: str
    operand: Expr


@dataclass
class ArrayAccess:
    array: Expr
    index: Expr


@dataclass
class MemberAccess:
    obj: Expr
    member: str
    is_ptr: bool


@dataclass
class Cast:
    type_name: str
    operand: Expr


@dataclass
class FunctionCall:
    name: str
    args: list[Expr]


@dataclass
class TemplateCall:
    """模板函数调用 func<N>(args)"""

    name: str
    template_args: str
    args: list[Expr]


@dataclass
class Paren:
    """显式保留的括号（用于特殊情况）"""

    inner: Expr


Expr = Union[Identifier, Number, BinaryOp, UnaryOp, ArrayAccess, MemberAccess, Cast, FunctionCall, TemplateCall, Paren]


# ============== 运算符优先级 ==============

PRECEDENCE = {
    "||": 1,
    "&&": 2,
    "|": 3,
    "^": 4,
    "&": 5,
    "==": 6,
    "!=": 6,
    "<": 7,
    ">": 7,
    "<=": 7,
    ">=": 7,
    "<<": 8,
    ">>": 8,
    "+": 9,
    "-": 9,
    "*": 10,
    "/": 10,
    "%": 10,
}

LEFT_ASSOC = set(PRECEDENCE.keys())


# ============== 词法分析器 ==============

TOKEN_PATTERN = re.compile(
    r"""
    (0x[0-9a-fA-F]+[uUlL]*)                  # 十六进制
    |(\d+(?:\.\d+)?(?:e[+-]?\d+)?[fFlLuU]*)  # 数字
    |([a-zA-Z_]\w*)                          # 标识符
    |(>>|<<|<=|>=|==|!=|&&|\|\||->|::)       # 双字符运算符
    |([+\-*/%&|^<>!~])                       # 单字符运算符
    |([()\[\],.;:])                          # 括号和分隔符
    |(\s+)                                   # 空白
""",
    re.VERBOSE,
)


def tokenize(expr: str) -> list[str]:
    tokens = []
    pos = 0
    for m in TOKEN_PATTERN.finditer(expr):
        if m.start() != pos:
            # 有未匹配的字符
            tokens.append(expr[pos : m.start()])
        token = m.group(0)
        if not token.isspace():
            tokens.append(token)
        pos = m.end()
    if pos < len(expr):
        tokens.append(expr[pos:])
    return tokens


# ============== 语法分析器 ==============


class Parser:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset=0) -> str | None:
        idx = self.pos + offset
        if 0 <= idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def consume(self) -> str:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def expect(self, expected: str):
        if self.pos >= len(self.tokens):
            raise SyntaxError(f"Expected '{expected}', got EOF")
        token = self.consume()
        if token != expected:
            raise SyntaxError(f"Expected '{expected}', got '{token}'")
        return token

    def parse(self) -> Expr:
        return self.parse_expr(0)

    def parse_expr(self, min_prec: int) -> Expr:
        left = self.parse_unary()

        while True:
            op = self.peek()
            if op is None or op not in PRECEDENCE:
                break

            prec = PRECEDENCE[op]
            if prec < min_prec:
                break

            self.consume()
            right = self.parse_expr(prec + 1)
            left = BinaryOp(op, left, right)

        return left

    def parse_unary(self) -> Expr:
        op = self.peek()

        if op in ("-", "!", "~"):
            self.consume()
            operand = self.parse_unary()
            return UnaryOp(op, operand)

        # 取地址 & 或解引用 *
        if op == "&" or op == "*":
            # 判断是否是一元运算符
            # 如果前面没有操作数，或者前面是运算符/左括号，则是一元
            self.consume()
            operand = self.parse_unary()
            return UnaryOp(op, operand)

        # 类型转换 (type*)expr 或 (type)expr
        if op == "(":
            save_pos = self.pos
            self.consume()

            # 尝试解析类型名
            if self.peek() and re.match(r"^[a-zA-Z_]\w*$", self.peek()):
                type_name = self.consume()

                # 处理指针类型 type*
                while self.peek() == "*":
                    type_name += self.consume()

                if self.peek() == ")":
                    self.consume()
                    # 检查后面是否是表达式
                    next_tok = self.peek()
                    if next_tok and (next_tok[0].isalnum() or next_tok[0] == "_" or next_tok == "(" or next_tok == "&" or next_tok == "*"):
                        operand = self.parse_unary()
                        return Cast(type_name, operand)

            # 不是类型转换，回溯，作为括号表达式处理
            self.pos = save_pos

        return self.parse_postfix()

    def parse_postfix(self) -> Expr:
        expr = self.parse_primary()

        while True:
            op = self.peek()
            if op == "[":
                self.consume()
                index = self.parse_expr(0)
                self.expect("]")
                expr = ArrayAccess(expr, index)
            elif op == ".":
                self.consume()
                member = self.consume()
                expr = MemberAccess(expr, member, False)
            elif op == "->":
                self.consume()
                member = self.consume()
                expr = MemberAccess(expr, member, True)
            else:
                break

        return expr

    def parse_primary(self) -> Expr:
        token = self.peek()

        if token == "(":
            self.consume()
            expr = self.parse_expr(0)
            self.expect(")")
            return expr

        if token and (token[0].isdigit() or (token.startswith("0x"))):
            return Number(self.consume())

        if token and (token[0].isalpha() or token[0] == "_"):
            name = self.consume()

            # 命名空间 ::
            while self.peek() == "::":
                name += self.consume()
                if self.peek():
                    name += self.consume()

            # 模板参数 <...>
            template_args = None
            if self.peek() == "<":
                self.consume()
                template_content = []
                depth = 1
                while depth > 0 and self.peek():
                    t = self.consume()
                    if t == "<":
                        depth += 1
                    elif t == ">":
                        depth -= 1
                        if depth == 0:
                            break
                    template_content.append(t)
                template_args = "".join(template_content)

            # 函数调用 (...)
            if self.peek() == "(":
                self.consume()
                args = []
                if self.peek() != ")":
                    args.append(self.parse_expr(0))
                    while self.peek() == ",":
                        self.consume()
                        args.append(self.parse_expr(0))
                self.expect(")")

                if template_args is not None:
                    return TemplateCall(name, template_args, args)
                return FunctionCall(name, args)

            return Identifier(name)

        raise SyntaxError(f"Unexpected token: {token}")


# ============== 代码生成器 ==============


def generate(node: Expr, parent_op: str | None = None, is_right: bool = False) -> str:
    if isinstance(node, Identifier):
        return node.name

    if isinstance(node, Number):
        return node.value

    if isinstance(node, BinaryOp):
        left = generate(node.left, node.op, False)
        right = generate(node.right, node.op, True)
        result = f"{left} {node.op} {right}"

        if parent_op and parent_op in PRECEDENCE:
            my_prec = PRECEDENCE[node.op]
            parent_prec = PRECEDENCE[parent_op]

            need_parens = False
            if my_prec < parent_prec or my_prec == parent_prec and is_right and node.op in LEFT_ASSOC:
                need_parens = True

            if need_parens:
                return f"({result})"

        return result

    if isinstance(node, UnaryOp):
        operand = generate(node.operand, None, False)

        # 如果操作数是二元表达式，需要括号
        if isinstance(node.operand, BinaryOp):
            operand = f"({operand})"
        # 如果操作数也是一元运算符且是不同的运算符，可能需要括号
        elif isinstance(node.operand, UnaryOp) and node.op == node.operand.op:
            pass  # 相同一元运算符可以省略

        # 取地址运算符后的标识符不需要括号
        if node.op == "&" and isinstance(node.operand, (Identifier, MemberAccess, ArrayAccess)):
            return f"&{operand}"

        return f"{node.op}{operand}"

    if isinstance(node, ArrayAccess):
        array = generate(node.array)
        index = generate(node.index)
        return f"{array}[{index}]"

    if isinstance(node, MemberAccess):
        obj = generate(node.obj)
        op = "->" if node.is_ptr else "."
        return f"{obj}{op}{node.member}"

    if isinstance(node, Cast):
        operand = generate(node.operand)
        if isinstance(node.operand, BinaryOp):
            operand = f"({operand})"
        return f"({node.type_name}){operand}"

    if isinstance(node, FunctionCall):
        args = ", ".join(generate(arg) for arg in node.args)
        return f"{node.name}({args})"

    if isinstance(node, TemplateCall):
        args = ", ".join(generate(arg) for arg in node.args)
        return f"{node.name}<{node.template_args}>({args})"

    if isinstance(node, Paren):
        return f"({generate(node.inner)})"

    return str(node)


# ============== 表达式提取和替换 ==============


def find_bracket_expr(line: str) -> list[tuple]:
    """找出行中所有可能的括号表达式及其位置"""
    results = []
    i = 0
    while i < len(line):
        if line[i] == "[":
            # 找到 [ 对应的 ]
            depth = 1
            j = i + 1
            while j < len(line) and depth > 0:
                if line[j] == "[":
                    depth += 1
                elif line[j] == "]":
                    depth -= 1
                j += 1
            if depth == 0:
                content = line[i + 1 : j - 1]
                if "(" in content:  # 只处理包含括号的表达式
                    results.append((i + 1, j - 1, content, "["))
            i = j
        else:
            i += 1
    return results


def find_ptr_arithmetic(line: str) -> list[tuple]:
    """
    找出 ptr + (complex_expr) 中的 complex_expr
    关键：只替换括号内的内容，保留括号本身
    """
    results = []

    # 模式: + (expr)
    # 找到所有 + ( 的位置
    i = 0
    while i < len(line) - 1:
        # 找 + (
        if line[i] == "+" and i + 1 < len(line):
            # 跳过空格
            j = i + 1
            while j < len(line) and line[j] == " ":
                j += 1

            if j < len(line) and line[j] == "(":
                # 找到配对的 )
                paren_start = j
                depth = 1
                k = j + 1
                while k < len(line) and depth > 0:
                    if line[k] == "(":
                        depth += 1
                    elif line[k] == ")":
                        depth -= 1
                    k += 1

                if depth == 0:
                    paren_end = k - 1  # ) 的位置
                    content = line[paren_start + 1 : paren_end]

                    # 只处理复杂表达式（有多个嵌套括号）
                    if content.count("(") >= 3:
                        # 返回括号内部的位置，这样替换时保留括号
                        results.append((paren_start + 1, paren_end, content, "("))

                i = k
                continue
        i += 1

    return results


def simplify_expr(expr: str) -> str:
    """简化单个表达式"""
    try:
        tokens = tokenize(expr)
        if not tokens:
            return expr

        parser = Parser(tokens)
        ast = parser.parse()

        if parser.pos < len(tokens):
            return expr

        return generate(ast)
    except Exception:
        return expr


def simplify_line(line: str) -> str:
    """简化一行代码中的括号"""
    if line.count("(") + line.count(")") < 5:
        return line

    stripped = line.strip()
    if stripped.startswith("#") or stripped.startswith("//"):
        return line

    # 找到所有需要简化的表达式
    all_exprs = []
    all_exprs.extend(find_bracket_expr(line))
    all_exprs.extend(find_ptr_arithmetic(line))

    # 按起始位置排序
    all_exprs.sort(key=lambda x: x[0])

    # 过滤掉重叠的范围（只保留不重叠的）
    non_overlapping = []
    last_end = -1
    for item in all_exprs:
        start, end, content, bracket_type = item
        if start >= last_end:  # 不重叠
            non_overlapping.append(item)
            last_end = end

    # 从后往前替换，避免位置偏移
    result = line
    for item in reversed(non_overlapping):
        start, end, content, bracket_type = item
        simplified = simplify_expr(content)
        result = result[:start] + simplified + result[end:]

    return result


def process_file(input_path: str, output_path: str = None):
    """处理整个文件"""
    with open(input_path) as f:
        lines = f.readlines()

    result = []
    for i, line in enumerate(lines):
        try:
            simplified = simplify_line(line.rstrip("\n"))
            result.append(simplified)
        except Exception as e:
            print(f"Warning: Line {i + 1} 处理失败: {e}", file=sys.stderr)
            result.append(line.rstrip("\n"))

    output = "\n".join(result) + "\n"

    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
        print(f"已保存到 {output_path}")
    else:
        print(output)

    return output


if __name__ == "__main__":
    args = sys.argv[1]
    print(sys.argv[1])
    input_file = f"/root/.tilelang/cache/{args}/host_kernel.cu"
    output_file = f"/root/.tilelang/cache/{args}/host_kernel_simplified_v2.cu"

    process_file(input_file, output_file)
