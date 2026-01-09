#!/usr/bin/env python3
"""
简单的括号简化脚本 - 根据 C++ 运算符优先级去除多余括号

用法:
    python simplify_parens.py input.cu
    python simplify_parens.py input.cu -o output.cu
"""

import re
import sys

# C++ 运算符优先级 (数字越大优先级越高)
PRECEDENCE = {
    "||": 4,
    "&&": 5,
    "|": 6,
    "^": 7,
    "&": 8,
    "==": 9,
    "!=": 9,
    "<": 10,
    "<=": 10,
    ">": 10,
    ">=": 10,
    "<<": 11,
    ">>": 11,
    "+": 12,
    "-": 12,
    "*": 13,
    "/": 13,
    "%": 13,
}

# 常见的 C/C++ 类型关键字
TYPE_KEYWORDS = {
    "int",
    "float",
    "double",
    "char",
    "short",
    "long",
    "unsigned",
    "signed",
    "void",
    "bool",
    "size_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "half",
    "half_t",
    "float16_t",
    "uchar",
    "uint",
    "ulong",
    "ushort",
}


def find_matching_paren(s, start):
    """找到与 start 位置的 '(' 匹配的 ')' 位置"""
    depth = 1
    i = start + 1
    while i < len(s) and depth > 0:
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
        i += 1
    return i - 1 if depth == 0 else -1


def is_type_cast(inner):
    """检查字符串是否是类型转换，如 int, half_t*, unsigned int"""
    inner = inner.strip()
    # 去掉指针符号
    inner = inner.rstrip("* ")
    # 检查是否是类型关键字
    if inner in TYPE_KEYWORDS:
        return True
    # 检查是否是类型名模式 (以 _t 结尾，或者像 half_t 这样)
    return re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*_t$", inner)
    #     return True
    # # TODO: 处理更复杂的类型如 const char*, unsigned long long 等
    # return False


def is_followed_by_operand(s, pos):
    """检查 pos 位置后面是否跟着操作数（用于判断是否是类型转换）"""
    rest = s[pos:].lstrip()
    # 如果后面是标识符、数字、或左括号，可能是类型转换
    return rest and (rest[0].isalnum() or rest[0] == "_" or rest[0] == "(")
    #     return True
    # return False


def simplify_once(s):
    """单次遍历，去除一层多余括号"""
    result = []
    i = 0
    changed = False

    while i < len(s):
        if s[i] != "(":
            result.append(s[i])
            i += 1
            continue

        # 找到匹配的右括号
        outer_end = find_matching_paren(s, i)
        if outer_end == -1:
            result.append(s[i])
            i += 1
            continue

        inner = s[i + 1 : outer_end]

        # 检查是否是类型转换 (int), (half_t*) 等
        if is_type_cast(inner):
            # 保留类型转换括号
            result.append(s[i : outer_end + 1])
            i = outer_end + 1
            continue

        # 情况1: ((x)) -> (x)  双层括号
        if inner.startswith("(") and inner.endswith(")"):
            inner_paren_end = find_matching_paren(inner, 0)
            if inner_paren_end == len(inner) - 1:
                # 整个内部就是一个括号表达式，去掉外层
                result.append(inner)
                i = outer_end + 1
                changed = True
                continue

        # 情况2: 括号包裹的是简单变量，如 (x), (threadIdx)
        # 但要排除类型转换后面的情况 (int)(x)
        inner_stripped = inner.strip()
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", inner_stripped):
            # 检查前面是否是类型转换 (type)(x) 的情况
            before = "".join(result).rstrip()
            if before.endswith(")"):
                # 可能是 (int)(x)，保留
                result.append(s[i : outer_end + 1])
                i = outer_end + 1
                continue
            # 不是类型转换，可以去掉括号
            result.append(inner)
            i = outer_end + 1
            changed = True
            continue

        # 情况3: 括号包裹的是数字字面量 (123)
        if re.match(r"^[0-9]+\.?[0-9]*[fFlL]?$", inner_stripped):
            before = "".join(result).rstrip()
            if before.endswith(")"):
                # 可能是 (float)(1.0)，保留
                result.append(s[i : outer_end + 1])
                i = outer_end + 1
                continue
            result.append(inner)
            i = outer_end + 1
            changed = True
            continue

        # 其他情况：保留括号
        result.append(s[i])
        i += 1

    return "".join(result), changed


def simplify(code):
    """反复简化直到没有变化"""
    prev = None
    while code != prev:
        prev = code
        code, _ = simplify_once(code)
    return code


def main():
    if len(sys.argv) < 2:
        # 交互模式：测试用例
        tests = [
            "((2 + 3))",
            "((x))",
            "(((int)threadIdx.x))",
            "((int)threadIdx.x)",
            "(int)threadIdx.x",
            "((a * b) + (c * d))",
            "(((int)threadIdx.x) >> 2)",
            "((((int)threadIdx.x) & 3) >> 1)",
            "((((int)threadIdx.x) >> 2) * 32)",
            "(((((int)threadIdx.x) & 31) >> 4))",
            "((half_t*)buf)",
        ]
        print("测试用例:")
        for t in tests:
            result = simplify(t)
            status = "✓" if result != t else "="
            print(f"  {status} {t:45} => {result}")
        return

    # 文件模式
    with open(sys.argv[1]) as f:
        code = f.read().replace("((int)threadIdx.x)", "threadIdx.x")

    result = simplify(code)

    if len(sys.argv) >= 4 and sys.argv[2] == "-o":
        with open(sys.argv[3], "w") as f:
            f.write(result)
        print(f"已写入 {sys.argv[3]}")
    else:
        print(result)


if __name__ == "__main__":
    main()
