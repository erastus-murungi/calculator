import sys
from random import choice, randint, random
from typing import Final

from ep import _ep_entry

ops: Final[tuple[str, ...]] = ("*", "+", "^", "/", "//", "-")


def skewed_ops_choice(skew_prob=0.01):
    """Choose ^ with a lower prob"""
    op = choice(ops)
    if op == "^":
        if random() < skew_prob:
            return op
    return op


def generate_random_arithmetic_expression(
    depth: int = 1,
    allowed_symbols: tuple[str] = (),
    grouping_prob: float = 0.3,
    unary_prob=0.1,
) -> str:
    if depth == 0:
        if random() < unary_prob:
            return f"-{randint(1, 5)}"
        return str(randint(1, 4))
    else:
        lhs = generate_random_arithmetic_expression(
            depth - 1, allowed_symbols, grouping_prob
        )
        rhs = generate_random_arithmetic_expression(
            depth - 1, allowed_symbols, grouping_prob
        )
        if random() < grouping_prob:
            return f"({lhs} {skewed_ops_choice()} {rhs})"
        return f"{lhs} {skewed_ops_choice()} {rhs}"


def test_expressions(n_expressions: int = 10, depth=2) -> dict[str, str]:
    expr_to_result = {}
    n_valid = 0
    while n_valid < n_expressions:
        try:
            random_expr = generate_random_arithmetic_expression(depth)
            res = eval(random_expr.replace("^", "**"))
            expr_to_result[random_expr] = res
            n_valid += 1
        except Exception as e:
            print(e)
            continue
    return expr_to_result


def to_num(val_str):
    try:
        return complex(val_str)
    except TypeError:
        return float(val_str)


def test_equality(from_python, from_ep):
    for index_of_line, (line, val_from_python, val_from_ep) in enumerate(
        zip(from_python.keys(), from_python.values(), from_ep.values())
    ):
        if to_num(val_from_ep) != to_num(val_from_python):
            print(
                f"different eval results for line {index_of_line} {line}\n"
                f"from python = {val_from_python}\n"
                f"from ep     = {val_from_ep}"
            )
            sys.exit(1)


if __name__ == "__main__":
    # expr = generate_random_arithmetic_expression(depth=5)
    # print(expr)
    # print(eval(expr.replace("^", "**")))

    # _from_python = test_expressions(depth=2)
    # _from_ep = _ep_entry("\n".join(_from_python.keys()), False, False)
    # test_equality(_from_python, _from_ep)

    # _ep_entry("(4 * 3) - 3 ^ 2", False, True)
    # print("correct =>", eval("(4 * 3) - 3 ^ 2".replace("^", "**")))

    _ep_entry("c_cos(complex(10, 17))", False, False)
    #
    # _ep_entry("-3 * -2", False, False)
    # print("correct =>", eval("-3 * -2".replace("^", "**")))
