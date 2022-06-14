import pprint
from random import randint, choice, random
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
    depth: int = 1, allowed_symbols: tuple[str] = (), grouping_prob: float = 0.3
) -> str:
    if depth == 0:
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


def test_expressions(n_expressions: int = 10) -> dict[str, str]:
    expr_to_result = {}
    n_valid = 0
    while n_valid < n_expressions:
        try:
            random_expr = generate_random_arithmetic_expression(2)
            res = eval(random_expr.replace("^", "**"))
            expr_to_result[random_expr] = res
            n_valid += 1
        except Exception as e:
            print(e)
            continue
    return expr_to_result


if __name__ == "__main__":
    # expr = generate_random_arithmetic_expression(depth=5)
    # print(expr)
    # print(eval(expr.replace("^", "**")))
    expressions = test_expressions()
    values = _ep_entry("\n".join(expressions.keys()), False, False)
    expected_values = expressions.values()
    print(values.values(), expected_values)
