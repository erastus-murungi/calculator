import re
from typing import Optional

from termcolor import colored

from ast_nodes import Node
from semantic_checker import SemanticChecker
from tokenizer import Name


def evaluate(
    semantic_checker: SemanticChecker, nodes: list[Node]
) -> list[Optional[float]]:
    results = []
    values = {}
    for node in nodes:
        node.evaluate(
            semantic_checker.node_types,
            semantic_checker.node_to_env[node],
            semantic_checker.exception_processor,
            semantic_checker.exceptions,
            values,
        )
        results.append(values[node])

    lines = []
    for node in nodes:
        lines.append(
            format_line(semantic_checker.exception_processor.lines[node.pos.line])
        )
    for node, line in zip(nodes, lines):
        val = values[node]
        print(f"In [{colored(str(node.pos.line), 'green', attrs=['bold'])}]: {line}")
        if val:
            print(
                f"Out[{colored(str(node.pos.line), 'magenta', attrs=['bold'])}]: => {colored(str(val), 'blue', attrs=['bold'])}"
            )

    return results


def format_line(line):
    def color_name(matchobj):
        text = matchobj.group(0)
        if text == "let" or text == "func":
            return colored(matchobj.group(0), "green", attrs=["bold"])
        return matchobj.group(0)

    line = re.sub(Name, color_name, line)
    return line
