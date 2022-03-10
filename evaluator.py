import re

from ast_nodes import Node, Store
from semantic_checker import SemanticChecker
from termcolor import colored

from tokenizer import Floatnumber, Name, Intnumber, group


def evaluate(semantic_checker: SemanticChecker, nodes: list[Node]):
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
    max_length = max(map(len, lines))
    for node, line in zip(nodes, lines):
        val = values[node]
        print(
            f"{node.pos.line:<3}{line.ljust(max_length): >10} {colored('' if val is None else val, 'green')}"
        )

    return results


def format_line(line):
    def color_num(matchobj):
        num = matchobj.group(0)
        if '.' in num:
            pass
        return colored(matchobj.group(0), "yellow")

    def color_name(matchobj):
        if not matchobj.group(0)[0].isdigit():
            if (matchobj.group(0)) != "let":
                return colored(matchobj.group(0), "blue")
            else:
                return colored(matchobj.group(0), "magenta")
        return matchobj.group(0)

    line = re.sub(group(Intnumber, Floatnumber), color_num, line)
    line = re.sub(Name, color_name, line)
    return line
