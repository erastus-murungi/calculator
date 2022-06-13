import re
from typing import Optional
from termcolor import colored
from core import Node, EPContext
from tokenizer import Name


def evaluate(nodes: list[Node], ep_context: EPContext) -> list[Optional[float]]:
    results = []
    values = {}
    ep_context.set_node_to_value_mapping(values)
    for node in nodes:
        try:
            node.evaluate(ep_context)
            results.append(values[node])
        except KeyError as e:
            raise ValueError("literal not found in scope") from e

    lines = []
    for node in nodes:
        lines.append(
            format_line(ep_context.get_exception_processor().lines[node.pos.line])
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
        if text == "let" or text == "func" or text == "complex":
            return colored(matchobj.group(0), "green", attrs=["bold"])
        return matchobj.group(0)

    line = re.sub(Name, color_name, line)
    return line
