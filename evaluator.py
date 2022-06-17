import re
from numbers import Number
from typing import Optional

from termcolor import colored

from core import EPContext, Node, State
from tokenizer import Name


def evaluate(nodes: list[Node], ep_context: EPContext) -> dict[str, Optional[Number]]:
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
    ep_context.set_state(State.EVALUATION_COMPLETE)
    return {node.source(): values[node] for node, line in zip(nodes, lines)}


def format_line(line):
    def color_name(matchobj):
        text = matchobj.group(0)
        if text == "let" or text == "func" or text == "complex":
            return colored(matchobj.group(0), "green", attrs=["bold"])
        return matchobj.group(0)

    line = re.sub(Name, color_name, line)
    return line
