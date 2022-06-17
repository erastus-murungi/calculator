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
    ep_context.set_state(State.EVALUATION_COMPLETE)
    return {node.source(): values[node] for node in nodes}


def pretty_print_results(line_to_result: dict[str, Optional[Number]]) -> None:
    lines = [format_line(line) for line in line_to_result]
    for line_num, (line, result) in enumerate(zip(lines, line_to_result.values())):
        print(f"eval ({colored(str(line_num), 'green', attrs=['bold'])})> {line}")
        if result:
            print(f"=> {colored(str(result), 'blue', attrs=['bold'])}")


def format_line(line):
    def color_name(matchobj):
        text = matchobj.group(0)
        if text == "let" or text == "func" or text == "complex":
            return colored(matchobj.group(0), "green", attrs=["bold"])
        return matchobj.group(0)

    line = re.sub(Name, color_name, line)
    return line
