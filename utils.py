import collections
import json
import re
import subprocess
import sys
import typing
from numbers import Number
from typing import Optional

from termcolor import colored

from core import Node, Operator
from tokenizer import Name, Token, TokenType

FILENAME = "ast"
AST_DOT_FILEPATH = FILENAME + "." + "dot"
AST_GRAPH_TYPE = "pdf"
AST_OUTPUT_FILENAME = FILENAME + "." + AST_GRAPH_TYPE


def escape(s: str):
    return (
        s.replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\b", "\\b")
        .replace("\r", "\\r")
        .replace("\f", "\\f")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("<", "\\<")
        .replace(">", "\\>")
        .replace("\n", "\\l")
        .replace("||", "\\|\\|")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def graph_prologue():
    return (
        'digraph G {  graph [fontname = "Courier New"];\n'
        + ' node [fontname = "Courier", style = rounded];\n'
        + ' edge [fontname = "Courier"];'
    )


def graph_epilogue():
    return "}"


def print_ast(ast_nodes: list[Node]):
    graph = [graph_prologue()]
    seen = set()
    for line_num, ast_node in enumerate(ast_nodes):
        name = f"line_{line_num}"
        graph.append(f"subgraph cluster_{escape(name)} {{ \n label = {escape(name)}")
        edges = []
        stack = []
        nodes = []
        stack.append(ast_node)
        seen.add(ast_node)

        while stack:
            node = stack.pop()
            if isinstance(node, Operator):
                nodes.append(
                    f'   {id(node)} [shape=doublecircle, style=filled, fillcolor=black, fontcolor=white, label="{escape(node.source())}"];'
                )
            elif node.is_terminal():
                nodes.append(
                    f'   {id(node)} [shape=oval, label="{escape(node.source())}"];'
                )
            else:
                nodes.append(
                    f'   {id(node)} [shape=record, style=filled, fillcolor=white, label="<from_node>{escape(node.source())}"];'
                )

            for child in node.children():
                edges.append(f"    {id(node)}:from_false -> {id(child)}:from_node;")
                if child not in seen:
                    stack.append(child)

        graph.extend(edges)
        graph.extend(nodes)
        graph.append(graph_epilogue())
    graph.append(graph_epilogue())
    with open(AST_DOT_FILEPATH, "w") as f:
        f.write("\n".join(graph))

    create_graph_pdf()


def create_graph_pdf(
    dot_filepath=AST_DOT_FILEPATH,
    output_filepath=AST_OUTPUT_FILENAME,
    output_filetype=AST_GRAPH_TYPE,
):
    dot_exec_filepath = (
        "/usr/local/bin/dot" if sys.platform == "darwin" else "/usr/bin/dot"
    )
    args = [
        dot_exec_filepath,
        f"-T{output_filetype}",
        f"-Gdpi={96}",
        dot_filepath,
        "-o",
        output_filepath,
    ]
    subprocess.run(args)
    subprocess.run(["open", output_filepath])
    subprocess.run(["rm", AST_DOT_FILEPATH])


def pretty_print_results(results: dict[str, None | Number | list[Number]]) -> None:
    lines = [format_line(line) for line in results]
    for line_num, (line, result) in enumerate(zip(lines, results.values())):
        print(f"eval ({colored(str(line_num), 'green', attrs=['bold'])})> {line}")
        if result:
            print(f"=> {colored(str(result), 'blue', attrs=['bold'])}")


def format_line(line):
    def color_name(matchobj):
        text = matchobj.group(0)
        if text in ("let", "in", "complex", "def", "return", "const"):
            return colored(matchobj.group(0), "green", attrs=["bold"])
        return matchobj.group(0)

    line = re.sub(Name, color_name, line)
    return line


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
            # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def ep_to_json(results: dict[str, None | Number | list[Number]], out: str):
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4, cls=ComplexEncoder)


def dump_tokens_to_stdout(tokens: tuple[Token, ...]) -> None:
    for token in tokens:
        if (
            token.token_type == TokenType.WHITESPACE
            or token.token_type == TokenType.NEWLINE
        ):
            continue
        print(
            f"{colored(token.token_type.name.lower(), 'magenta')}\t\t{token.lexeme}\t\tLoc={token.loc}"
        )
