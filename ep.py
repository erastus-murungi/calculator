#!/usr/bin/env python3
import json
from numbers import Number
from parser import Parser
from typing import Optional

import click
from termcolor import colored

from core import EPContext, ExceptionProcessor
from evaluator import evaluate, pretty_print_results
from semantics import check_semantics
from tokenizer import Token, Tokenizer, TokenType
from utils import print_ast


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
            # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def dump_tokens_to_stdout(tokens: tuple[Token]) -> None:
    for token in tokens:
        if (
            token.token_type == TokenType.WHITESPACE
            or token.token_type == TokenType.NEWLINE
        ):
            continue
        print(
            f"{colored(token.token_type.name.lower(), 'magenta')}\t\t{token.lexeme}\t\tLoc={token.loc}"
        )


@click.command(name="EP", help="EP stands for Eval Print")
@click.option(
    "--dump-tokens",
    "-d",
    is_flag=True,
    default=False,
    help="Tokenize source code and print out the tokens",
)
@click.option(
    "--view-ast-dot",
    "-a",
    is_flag=True,
    default=False,
    help="Generate a dot file and convert to a pdf with AST graphs",
)
@click.option(
    "--print-results",
    "-p",
    is_flag=True,
    default=False,
    help="Print results to stdout",
)
@click.option(
    "--out",
    is_flag=False,
    type=click.STRING,
    default=None,
    help="Relative path to write results as json",
)
@click.argument("file", type=click.File("r"))
def ep_entry(
    file, dump_tokens: bool, view_ast_dot: bool, out: str, print_results: bool
) -> dict[str, Number | None]:
    return _ep_entry(
        file.read(), dump_tokens, view_ast_dot, out, print_results, file.name
    )


def _ep_entry(
    source_code: str,
    dump_tokens: bool,
    view_ast_dot: bool,
    out: Optional[str],
    print_results: bool,
    filename: str = "",
) -> dict[str, Number | None]:
    ctx = EPContext()
    ctx.set_source_code(source_code)
    ctx.set_exception_processor(ExceptionProcessor(source_code, filename))
    tokenizer = Tokenizer(ctx)
    if dump_tokens:
        dump_tokens_to_stdout(tuple(tokenizer.get_tokens()))
    parser = Parser(tokenizer.get_tokens(), ctx)
    if view_ast_dot:
        print_ast(parser.nodes)
    check_semantics(parser.nodes, ctx)
    results = evaluate(parser.nodes, ctx)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4, cls=ComplexEncoder)
    if print_results:
        pretty_print_results(results)
    return results


if __name__ == "__main__":
    ep_entry()
