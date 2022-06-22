#!/usr/bin/env python3
from numbers import Number
from parser import Parser
from pprint import pprint
from typing import Optional

import click

from core import EPContext, ExceptionProcessor
from evaluator import evaluate
from semantics import check_semantics
from tokenizer import Tokenizer
from utils import (dump_tokens_to_stdout, ep_to_json, pretty_print_results,
                   print_ast)


@click.command(
    name="EP",
    help="EP(Eval Print) is a simple language which supports pure functions and variables",
)
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
) -> dict[str, Number | None | list[Number]]:
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
) -> dict[str, None | Number | list[Number]]:

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
        ep_to_json(results, out)
    if print_results:
        pretty_print_results(results)
    return results


if __name__ == "__main__":
    ep_entry()
