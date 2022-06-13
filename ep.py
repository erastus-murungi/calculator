#!/usr/bin/env python3

import click
from termcolor import colored

from core import EPContext, ExceptionProcessor
from evaluator import evaluate
from parser import Parser
from semantics import check_semantics
from tokenizer import Tokenizer, Token, TokenType
from utils import print_ast


def dump_tokens_to_stdout(tokens: list[Token]) -> None:
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
    is_flag="True",
    default=False,
    help="Tokenize source code and print out the tokens",
)
@click.option(
    "--view-ast-dot",
    "-a",
    is_flag="True",
    default=False,
    help="Generate a dot file and convert to a pdf with AST graphs",
)
@click.argument("filename")
def ep_entry(filename: str, dump_tokens: bool, view_ast_dot: bool) -> None:
    with open(filename, "r") as f:
        ctx = EPContext()
        source_code: str = f.read()
        ctx.set_source_code(source_code)
        ctx.set_exception_processor(ExceptionProcessor(source_code, filename))
        tokenizer = Tokenizer(ctx)
        if dump_tokens:
            dump_tokens_to_stdout(list(tokenizer.get_tokens()))
        parser = Parser(tokenizer.get_tokens(), ctx)
        if view_ast_dot:
            print_ast(parser.nodes)
        check_semantics(parser.nodes, ctx)
        values = evaluate(parser.nodes, ctx)


if __name__ == "__main__":
    ep_entry()
