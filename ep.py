#!/usr/bin/env python3

import itertools
from pprint import pprint
from utils import print_ast
from exc_processor import ExceptionProcessor
from parser import Parser
from semantic_checker import SemanticChecker
from tokenizer import Tokenizer, Token, TokenType
from evaluator import evaluate
import click
from termcolor import colored


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
        source_code: str = f.read()
        processor = ExceptionProcessor(source_code, filename)
        tokens = Tokenizer(source_code, processor).get_tokens()
        if dump_tokens:
            tokens, tokens_copy = itertools.tee(tokens)
            dump_tokens_to_stdout(list(tokens_copy))
        parser = Parser(tokens, processor)
        if view_ast_dot:
            print_ast(parser.nodes)
        semantic_checker = SemanticChecker(parser.nodes, processor)
        values = evaluate(semantic_checker, parser.nodes)


if __name__ == "__main__":
    ep_entry()
