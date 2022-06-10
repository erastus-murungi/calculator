import itertools
from pprint import pprint

import utils
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
@click.argument("filename")
def ep_entry(filename: str, dump_tokens: bool):
    with open(filename, "r") as f:
        source_code: str = f.read()
        processor = ExceptionProcessor(source_code, filename)
        tokens = Tokenizer(source_code, processor)._tokenize()
        if dump_tokens:
            tokens, tokens_copy = itertools.tee(tokens)
            dump_tokens_to_stdout(list(tokens_copy))

        parser = Parser(tokens, processor)
        utils.print_ast(parser.nodes)


if __name__ == "__main__":
    ep_entry()
    # processor = ExceptionProcessor(to_eval)
    # tokenizer = Tokenizer(to_eval, processor)
    # parser = Parser(tokenizer.tokenize(), processor)
    # semantic_checker = SemanticChecker(parser.root, processor)
    # values = evaluate(semantic_checker, parser.root)

    # pprint(tuple(semantic_checker.node_to_env.items()))
    # pprint(parser.root)
    # pprint(semantic_checker.node_types)
