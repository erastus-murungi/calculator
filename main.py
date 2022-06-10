from pprint import pprint

from exc_processor import ExceptionProcessor
from parser import Parser
from semantic_checker import SemanticChecker
from tokenizer import Tokenizer, Token
from evaluator import evaluate
import click
from termcolor import colored


def dump_tokens_to_stdout(tokens: list[Token]) -> None:
    for token in tokens:
        print(f"{colored(token.token_type.name, 'magenta')}\t{token.lexeme}\t\tLoc={token.pos}")


@click.command(name="Calc", help="calculator")
@click.option(
    "--dump-tokens",
    "-d",
    is_flag="True",
    default=False,
    help="Tokenize source code and print out the tokens",
)
@click.argument("filename")
def main(filename: str, dump_tokens: bool):
    with open(filename, "r") as f:
        source_code: str = f.read()
        processor = ExceptionProcessor(source_code, filename)
        tokens = Tokenizer(source_code, processor).tokenize()
        if dump_tokens:
            dump_tokens_to_stdout(list(tokens))


if __name__ == "__main__":
    main()
    # processor = ExceptionProcessor(to_eval)
    # tokenizer = Tokenizer(to_eval, processor)
    # parser = Parser(tokenizer.tokenize(), processor)
    # semantic_checker = SemanticChecker(parser.root, processor)
    # values = evaluate(semantic_checker, parser.root)

    # pprint(tuple(semantic_checker.node_to_env.items()))
    # pprint(parser.root)
    # pprint(semantic_checker.node_types)
