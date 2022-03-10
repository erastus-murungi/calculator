from exc_processor import ExceptionProcessor
from parser import Parser
from semantic_checker import SemanticChecker
from tokenizer import Tokenizer
from evaluator import evaluate

if __name__ == "__main__":
    to_eval = (
        "let a := 10.3\n"
        "let c := 15.4\n"
        "1944.66 + c * 3.4 + 1.4 + 2.3 + 4.5 \n"
        "(-2.0) + a \n"
        "let b := 10.2\n"
        "a + b\n"
        "let d := 10\n"
        "d + 30"
    )
    processor = ExceptionProcessor(to_eval)
    tokenizer = Tokenizer(to_eval, processor)
    parser = Parser(tokenizer.tokenize(), processor)
    semantic_checker = SemanticChecker(parser.root, processor)
    values = evaluate(semantic_checker, parser.root)

    # pprint(tuple(semantic_checker.node_to_env.items()))
    # pprint(parser.root)
    # pprint(semantic_checker.node_types)
