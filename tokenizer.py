from dataclasses import dataclass
from enum import Enum
from typing import Optional, Iterator, Final
import re

from exc_processor import Loc, ExceptionProcessor


class NoMatchFound(Exception):
    pass


LET = "let"  # define a binding
L_PAR = "("
R_PAR = ")"

# arithmetic operators
ADD = "+"
SUBTRACT = "-"
MULTIPLY = "*"
DIVIDE = "/"
MODULUS = "%"
EXPONENT = "^"
COMMA = ","
COMPLEX = "complex"  # used to define a complex number
F_DIV = "//"
DEFINE = ":="
FUNCTION = "func"
NEWLINE = "\n"
CONTINUATION = "\\n"


class TokenType(Enum):
    LET = "bind"  # define a binding
    L_PAR = "("
    R_PAR = ")"

    # arithmetic operators
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    MODULUS = "%"
    EXPONENT = "^"
    COMMA = ","
    COMPLEX = "complex"  # used to define a complex number
    F_DIV = "//"
    DEFINE = ":="
    FUNCTION = "func"
    NEWLINE = "\n"
    CONTINUATION = "\\n"
    ID = "identifier"
    FLOAT = "float"
    INT = "integer"
    WHITESPACE = "whitespace"
    BOGUS = "undefined"
    EOF = "EOF"

    def __repr__(self):
        return self.name


@dataclass
class Token:
    """
    A token has three components:
    1) Its type
    2) A lexeme -- the substring of the source code it represents
    3) The location in code of the lexeme
    """

    token_type: TokenType
    lexeme: str
    pos: Loc


# A dummy token with its own bogus type and its own bogus location
INVALID_TOKEN: Final[Token] = Token(TokenType.BOGUS, "", Loc("", -1, -1, -1))


def group(*choices):
    return "(" + "|".join(choices) + ")"


def reg_any(*choices):
    return group(*choices) + "*"


def maybe(*choices):
    return group(*choices) + "?"


# Regular expressions used to parse numbers
Hexnumber = r"0[xX](?:_?[0-9a-fA-F])+"
Binnumber = r"0[bB](?:_?[01])+"
Octnumber = r"0[oO](?:_?[0-7])+"
Decnumber = r"(?:0(?:_?0)*|[1-9](?:_?[0-9])*)"
Intnumber = group(Hexnumber, Binnumber, Octnumber, Decnumber)
Exponent = r"[eE][-+]?[0-9](?:_?[0-9])*"
Pointfloat = group(
    r"[0-9](?:_?[0-9])*\.(?:[0-9](?:_?[0-9])*)?", r"\.[0-9](?:_?[0-9])*"
) + maybe(Exponent)
Expfloat = r"[0-9](?:_?[0-9])*" + Exponent
Floatnumber = group(Pointfloat, Expfloat)
Imagnumber = group(r"[0-9](?:_?[0-9])*[jJ]", Floatnumber + r"[jJ]")
Name = r"\w+"


class Tokenizer:
    def __init__(self, string: str, exception_processor: ExceptionProcessor):
        self.exception_processor = exception_processor
        self.string = string
        self.line = 0
        self.column = 0
        self.offset = 0

    def to_next_char(self):
        self.offset += 1
        self.column += 1

    def skip_n_chars(self, n):
        self.offset += n
        self.column += n

    def match_number(self, number_regex: str) -> Optional[tuple[str, TokenType]]:
        match = re.match("^" + number_regex, self.string[self.offset :])
        ret_type = None
        lexeme = ""
        if match is not None:
            lexeme, ret_type = (
                match.group(0),
                TokenType.FLOAT
                if number_regex == Floatnumber
                else TokenType.INT
                if number_regex == Intnumber
                else TokenType.COMPLEX
                if number_regex == Imagnumber
                else None,
            )
        if ret_type is None:
            return None
        else:
            return lexeme, ret_type

    def match_identifier(self) -> Optional[tuple[str, TokenType]]:
        re_match = re.match("^" + Name, self.string[self.offset :])
        if re_match is not None:
            re_match = re_match.group(0)
            if re_match == LET:
                return LET, TokenType.LET
            if re_match == FUNCTION:
                return FUNCTION, TokenType.FUNCTION
            return re_match, TokenType.ID
        return None

    def get_identifier(self, pos) -> Token:
        ret = (
            self.match_number(Imagnumber)
            or self.match_number(Floatnumber)
            or self.match_number(Intnumber)
            or self.match_identifier()
        )
        if ret is None:
            raise NoMatchFound
        else:
            lexeme, ret_type = ret
            self.skip_n_chars(len(lexeme) - 1)
            return Token(ret_type, lexeme, pos)

    def current_char(self):
        return self.string[self.offset]

    def remaining_untokenized_code(self):
        return self.string[self.offset :]

    def tokenize(self) -> Iterator[Token]:
        filename = self.exception_processor.filename
        while self.offset < len(self.string):
            pos = Loc(filename, self.line, self.column, self.offset)
            if self.remaining_untokenized_code().startswith(F_DIV):
                token = Token(TokenType.F_DIV, F_DIV, pos)
                self.to_next_char()
            elif self.remaining_untokenized_code().startswith(DEFINE):
                token = Token(TokenType.DEFINE, DEFINE, pos)
                self.to_next_char()
            elif self.remaining_untokenized_code().startswith(CONTINUATION):
                token = Token(TokenType.WHITESPACE, CONTINUATION, pos)
                self.to_next_char()
            elif self.current_char() != NEWLINE and self.current_char().isspace():
                token = Token(TokenType.WHITESPACE, self.current_char(), pos)
            else:
                try:
                    token = self.get_identifier(pos)
                except NoMatchFound:
                    match TokenType(self.current_char()):
                        case TokenType.L_PAR:
                            token = Token(TokenType.L_PAR, L_PAR, pos)
                        case TokenType.R_PAR:
                            token = Token(TokenType.R_PAR, R_PAR, pos)
                        case TokenType.ADD:
                            token = Token(TokenType.ADD, ADD, pos)
                        case TokenType.SUBTRACT:
                            token = Token(TokenType.SUBTRACT, SUBTRACT, pos)
                        case TokenType.MODULUS:
                            token = Token(TokenType.MODULUS, MODULUS, pos)
                        case TokenType.DIVIDE:
                            token = Token(TokenType.DIVIDE, DIVIDE, pos)
                        case TokenType.MULTIPLY:
                            token = Token(TokenType.MULTIPLY, MULTIPLY, pos)
                        case TokenType.EXPONENT:
                            token = Token(TokenType.EXPONENT, EXPONENT, pos)
                        case TokenType.COMMA:
                            token = Token(TokenType.COMMA, COMMA, pos)
                        case TokenType.NEWLINE:
                            token = Token(TokenType.NEWLINE, NEWLINE, pos)
                            self.line += 1
                            self.column = -1
                        case _:
                            raise NoMatchFound
            yield token
            self.to_next_char()
        yield Token(
            TokenType.EOF,
            "",
            Loc(filename, self.line, self.column, self.offset),
        )
