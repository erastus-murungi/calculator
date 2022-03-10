from dataclasses import dataclass
from enum import Enum
from typing import Optional, Iterator
import re

from exc_processor import Pos, ExceptionProcessor


class NoMatchFound(Exception):
    pass


LET = "let"
L_PAR = "("
R_PAR = ")"
SUBTRACT = "-"
ADD = "+"
DIVIDE = "/"
MODULUS = "%"
EXPONENT = "^"
FLOAT = "float"
INT = "integer"
COMPLEX = "complex"
MULTIPLY = "*"
F_DIV = "//"
DEFINE = ":="
HEX = "hex"
ABS = "abs"
ID = "id"
INF = "inf"
NAN = "nan"
NEWLINE = "\n"
CONTINUATION = "\\n"


class TokenType(Enum):
    LET = "let"
    L_PAR = "("
    R_PAR = ")"
    SUBTRACT = "-"
    ADD = "+"
    DIVIDE = "/"
    MODULUS = "%"
    EXPONENT = "^"
    FLOAT = "float"
    INT = "integer"
    HEX = "hex"
    COMPLEX = "complex"
    MULTIPLY = "*"
    F_DIV = "//"
    DEFINE = ":="
    WHITESPACE = "whitespace"
    NEWLINE = "\n"
    UNDEFINED = "undefined"
    ABS = "abs"
    CONJUGATE = "conjugate"
    ID = "id"
    EOF = "EOF"

    def __repr__(self):
        return self.name


@dataclass
class Token:
    token_type: TokenType
    lexeme: str
    pos: Pos


INVALID_TOKEN = Token(TokenType.UNDEFINED, "", None)


def group(*choices):
    return "(" + "|".join(choices) + ")"


def any(*choices):
    return group(*choices) + "*"


def maybe(*choices):
    return group(*choices) + "?"


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

    def increment(self):
        self.offset += 1
        self.column += 1

    def increment_by(self, n):
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
            if re_match == ABS:
                return ABS, TokenType.ABS
            if re_match == INF:
                return INF, TokenType.FLOAT
            if re_match == NAN:
                return FLOAT, TokenType.FLOAT
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
            self.increment_by(len(lexeme) - 1)
            return Token(ret_type, lexeme, pos)

    def current_char(self):
        return self.string[self.offset]

    def tokenize(self) -> Iterator[Token]:
        while self.offset < len(self.string):
            pos = Pos(self.line, self.column, self.offset)
            if self.string[self.offset :].startswith(F_DIV):
                token = Token(TokenType.F_DIV, F_DIV, pos)
                self.increment()
            elif self.string[self.offset :].startswith(DEFINE):
                token = Token(TokenType.DEFINE, DEFINE, pos)
                self.increment()
            elif self.string[self.offset :].startswith(CONTINUATION):
                token = Token(TokenType.WHITESPACE, CONTINUATION, pos)
                self.increment()
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
                        case TokenType.NEWLINE:
                            token = Token(TokenType.NEWLINE, NEWLINE, pos)
                            self.line += 1
                            self.column = -1
                        case _:
                            raise NoMatchFound
            yield token
            self.increment()
        yield Token(TokenType.EOF, "", Pos(self.line, self.column, self.offset))
