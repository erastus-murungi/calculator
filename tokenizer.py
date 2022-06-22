import itertools
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

from core import EPContext, State, TokenLocation


class NoMatchFound(Exception):
    pass


class TokenType(Enum):
    # arithmetic operators
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    TRUE_DIV = "/"  # for example 3 / 2 = 1.5
    FLOOR_DIV = "//"  # for example
    MODULUS = "%"
    EXPONENT = "^"
    COMMA = ","
    COMPLEX = "complex"  # used to define a complex number
    DEFINE = ":="
    CONST = "const"  # define a binding
    FUNCTION = "def"  # define a function
    LET = "let"
    IN = "in"
    RETURN = "return"
    NEWLINE = "\n"
    CONTINUE = "\\n"

    L_PAR = "("
    R_PAR = ")"

    LEFT_BRACE = "{"
    RIGHT_BRACE = "}"

    ID = "identifier"
    FLOAT = "float"
    INT = "integer"
    WHITESPACE = "whitespace"
    EOF = "EOF"
    COMMENT = "#"

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
    loc: TokenLocation

    @staticmethod
    def from_token_type(token_type: TokenType, loc: TokenLocation):
        """A convenient constructor to avoid the frequent pattern:
        Token(TokenType.X, TokenType.X.value, loc)"""
        return Token(token_type, token_type.value, loc)


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
    """This class tokenizes an Ep program"""

    def __init__(self, ep_context: EPContext):
        self._exception_processor = ep_context.get_exception_processor()
        self._code = ep_context.get_source_code() + "\n"
        self._linenum = 0
        self._column = 0
        self._code_offset = 0
        self._token_iterable = self._tokenize()
        ep_context.set_state(State.LEXICAL_ANALYSIS_COMPLETE)

    def _to_next_char(self):
        self._code_offset += 1
        self._column += 1

    def _skip_n_chars(self, n):
        self._code_offset += n
        self._column += n

    def _match_number(self, number_regex: str) -> tuple[str, TokenType] | None:
        match = re.match("^" + number_regex, self._code[self._code_offset :])
        token_type = None
        lexeme = ""
        if match is not None:
            lexeme, token_type = (
                match.group(0),
                TokenType.FLOAT
                if number_regex == Floatnumber
                else TokenType.INT
                if number_regex == Intnumber
                else TokenType.COMPLEX
                if number_regex == Imagnumber
                else None,
            )
        if token_type is None:
            return None
        else:
            return lexeme, token_type

    def _match_keyword(self) -> Optional[tuple[str, TokenType]]:
        re_match = re.match("^" + Name, self._code[self._code_offset :])
        if re_match is not None:
            re_match = re_match.group(0)
            if re_match == TokenType.CONST.value:
                return TokenType.CONST.value, TokenType.CONST
            elif re_match == TokenType.FUNCTION.value:
                return TokenType.FUNCTION.value, TokenType.FUNCTION
            elif re_match == TokenType.LET.value:
                return TokenType.LET.value, TokenType.LET
            elif re_match == TokenType.IN.value:
                return TokenType.IN.value, TokenType.IN
            elif re_match == TokenType.RETURN.value:
                return TokenType.RETURN.value, TokenType.RETURN
            else:
                return re_match, TokenType.ID
        return None

    def _try_match_keyword_or_number(self, pos) -> Token:
        ret = (
            self._match_number(Imagnumber)
            or self._match_number(Floatnumber)
            or self._match_number(Intnumber)
            or self._match_keyword()
        )
        if ret is None:
            raise NoMatchFound(self._current_char())
        else:
            lexeme, ret_type = ret
            self._skip_n_chars(len(lexeme) - 1)
            return Token(ret_type, lexeme, pos)

    def _current_char(self):
        return self._code[self._code_offset]

    def _remaining_code(self):
        return self._code[self._code_offset :]

    def _tokenize(self) -> Iterator[Token]:
        filename = self._exception_processor.filename
        while self._code_offset < len(self._code):
            token_location = TokenLocation(
                filename, self._linenum, self._column, self._code_offset
            )
            # we greedily match two character tokens
            if (
                self._remaining_code().startswith(TokenType.FLOOR_DIV.value)
                or self._remaining_code().startswith(TokenType.DEFINE.value)
                or self._remaining_code().startswith(TokenType.CONTINUE.value)
            ):
                token_type = TokenType(
                    self._code[self._code_offset : self._code_offset + 2]
                )
                token = Token.from_token_type(token_type, token_location)
                self._to_next_char()
            # we try to match whitespace while avoiding NEWLINES because we
            # are using NEWLINES to split lines in our program
            elif (
                self._current_char() != TokenType.NEWLINE.value
                and self._current_char().isspace()
            ):
                token = Token(
                    TokenType.WHITESPACE, self._current_char(), token_location
                )
            else:
                try:
                    # try to match one character tokens
                    token_type = TokenType(self._current_char())
                    match token_type:
                        # We try to match tokens which are just one character
                        # These will only cause col and offset to increase by 1
                        case TokenType.L_PAR | TokenType.R_PAR | TokenType.ADD | TokenType.SUBTRACT | TokenType.MODULUS | TokenType.TRUE_DIV | TokenType.MULTIPLY | TokenType.EXPONENT | TokenType.COMMA | TokenType.LEFT_BRACE | TokenType.RIGHT_BRACE:
                            token = Token.from_token_type(token_type, token_location)
                        # Although a newline is only one character, we match it differently because it will cause
                        # col to reset to 0 and line to increase by 1
                        case TokenType.COMMENT:
                            token = self.handle_comment()
                        case TokenType.NEWLINE:
                            token = Token.from_token_type(token_type, token_location)
                            self._linenum += 1
                            # we set column to -1 because it will be incremented to 0 after the token has been yielded
                            self._column = -1
                        case _:
                            raise NoMatchFound
                except ValueError:
                    token = self._try_match_keyword_or_number(token_location)

            yield token
            self._to_next_char()

        # we must always end our stream of tokens with an EOF token
        yield Token(
            TokenType.EOF,
            "",
            TokenLocation(filename, self._linenum, self._column, self._code_offset),
        )

    def handle_comment(self):
        end_comment_pos = self._remaining_code().index("\n")
        if end_comment_pos == -1:
            raise ValueError()
        comment = self._remaining_code()[:end_comment_pos]
        token = Token(
            TokenType.COMMENT,
            comment,
            TokenLocation(
                self._exception_processor.filename,
                self._linenum,
                self._column,
                self._code_offset,
            ),
        )
        self._skip_n_chars(len(comment))
        self._linenum += 1
        # we set column to -1 because it will be incremented to 0 after the token has been yielded
        self._column = -1
        return token

    def get_tokens(self) -> Iterator[Token]:
        """
        :return: an iterator over the tokens
        """
        t1, t2 = itertools.tee(self._token_iterable)
        self._token_iterable = t1
        return t2
