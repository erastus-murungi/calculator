import sys
import traceback
from dataclasses import dataclass

import termcolor


@dataclass(frozen=True)
class TokenLocation:
    filename: str
    line: int
    col: int
    offset: int

    def __str__(self):
        return f"<{self.filename}:{self.line}:{self.col}>"


class ProcessingException(Exception):
    pass


class ExceptionProcessor:
    def __init__(self, string: str, filename: str):
        self.string = string
        self.lines: list[str] = self.string.split("\n")
        self.filename = filename

    def raise_exception(self, ob: object, loc: TokenLocation, message: str):
        line = self.lines[loc.line]
        try:
            raise ProcessingException(
                f"{line}\n from {ob.__class__.__qualname__} : {message}"
            )
        except ProcessingException as e:
            return e, traceback.format_stack()

    def raise_a_type_mismatch_exception(
        self, args: tuple["Value", ...], types, loc: TokenLocation
    ):
        line = self.lines[loc.line]
        s = termcolor.colored("error:", "red")
        print(
            f"{loc}: {s} arguments must all be of the same type \n"
            f"{loc.line} | {line}\n"
            f"The statically evaluated types of your arguments are:\n"
            + "\n".join([f"  {arg.source()} => {types[arg]}" for arg in args])
        )
        sys.exit(1)
