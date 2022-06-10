import traceback
from collections import namedtuple
from dataclasses import dataclass


@dataclass(frozen=True)
class Loc:
    filename: str
    line: int
    col: int
    offset: int

    def __str__(self):
        return f"<{self.filename}:{self.line}:{self.line}>"


class ProcessingException(Exception):
    pass


class ExceptionProcessor:
    def __init__(self, string: str, filename: str):
        self.string = string
        self.lines: list[str] = self.string.split("\n")
        self.filename = filename

    def raise_exception(self, ob: object, loc: Loc, message: str):
        line = self.lines[loc.line]
        try:
            raise ProcessingException(
                f"{line}\n from {ob.__class__.__qualname__} : {message}"
            )
        except ProcessingException as e:
            return e, traceback.format_stack()
