import sys
import traceback
from collections import namedtuple

Pos = namedtuple("Pos", "line col offset")


class ProcessingException(Exception):
    pass


class ExceptionProcessor:
    def __init__(self, string: str):
        self.string = string
        self.lines: list[str] = self.string.split("\n")

    def raise_exception(self, ob: object, pos: Pos, message: str):
        line = self.lines[pos.line]
        try:
            raise ProcessingException(
                f"{line}\n from {ob.__class__.__qualname__} : {message}"
            )
        except ProcessingException as e:
            return e, traceback.format_stack()
