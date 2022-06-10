from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from exc_processor import Loc, ExceptionProcessor


class Type(Enum):
    Integer = 0
    Float = 1
    Complex = 2
    Undefined = 3
    Error = 4

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class Node(ABC):
    pos: Loc = field(repr=False)

    def is_terminal(self):
        return len(self.children()) == 0

    @abstractmethod
    def children(self) -> tuple["Node", ...]:
        pass

    @abstractmethod
    def evaluate_type(
        self,
        types: dict["Node", Type],
        env: dict[str, "Expression"],
        exception_processor: ExceptionProcessor,
        exceptions: list[Exception],
    ):
        pass

    @abstractmethod
    def evaluate(self, types, env, exception_processor, exceptions, values):
        pass

    @abstractmethod
    def source(self):
        pass


@dataclass(frozen=True)
class Expression(Node, ABC):
    pass


@dataclass(frozen=True)
class Parenthesized(Expression):
    def evaluate(self, types, env, exception_processor, exceptions, values):
        self.body.evaluate(types, env, exception_processor, exceptions, values)
        values[self] = values[self.body]

    body: Expression

    def children(self):
        return (self.body,)

    def evaluate_type(self, types, env, exception_processor, exceptions):
        self.body.evaluate_type(types, env, exception_processor, exceptions)
        types[self] = types[self.body]

    def source(self):
        return f"{(self.body.source())}"


@dataclass(frozen=True)
class Value(Expression, ABC):
    literal: str

    def source(self):
        return f"{self.literal}"

    def __eq__(self, other):
        return self.literal.__eq__(other.literal)


@dataclass(frozen=True)
class RValue(Value):
    def children(self) -> tuple[Node, ...]:
        return ()

    def evaluate_type(
        self,
        types: dict["Node", Type],
        env: dict[str, "Expression"],
        exception_processor: ExceptionProcessor,
        exceptions: list[Exception],
    ):
        pass

    def evaluate(self, types, env, exception_processor, exceptions, values):
        pass

    def __eq__(self, other):
        return self.literal.__eq__(other.literal)


@dataclass(frozen=True)
class RValue(Value):
    def evaluate_type(self, types, env, exception_processor, exceptions):
        types[self] = Type.Undefined

    def evaluate(self, types, env, exception_processor, exceptions, values):
        values[self] = None

    def children(self):
        return ()


@dataclass(frozen=True)
class LValue(Value, ABC):
    @staticmethod
    @abstractmethod
    def get_type() -> Type:
        pass


@dataclass(frozen=True)
class FloatLiteral(LValue):
    @staticmethod
    def get_type() -> Type:
        return Type.Float

    def evaluate(self, types, env, exception_processor, exceptions, values):
        values[self] = self.raw_value

    def evaluate_type(self, types, env, exception_processor, exceptions):
        types[self] = FloatLiteral.get_type()

    def children(self):
        return ()

    raw_value: float


@dataclass(frozen=True)
class IntLiteral(LValue, ABC):
    raw_value: int

    def children(self):
        return ()

    def evaluate_type(self, types, env, exception_processor, exceptions):
        types[self] = IntLiteral.get_type()

    def evaluate(self, types, env, exception_processor, exceptions, values):
        values[self] = self.raw_value

    @staticmethod
    def get_type() -> Type:
        return Type.Integer


class DecimalLiteral(IntLiteral):
    pass


class HexLiteral(IntLiteral):
    pass


class BinLiteral(IntLiteral):
    pass


class OctLiteral(IntLiteral):
    pass


@dataclass(frozen=True)
class Store(Node):
    def source(self):
        return f"let {self.id.source()} := {self.expr.source()}"

    def evaluate_type(self, types, env, exception_processor, exceptions):
        self.id.evaluate_type(types, env, exception_processor, exceptions)
        self.expr.evaluate_type(types, env, exception_processor, exceptions)
        types[self] = Type.Undefined

    def evaluate(self, types, env, exception_processor, exceptions, values):
        self.expr.evaluate(types, env, exception_processor, exceptions, values)
        values[self] = None
        values[self.id] = values[self.expr]

    id: RValue
    expr: Expression

    def children(self):
        return self.id, self.expr


@dataclass(frozen=True)
class Load(Expression):
    def source(self):
        return self.id.source()

    def evaluate(self, types, env, exception_processor, exceptions, values):
        values[self] = values[self.id]

    def evaluate_type(self, types, env, exception_processor, exceptions):
        types[self] = Type.Undefined

    id: Value

    def children(self) -> Optional[tuple["Node", ...]]:
        return ()


class Operator(Node, ABC):
    def children(self) -> Optional[tuple["Node", ...]]:
        return ()

    def evaluate(self, types, env, exception_processor, exceptions, values):
        values[self] = None

    def evaluate_type(self, types, env, exception_processor, exceptions):
        types[self] = Type.Undefined


class UnaryOperator(Operator, ABC):
    @abstractmethod
    def evaluate_with_operand(self, operand):
        pass


class UnaryPlus(UnaryOperator):
    def source(self):
        return "+"

    def evaluate_with_operand(self, operand):
        return operand


class UnarySub(UnaryOperator):
    def source(self):
        return "-"

    def evaluate_with_operand(self, operand):
        return -operand


@dataclass(frozen=True)
class UnaryOp(Expression):
    def source(self):
        return f"{self.op.source()} {self.operand.source()}"

    def evaluate(self, types, env, exception_processor, exceptions, values):
        self.operand.evaluate(types, env, exception_processor, exceptions, values)
        value = self.op.evaluate_with_operand(values[self.operand])
        values[self] = value

    op: UnaryOperator
    operand: Expression

    def evaluate_type(self, types, env, exception_processor, exceptions):
        self.op.evaluate_type(types, env, exception_processor, exceptions)
        self.operand.evaluate_type(types, env, exception_processor, exceptions)
        types[self] = types[self.operand]

    def children(self) -> Optional[tuple["Node", ...]]:
        return self.op, self.operand


@dataclass(frozen=True)
class BinaryOperator(Operator, ABC):
    pass

    @abstractmethod
    def evaluate_with_operands(self, rhs, lhs):
        pass


class Add(BinaryOperator):
    def source(self):
        return "+"

    def evaluate_with_operands(self, rhs, lhs):
        return rhs + lhs


class Subtract(BinaryOperator):
    def source(self):
        return "-"

    def evaluate_with_operands(self, rhs, lhs):
        return rhs - lhs


class Multiply(BinaryOperator):
    def source(self):
        return "*"

    def evaluate_with_operands(self, rhs, lhs):
        return rhs * lhs


class Divide(BinaryOperator):
    def source(self):
        return "/"

    def evaluate_with_operands(self, rhs, lhs):
        return rhs / lhs


class FloorDivide(BinaryOperator):
    def source(self):
        return "//"

    def evaluate_with_operands(self, rhs, lhs):
        return rhs // lhs


class Modulus(BinaryOperator):
    def source(self):
        return "%"

    def evaluate_with_operands(self, rhs, lhs):
        return rhs % lhs


class Exponent(BinaryOperator):
    def source(self):
        return "^"

    def evaluate_with_operands(self, rhs, lhs):
        return rhs**lhs


@dataclass(frozen=True)
class BinaryOp(Expression):
    def source(self):
        return f"{self.left.source()} {self.op.source()} {self.right.source()}"

    def evaluate(self, types, env, exception_processor, exceptions, values):
        self.left.evaluate(types, env, exception_processor, exceptions, values)
        self.right.evaluate(types, env, exception_processor, exceptions, values)
        self.op.evaluate(types, env, exception_processor, exceptions, values)
        value = self.op.evaluate_with_operands(values[self.left], values[self.right])
        values[self] = value

    def evaluate_type(self, types, env, exception_processor, exceptions):
        self.left.evaluate_type(types, env, exception_processor, exceptions)
        self.right.evaluate_type(types, env, exception_processor, exceptions)
        self.op.evaluate_type(types, env, exception_processor, exceptions)
        if types[self.left] != types[self.right]:
            exceptions.append(
                exception_processor.raise_exception(
                    self, self.pos, f"{types[self.left]} and {types[self.right]}"
                )
            )
            types[self] = Type.Error
        else:
            types[self] = types[self.left]

    def children(self) -> Optional[tuple["Node", ...]]:
        return self.left, self.op, self.right

    op: BinaryOperator
    left: Expression
    right: Expression


@dataclass(frozen=True)
class FunctionCall(Expression):
    def source(self):
        return f"{self.name} ({', '.join([arg.source() for arg in self.arguments])})"

    def children(self) -> Optional[tuple["Node", ...]]:
        return self.body.children()

    name: str
    arguments: tuple[LValue, ...]
    body: Expression

    def evaluate(self, types, env, exception_processor, exceptions, values):
        return self.body.evaluate(types, env, exception_processor, exceptions, values)

    def get_all_l_values(self):
        def recurse_on_children(node):
            children = node.children()
            return children + tuple(map(recurse_on_children, children))

        return tuple(
            filter(lambda n: isinstance(n, LValue), recurse_on_children(self.body))
        )

    @staticmethod
    def all_same_type(elements):
        return all(isinstance(sub, type(elements[0])) for sub in elements[1:])

    def evaluate_type(self, types, env, exception_processor, exceptions):
        _type = Type.Undefined
        all_constants = self.get_all_l_values()
        if all_constants:
            if not self.all_same_type(all_constants):
                raise ValueError()
            else:
                _type = all_constants[0].get_type()
        if self.arguments:
            if not self.all_same_type(self.arguments):
                raise ValueError()
            else:
                type_args = self.arguments[0].get_type()
                if _type != Type.Undefined and type_args != _type:
                    raise ValueError
                _type = type_args
        return _type


@dataclass(frozen=True)
class FunctionDef(Node):
    name: str
    parameters: tuple[RValue, ...]
    body: Expression

    def source(self):
        return f"{self.name} ({', '.join([arg.source() for arg in self.parameters])}) := {self.body.source()}"

    def evaluate_type(self, types, env, exception_processor, exceptions):
        types[self] = Type.Undefined

    def evaluate(self, types, env, exception_processor, exceptions, values):
        values[self] = None

    def children(self) -> Optional[tuple["Node", ...]]:
        return self.body.children()
