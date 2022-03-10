from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from exc_processor import Pos, ExceptionProcessor


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
    pos: Pos = field(repr=False)

    @abstractmethod
    def children(self) -> Optional[tuple["Node", ...]]:
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


@dataclass(frozen=True)
class NumLiteral(Expression, ABC):
    literal: str


@dataclass(frozen=True)
class FloatLiteral(NumLiteral):
    def evaluate(self, types, env, exception_processor, exceptions, values):
        values[self] = self.raw_value

    def evaluate_type(self, types, env, exception_processor, exceptions):
        types[self] = Type.Float

    def children(self):
        return None

    raw_value: float


@dataclass(frozen=True)
class IntLiteral(NumLiteral, ABC):
    raw_value: int

    def children(self):
        return None

    def evaluate_type(self, types, env, exception_processor, exceptions):
        types[self] = Type.Integer

    def evaluate(self, types, env, exception_processor, exceptions, values):
        values[self] = self.raw_value


class DecimalLiteral(IntLiteral):
    pass


class HexLiteral(IntLiteral):
    pass


class BinLiteral(IntLiteral):
    pass


class OctLiteral(IntLiteral):
    pass


@dataclass(frozen=True)
class Declaration(Node):
    def evaluate_type(self, types, env, exception_processor, exceptions):
        types[self] = Type.Undefined

    def evaluate(self, types, env, exception_processor, exceptions, values):
        values[self] = None

    var: str

    def children(self):
        return None


@dataclass(frozen=True)
class Store(Node):
    def evaluate_type(self, types, env, exception_processor, exceptions):
        self.id.evaluate_type(types, env, exception_processor, exceptions)
        self.expr.evaluate_type(types, env, exception_processor, exceptions)
        types[self] = Type.Undefined

    def evaluate(self, types, env, exception_processor, exceptions, values):
        self.expr.evaluate(types, env, exception_processor, exceptions, values)
        values[self] = None

    id: Declaration
    expr: Expression

    def children(self):
        return self.id, self.expr


@dataclass(frozen=True)
class Load(Expression):
    def evaluate(self, types, env, exception_processor, exceptions, values):
        values[self] = values[env[self.id.var]]

    def evaluate_type(self, types, env, exception_processor, exceptions):
        types[self] = types[env[self.id.var]]

    id: Declaration

    def children(self) -> Optional[tuple["Node", ...]]:
        return (self.id,)


class Operator(Node, ABC):
    def children(self) -> Optional[tuple["Node", ...]]:
        return None

    def evaluate(self, types, env, exception_processor, exceptions, values):
        values[self] = None

    def evaluate_type(self, types, env, exception_processor, exceptions):
        types[self] = Type.Undefined


class UnaryOperator(Operator, ABC):
    @abstractmethod
    def evaluate_with_operand(self, operand):
        pass


class UnaryPlus(UnaryOperator):
    def evaluate_with_operand(self, operand):
        return operand


class UnarySub(UnaryOperator):
    def evaluate_with_operand(self, operand):
        return -operand


@dataclass(frozen=True)
class UnaryOp(Expression):
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
    def evaluate_with_operands(self, rhs, lhs):
        return rhs + lhs


class Subtract(BinaryOperator):
    def evaluate_with_operands(self, rhs, lhs):
        return rhs - lhs


class Multiply(BinaryOperator):
    def evaluate_with_operands(self, rhs, lhs):
        return rhs * lhs


class Divide(BinaryOperator):
    def evaluate_with_operands(self, rhs, lhs):
        return rhs / lhs


class FloorDivide(BinaryOperator):
    def evaluate_with_operands(self, rhs, lhs):
        return rhs // lhs


class Modulus(BinaryOperator):
    def evaluate_with_operands(self, rhs, lhs):
        return rhs % lhs


class Exponent(BinaryOperator):
    def evaluate_with_operands(self, rhs, lhs):
        return rhs ** lhs


@dataclass(frozen=True)
class BinaryOp(Expression):
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
        return self.op, self.left, self.right

    op: BinaryOperator
    left: Expression
    right: Expression
