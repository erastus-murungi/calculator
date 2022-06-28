from collections import namedtuple
from enum import Enum
from typing import Iterator

from core import *
from core import Expression, RValue
from tokenizer import Token, TokenType

OpInfo = namedtuple("OpInfo", ("lexeme", "left_precedence", "right_precedence"))

OPERATOR_PRECEDENCE = {
    "^": OpInfo("^", 30, 30),
    "*": OpInfo("^", 20, 21),
    "/": OpInfo("^", 20, 21),
    "//": OpInfo("^", 20, 21),
    "%": OpInfo("^", 20, 21),
    "+": OpInfo("^", 10, 11),
    "-": OpInfo("^", 10, 11),
}

UNARY_OP_PRECEDENCE = {"-": 20, "+": 20}


class OpAssoc(Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


OPERATOR_ASSOC = {
    "^": OpAssoc.RIGHT,
    "*": OpAssoc.LEFT,
    "/": OpAssoc.LEFT,
    "//": OpAssoc.LEFT,
    "%": OpAssoc.LEFT,
    "+": OpAssoc.LEFT,
    "-": OpAssoc.LEFT,
}


class Parser:
    def __init__(self, tokens: Iterator[Token], ep_context: EPContext):
        self.ep_context = ep_context
        self.tokens = tuple(
            filter(
                lambda token: token.token_type != TokenType.WHITESPACE
                and token.token_type != TokenType.COMMENT
                and token.token_type != TokenType.NEWLINE,
                tokens,
            )
        )
        self.pos = 0
        self.functions: dict[str, FunctionDef] = {}
        self.nodes: list[Node] = self.parse()
        ep_context.set_state(State.SYNTACTIC_ANALYSIS_COMPLETE)

    def get_current_token(self) -> Token:
        return self.tokens[self.pos]

    def gobble_token(self):
        self.pos += 1

    def get_current_token_type(self):
        return self.get_current_token().token_type

    def consume_token(self, expected_type: TokenType, message: str = "") -> Token:
        if self.get_current_token_type() != expected_type:
            self.ep_context.get_exception_processor().raise_parsing_error(
                self.ep_context, self.get_current_token(), message, expected_type
            )
        token = self.get_current_token()
        self.gobble_token()
        return token

    def consume_token_no_check(self) -> Token:
        token = self.get_current_token()
        self.gobble_token()
        return token

    def parse_number(self):
        match self.get_current_token_type():
            case TokenType.INT:
                return self.parse_int_literal()
            case TokenType.FLOAT:
                return self.parse_float_literal()
            case _:
                return self.parse_complex_literal()

    # unary_expression ::= ( '-' | '+') expression
    # atom ::= '(' expression ')' | NUMBER | VARIABLE | unary_expression
    def parse_atom(self):
        match self.get_current_token_type():
            case TokenType.ADD | TokenType.SUBTRACT:  # unary_expression ::= ( '-' | '+') expression
                return self.parse_unary_op_expr(
                    UNARY_OP_PRECEDENCE[self.get_current_token().lexeme]
                )
            case TokenType.L_PAR:  # '(' expression ')'
                return self.parse_parenthesized_expression()  # NUMBER
            case TokenType.INT | TokenType.FLOAT | TokenType.COMPLEX:
                return self.parse_number()
            case TokenType.ID:  # VARIABLE
                return self.parse_load_or_function_call()
            case TokenType.LET:
                return self.parse_let_in()
            case TokenType.LEFT_BRACE:
                return self.parse_expression_vector()
        if self.get_current_token_type() == TokenType.EOF:
            self.ep_context.get_exception_processor().raise_parsing_error(
                self.ep_context,
                self.get_current_token(),
                "source ended unexpectedly",
                "",
            )
        else:
            self.ep_context.get_exception_processor().raise_parsing_error(
                self.ep_context, self.get_current_token(), "error parsing atom", ""
            )

    def parse_expression(self, min_precedence=1):
        lhs = self.parse_atom()
        while True:
            if (
                self.get_current_token_type() == TokenType.EOF
                or not self.current_token_is_binary_op()
                or OPERATOR_PRECEDENCE[self.get_current_token().lexeme].left_precedence
                < min_precedence
            ):
                break
            precedence = OPERATOR_PRECEDENCE[
                self.get_current_token().lexeme
            ].right_precedence
            assoc = OPERATOR_ASSOC[self.get_current_token().lexeme]
            next_min_precedence = (
                precedence + 1 if assoc == OpAssoc.LEFT else precedence
            )

            op: Token = self.consume_token_no_check()

            rhs = self.parse_expression(next_min_precedence)
            lhs = BinaryOp(
                lhs.start_location, BinaryOperator(op.loc, op.lexeme), lhs, rhs
            )
        return lhs

    def current_token_is_binary_op(self):
        return self.get_current_token_type() in {
            TokenType.ADD,
            TokenType.SUBTRACT,
            TokenType.MODULUS,
            TokenType.MULTIPLY,
            TokenType.TRUE_DIV,
            TokenType.FLOOR_DIV,
            TokenType.EXPONENT,
        }

    def parse_expr_entry(self) -> Expression:
        return self.parse_expression()

    def parse_num_literal(self, should_negate: bool = False):
        if self.get_current_token_type() == TokenType.INT:
            return self.parse_int_literal(should_negate)
        elif self.get_current_token_type() == TokenType.FLOAT:
            return self.parse_float_literal(should_negate)
        else:
            return self.parse_complex_literal()

    def parse_rvalue_vector(self) -> RValueVector:
        assert self.get_current_token_type() == TokenType.LEFT_BRACE
        return RValueVector(self.get_current_token().loc, self._parse_rvalue_vector())

    def parse_store(self) -> Store:
        """This is the only type of statement allowed by the program"""
        pos: TokenLocation = self.consume_token(
            TokenType.CONST, "expected const keyword"
        ).loc

        store_loc: RValueVector | RValue
        if self.get_current_token_type() == TokenType.LEFT_BRACE:
            store_loc = self.parse_rvalue_vector()
        else:
            var: Token = self.consume_token(TokenType.ID, "expected an identifier")
            store_loc = RValue(var.loc, var.lexeme)
        self.consume_token(TokenType.DEFINE, "expected :=")
        expr: Expression = self.parse_expr_entry()
        return Store(pos, store_loc, expr)

    def parse_load_or_function_call(self):
        token = self.consume_token_no_check()
        if self.get_current_token_type() == TokenType.L_PAR:
            return self.parse_function_call(token.lexeme)
        return Load(token.loc, RValue(token.loc, token.lexeme))

    def parse_function_call(self, func_name: str):
        try:
            funcdef = self.functions[func_name]
            pos, args = self.parse_args()
            return FunctionCall(pos, funcdef, args)
        except KeyError:
            try:
                pyfunc: Callable = self.ep_context.get_global_env()[func_name]
                pos, args = self.parse_args()
                return PyFunctionCall(pos, func_name, pyfunc, args)
            except KeyError:
                self.ep_context.get_exception_processor().raise_parsing_error(
                    self.ep_context,
                    self.get_current_token(),
                    f"{func_name} undefined",
                    TokenType.FUNCTION,
                )

    def parse_let_in(self):
        pos = self.get_current_token().loc
        bindings = []
        while self.get_current_token_type() != TokenType.RETURN:
            self.consume_token(TokenType.LET, "expected let keyword")
            var: Token = self.consume_token(TokenType.ID, "expected an identifier")
            self.consume_token(TokenType.DEFINE, "expected :=")
            expr: Expression = self.parse_expr_entry()
            bindings.append((RValue(var.loc, var.lexeme), expr))
            self.consume_token(TokenType.IN, "expected in")
        self.consume_token(TokenType.RETURN, "expected return")
        ret_expr = self.parse_expr_entry()
        return LetIn(pos, tuple(bindings), ret_expr)

    def parse_int_literal(self, should_negate: bool = False):
        int_literal = self.consume_token(TokenType.INT, "expected an int literal")
        offset = int_literal.loc
        lexeme = "-" + int_literal.lexeme if should_negate else int_literal.lexeme
        if lexeme.startswith("0b") or lexeme.startswith("0B"):
            return BinLiteral(offset, int(lexeme[2:], 2), lexeme)
        elif lexeme.startswith("0o") or lexeme.startswith("0O"):
            return OctLiteral(offset, int(lexeme[2:], 8), lexeme)
        elif lexeme.startswith("0X") or lexeme.startswith("0x"):
            return HexLiteral(offset, int(lexeme[2:], 16), lexeme)
        else:
            return DecimalLiteral(offset, int(lexeme, 10), lexeme)

    def parse_expression_vector(self):
        return ExprVector(self.get_current_token().loc, self._parse_expression_vector())

    def _parse_expression_vector(self) -> tuple[Expression, ...]:
        self.consume_token(
            TokenType.LEFT_BRACE, f"expected {TokenType.LEFT_BRACE} to start a vector"
        )
        ret: list[Expression] = []
        while self.get_current_token_type() != TokenType.RIGHT_BRACE:
            unit = self.parse_expr_entry()
            ret.append(unit)
            if self.get_current_token_type() == TokenType.COMMA:
                self.consume_token(TokenType.COMMA)
        self.consume_token(
            TokenType.RIGHT_BRACE, "expected to close with a right parenthesis"
        )
        return tuple(ret)

    def _parse_rvalue_vector(self) -> tuple[RValue, ...]:
        self.consume_token(
            TokenType.LEFT_BRACE, f"expected {TokenType.LEFT_BRACE} to start a vector"
        )
        ret: list[RValue] = []
        while self.get_current_token_type() != TokenType.RIGHT_BRACE:
            token = self.consume_token(TokenType.ID, "expected variable name")
            unit = RValue(token.loc, token.lexeme)
            ret.append(unit)
            if self.get_current_token_type() == TokenType.COMMA:
                self.consume_token(TokenType.COMMA)
        self.consume_token(
            TokenType.RIGHT_BRACE, "expected to close with a right parenthesis"
        )
        return tuple(ret)

    def parse(self):
        nodes = []
        while self.get_current_token_type() != TokenType.EOF:
            if self.get_current_token_type() == TokenType.CONST:
                nodes.append(self.parse_store())
            elif self.get_current_token_type() == TokenType.FUNCTION:
                nodes.append(self.parse_function_definition())
            else:
                nodes.append(self.parse_expr_entry())
            if self.get_current_token_type() == TokenType.EOF:
                self.consume_token_no_check()
                break
        return nodes

    def parse_parenthesized_expression(self):
        token = self.consume_token_no_check()
        expression = self.parse_expression()
        self.consume_token(TokenType.R_PAR, "expected )")
        return Parenthesized(token.loc, body=expression)

    def parse_unary_op_expr(self, precedence):
        token = self.consume_token_no_check()
        expression = self.parse_atom()
        return UnaryOp(token.loc, UnaryOperator(token.loc, token.lexeme), expression)

    def parse_float_literal(self, should_negate: bool = False):
        token = self.consume_token(TokenType.FLOAT, "expected float")
        lexeme = "-" + token.lexeme if should_negate else token.lexeme
        return FloatLiteral(token.loc, float(lexeme), lexeme)

    def parse_args(self) -> tuple[TokenLocation, tuple[Expression, ...]]:
        args: list[Expression] = []
        pos = self.consume_token(TokenType.L_PAR, "expected left param").loc
        while self.get_current_token_type() != TokenType.R_PAR:
            if self.get_current_token_type() == TokenType.COMMA:
                self.consume_token(TokenType.COMMA, "expected a comma")
            args.append(self.parse_expr_entry())
            if (
                self.get_current_token_type() == TokenType.COMMA
                or self.get_current_token_type() == TokenType.R_PAR
            ):
                continue
            raise ValueError(f"unexpected token type: {self.get_current_token_type()}")
        self.consume_token(TokenType.R_PAR)
        return pos, tuple(args)

    def parse_params(
        self, pos: TokenLocation
    ) -> tuple[TokenLocation, tuple[RValue, ...]]:
        params: list[RValue] = []
        while self.get_current_token_type() != TokenType.R_PAR:
            if self.get_current_token_type() == TokenType.COMMA:
                self.consume_token(TokenType.COMMA, "expected a comma")
            param_token = self.consume_token(TokenType.ID, "expected a parameter name")
            params.append(RValue(param_token.loc, param_token.lexeme))
            if (
                self.get_current_token_type() == TokenType.COMMA
                or self.get_current_token_type() == TokenType.R_PAR
            ):
                continue
            raise ValueError(f"unexpected token type: {self.get_current_token_type()}")
        self.consume_token(TokenType.R_PAR)
        return pos, tuple(params)

    def parse_function_definition(self):
        self.consume_token(TokenType.FUNCTION, "expected the keyword def")
        func_name = self.consume_token(
            TokenType.ID, "expected the function name after seeing the func keyword"
        )
        loc = self.consume_token(
            TokenType.L_PAR, "expected ( before function parameters"
        ).loc
        pos, parameters = self.parse_params(loc)
        assert (
            isinstance(param, RValue) for param in parameters
        ), "all parameters must be RValues"
        self.consume_token(TokenType.DEFINE, "expected a define")
        func_def = FunctionDef(
            pos, func_name.lexeme, parameters, self.parse_expression()
        )
        self.functions[func_name.lexeme] = func_def
        return func_def

    def get_num_literal_for_complex_literal(self):
        if self.get_current_token_type() == TokenType.SUBTRACT:
            self.consume_token_no_check()
            real_part = self.parse_num_literal(True)
        elif self.get_current_token_type() == TokenType.ADD:
            self.consume_token_no_check()
            real_part = self.parse_num_literal()
        else:
            real_part = self.parse_num_literal()
        return real_part

    def parse_complex_literal(self) -> ComplexLiteral:
        complex_token = self.consume_token(
            TokenType.COMPLEX, "expected complex keyword"
        )
        self.consume_token(TokenType.L_PAR, "expected left parenthesis")
        real_part = self.get_num_literal_for_complex_literal()
        self.consume_token(
            TokenType.COMMA,
            "expected comma to split real and imag part of complex number",
        )
        imag_part = self.get_num_literal_for_complex_literal()
        self.consume_token(TokenType.R_PAR, "expected right parenthesis")
        return ComplexLiteral(
            complex_token.loc,
            f"{complex_token.lexeme}({real_part.source()}, {imag_part.source()})",
            real_part,
            imag_part,
        )

    def __repr__(self):
        return f"Parser(pos={self.pos}, tok={self.get_current_token().lexeme})"
