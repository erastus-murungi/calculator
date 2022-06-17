from collections import namedtuple
from typing import Iterator

from core import *
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
            filter(lambda token: token.token_type != TokenType.WHITESPACE, tokens)
        )
        self.pos = 0
        self.functions: dict[str, FunctionDef] = {}
        self.nodes: list[Node] = self.parse()
        ep_context.set_state(State.SYNTACTIC_ANALYSIS_COMPLETE)

    def get_current_token(self):
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
            lhs = BinaryOp(lhs.pos, BinaryOperator(op.loc, op.lexeme), lhs, rhs)
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

    def parse_store(self):
        """This is the only type of statement allowed by the program"""
        pos: TokenLocation = self.consume_token(
            TokenType.LET, "expected let keyword"
        ).loc
        var: Token = self.consume_token(TokenType.ID, "expected an identifier")
        self.consume_token(TokenType.DEFINE, "expected :=")
        expr: Expression = self.parse_expr_entry()
        return Store(pos, RValue(var.loc, var.lexeme), expr)

    def parse_load_or_function_call(self):
        token = self.consume_token_no_check()
        if self.get_current_token_type() == TokenType.L_PAR:
            return self.parse_function_call(token.lexeme)
        return Load(token.loc, RValue(token.loc, token.lexeme))

    def parse_function_call(self, func_name: str):
        try:
            funcdef = self.functions[func_name]
            pos, args = self.parse_args_or_params(is_parameter=False)
            return FunctionCall(pos, funcdef, args)
        except KeyError:
            try:
                funcdef = self.ep_context.get_global_env()[func_name]
                pos, args = self.parse_args_or_params(is_parameter=False)
                return PyFunctionCall(pos, funcdef, args)
            except KeyError:
                self.ep_context.get_exception_processor().raise_parsing_error(
                    self.ep_context,
                    self.get_current_token(),
                    f"{func_name} undefined",
                    TokenType.FUNCTION,
                )

    def parse_int_literal(self, should_negate: bool = False):
        int_literal = self.consume_token(TokenType.INT, "expected an int literal")
        offset = int_literal.loc
        if should_negate:
            lexeme = "-" + int_literal.lexeme
        else:
            lexeme = int_literal.lexeme
        if lexeme.startswith("0b") or lexeme.startswith("0B"):
            return BinLiteral(offset, lexeme, int(lexeme[2:], 2))
        elif lexeme.startswith("0o") or lexeme.startswith("0O"):
            return OctLiteral(offset, lexeme, int(lexeme[2:], 8))
        elif lexeme.startswith("0X") or lexeme.startswith("0x"):
            return HexLiteral(offset, lexeme, int(lexeme[2:], 16))
        else:
            return DecimalLiteral(offset, lexeme, int(lexeme, 10))

    def parse(self):
        nodes = []
        while self.get_current_token_type() != TokenType.EOF:
            if self.get_current_token_type() == TokenType.LET:
                nodes.append(self.parse_store())
            elif self.get_current_token_type() == TokenType.FUNCTION:
                nodes.append(self.parse_function_definition())
            else:
                nodes.append(self.parse_expr_entry())
            if self.get_current_token_type() == TokenType.EOF:
                self.consume_token_no_check()
                break
            else:
                self.consume_token(
                    TokenType.NEWLINE, "expected a newline character to split lines"
                )
        return nodes

    def parse_parenthesized_expression(self):
        token = self.consume_token_no_check()
        expression = self.parse_expression()
        self.consume_token(TokenType.R_PAR, "expected )")
        return Parenthesized(token.loc, body=expression)

    def parse_unary_op_expr(self, precedence):
        token = self.consume_token_no_check()
        expression = self.parse_atom()
        if token.token_type == TokenType.ADD:
            return UnaryOp(token.loc, UnaryPlus(token.loc), expression)
        elif token.token_type == TokenType.SUBTRACT:
            return UnaryOp(token.loc, UnarySub(token.loc), expression)
        raise ValueError

    def parse_float_literal(self, should_negate: bool = False):
        token = self.consume_token(TokenType.FLOAT, "expected float")
        if should_negate:
            lexeme = "-" + token.lexeme
        else:
            lexeme = token.lexeme
        return FloatLiteral(token.loc, lexeme, float(lexeme))

    def parse_args_or_params(
        self, is_parameter: bool
    ) -> tuple[TokenLocation, tuple[Value]]:
        parameters_or_args: list[RValue] | list[Value] = []
        pos = self.consume_token(TokenType.L_PAR, "expected left param").loc
        while self.get_current_token_type() != TokenType.R_PAR:
            if self.get_current_token_type() == TokenType.COMMA:
                self.consume_token(TokenType.COMMA, "expected a comma")
            if is_parameter:
                param_token = self.consume_token(
                    TokenType.ID, "expected a parameter name"
                )
                parameters_or_args.append(RValue(param_token.loc, param_token.lexeme))
            else:
                if self.get_current_token_type() == TokenType.ID:
                    token = self.consume_token(TokenType.ID, "expected an identifier")
                    param_or_arg = RValue(token.loc, token.lexeme)
                else:
                    param_or_arg = self.parse_num_literal()
                parameters_or_args.append(param_or_arg)
            if (
                self.get_current_token_type() == TokenType.COMMA
                or self.get_current_token_type() == TokenType.R_PAR
            ):
                continue
            raise ValueError(f"unexpected token type: {self.get_current_token_type()}")
        self.consume_token(TokenType.R_PAR)
        return pos, tuple(parameters_or_args)

    def parse_function_definition(self):
        self.consume_token(TokenType.FUNCTION, "expected the keyword func")
        func_name = self.consume_token(
            TokenType.ID, "expected the function name after seeing the func keyword"
        )
        pos, parameters = self.parse_args_or_params(is_parameter=True)
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
