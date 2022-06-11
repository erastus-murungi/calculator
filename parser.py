from typing import Iterator

from ast_nodes import *
from tokenizer import Token, TokenType


class Parser:
    def __init__(
        self, tokens: Iterator[Token], exception_processor: ExceptionProcessor
    ):
        self.exception_processor: ExceptionProcessor = exception_processor
        self.tokens = tuple(
            filter(lambda token: token.token_type != TokenType.WHITESPACE, tokens)
        )
        self.pos = 0
        self.functions: dict[str, FunctionDef] = {}
        self.nodes: list[Node] = self.parse()

    def get_current_token(self):
        return self.tokens[self.pos]

    def advance(self):
        self.pos += 1

    def get_current_token_type(self):
        return self.get_current_token().token_type

    def consume_token(self, expected_type: TokenType, message: str = "") -> Token:
        if self.get_current_token_type() != expected_type:
            raise ValueError(
                f"expected {expected_type.value} got {self.get_current_token_type().value} : {message}"
            )
        token = self.get_current_token()
        self.advance()
        return token

    def consume_token_no_check(self) -> Token:
        token = self.get_current_token()
        self.advance()
        return token

    def parse_expr_entry(self) -> Expression:
        expr = self.parse_add_sub()
        if self.get_current_token_type() == TokenType.NEWLINE:
            self.consume_token_no_check()
        return expr

    def parse_add_sub(self):
        lhs = self.parse_mul_div_rem()
        if (
            self.get_current_token_type() == TokenType.ADD
            or self.get_current_token_type() == TokenType.SUBTRACT
        ):
            op_token = self.consume_token_no_check()
            rhs = self.parse_add_sub()
            if op_token.token_type == TokenType.ADD:
                return BinaryOp(lhs.pos, Add(op_token.loc), lhs, rhs)
            elif op_token.token_type == TokenType.SUBTRACT:
                return BinaryOp(lhs.pos, Subtract(op_token.loc), lhs, rhs)
            raise ValueError
        return lhs

    def parse_mul_div_rem(self):
        lhs: Expression = self.parse_exponent()
        if (
            self.get_current_token_type() == TokenType.MULTIPLY
            or self.get_current_token_type() == TokenType.FLOOR_DIV
            or self.get_current_token_type() == TokenType.TRUE_DIVIDE
            or self.get_current_token() == TokenType.MODULUS
        ):
            op_token = self.consume_token_no_check()
            rhs = self.parse_mul_div_rem()
            match op_token.token_type:
                case TokenType.MULTIPLY:
                    return BinaryOp(lhs.pos, Multiply(op_token.loc), lhs, rhs)
                case TokenType.FLOOR_DIV:
                    return BinaryOp(lhs.pos, FloorDivide(op_token.loc), lhs, rhs)
                case TokenType.TRUE_DIVIDE:
                    return BinaryOp(lhs.pos, Divide(op_token.loc), lhs, rhs)
                case TokenType.MODULUS:
                    return BinaryOp(lhs.pos, Modulus(op_token.loc), lhs, rhs)
                case _:
                    raise ValueError
        return lhs

    def parse_exponent(self) -> Expression:
        base: Expression = self.parse_expr()
        if self.get_current_token_type() == TokenType.EXPONENT:
            op_offset = self.consume_token_no_check().loc
            power: Expression = self.parse_exponent()
            return BinaryOp(base.pos, Exponent(op_offset), base, power)
        return base

    def parse_num_literal(self, should_negate: bool = False):
        if self.get_current_token_type() == TokenType.INT:
            return self.parse_int_literal(should_negate)
        elif self.get_current_token_type() == TokenType.FLOAT:
            return self.parse_float_literal(should_negate)
        else:
            return self.parse_complex_literal()

    def parse_expr(self):
        match self.get_current_token_type():
            case TokenType.ADD | TokenType.SUBTRACT:
                return self.parse_unary_op_expr()
            case TokenType.L_PAR:
                return self.parse_parenthesized_expression()
            case TokenType.INT | TokenType.FLOAT:
                return self.parse_num_literal()
            case TokenType.COMPLEX:
                return self.parse_complex_literal()
            case TokenType.ID:
                return self.parse_load_or_function_call()
        raise ValueError

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
        funcdef = self.functions[func_name]
        pos, args = self.parse_args_or_params(is_parameter=False)
        return FunctionCall(pos, funcdef, args)

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
        return nodes

    def parse_parenthesized_expression(self):
        token = self.consume_token_no_check()
        expression = self.parse_exponent()
        self.consume_token(TokenType.R_PAR, "expected (")
        return Parenthesized(token.loc, body=expression)

    def parse_unary_op_expr(self):
        token = self.consume_token_no_check()
        expression = self.parse_expr_entry()
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
        parameters_or_args: list[Value] = []
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
        function_name = self.consume_token(TokenType.ID, "expected the function name")
        pos, parameters = self.parse_args_or_params(is_parameter=True)
        self.consume_token(TokenType.DEFINE, "expected a define")
        func_def = FunctionDef(
            pos, function_name.lexeme, parameters, self.parse_expr_entry()
        )
        self.functions[function_name.lexeme] = func_def
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
        self.consume_token(TokenType.COMMA, "expected comma")
        imag_part = self.get_num_literal_for_complex_literal()
        self.consume_token(TokenType.R_PAR, "expected right parenthesis")
        return ComplexLiteral(
            complex_token.loc,
            f"{complex_token.lexeme}({real_part.source()}, {imag_part.source()})",
            real_part,
            imag_part,
        )
