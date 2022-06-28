import sys

from core import (
    Env,
    EPContext,
    Expression,
    FunctionDef,
    LetIn,
    Load,
    Node,
    NodeToEnvMapping,
    NodeToTypeMapping,
    ProcessingException,
    RValue,
    RValueVector,
    State,
    Store,
)


def check_semantics(
    nodes: list[Node],
    ep_context: EPContext,
):
    exceptions: list[ProcessingException] = []
    ep_context.set_node_to_env_mapping(populate_env(nodes))
    check_scope(nodes, ep_context)
    ep_context.set_node_to_type_mapping(evaluate_types(nodes, ep_context))
    check_exceptions(exceptions)
    ep_context.set_state(State.SEMANTIC_ANALYSIS_COMPLETE)


def check_exceptions(exceptions):
    if len(exceptions) != 0:
        for exception in exceptions:
            exc, tb = exception
            print(exc, file=sys.stderr)
            print(*tb, file=sys.stderr)


def populate_env_let_in(
    let_in: LetIn, node_to_env: dict[Node, Env], env: Env, seen: set[str]
) -> Env:
    for binding_name, expr in let_in.bindings:
        for literal in binding_name.literals():
            if literal in seen:
                raise ValueError("redeclared binding")
            seen.add(literal)
            env.add(literal)
            node_to_env[binding_name] = env
            node_to_env[expr] = env
            env = env.add_child()
        node_to_env[expr] = env
    node_to_env[let_in.ret] = env
    return env


def populate_env(
    nodes: list[Node],
) -> NodeToEnvMapping:
    env = Env()
    node_to_env: NodeToEnvMapping = {}
    seen: set[str] = set()
    for node in nodes:
        if isinstance(node, Store):
            for literal in node.id.literals():
                if literal in seen:
                    raise ValueError("redeclared binding")
                seen.add(literal)
                env.add(literal)
                env = env.add_child()
        elif isinstance(node, FunctionDef):
            for param in node.parameters:
                for literal in param.literals():
                    env.add(literal)
                    node_to_env[param] = env
                    env = env.add_child()
            if isinstance(node.body, LetIn):
                env = populate_env_let_in(
                    node.body,
                    node_to_env,
                    env,
                    {param.literals() for param in node.parameters},
                )
        elif not isinstance(node, Expression):
            raise ValueError(node)
        node_to_env[node] = env
    return node_to_env


def check_scope(nodes: list[Node], ep_context: EPContext):
    def _check_scope_for_node(node: Node, env: Env):
        if isinstance(node, RValue):
            for literal in node.literals():
                if literal not in env:
                    raise ValueError(f"binding '{literal}' not defined")
        if node.children() is not None:
            children: tuple[Node, ...] = node.children()
            for child in children:
                _check_scope_for_node(child, env)

    for root_node in nodes:
        if isinstance(root_node, Expression):
            _check_scope_for_node(
                root_node, ep_context.get_node_to_env_mapping()[root_node]
            )


def evaluate_types(nodes: list[Node], ep_context: EPContext):
    node_to_type_mapping: NodeToTypeMapping = {}
    ep_context.set_node_to_type_mapping(node_to_type_mapping)
    for root_node in nodes:
        root_node.evaluate_type(ep_context)
    return node_to_type_mapping
