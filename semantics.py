import sys

from core import (
    Expression,
    Store,
    Node,
    FunctionDef,
    RValue,
    EPContext,
    ProcessingException,
    Env,
    State,
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


def populate_env(
    nodes: list[Node],
) -> dict[Node, Env]:
    env = Env()
    node_to_env = {}
    seen = set()
    for node in nodes:
        if isinstance(node, Store):
            if node.id.literal in seen:
                raise ValueError("redeclared binding")
            seen.add(node.id.literal)
            env.add(node.id.literal)
            env = env.add_child()
        elif isinstance(node, FunctionDef):
            for param in node.parameters:
                env.add(param.literal)
                node_to_env[param] = env
                env = env.add_child()
            env = env.add_child()
        elif not isinstance(node, Expression):
            raise ValueError(node)
        node_to_env[node] = env
    return node_to_env


def check_scope(nodes: list[Node], ep_context: EPContext):
    def _check_scope_for_node(node: Node, env: Env):
        if isinstance(node, RValue):
            if node.literal not in env:
                raise ValueError(f"binding '{node.literal}' not defined")
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
    types = {}
    ep_context.set_node_to_type_mapping(types)
    for root_node in nodes:
        root_node.evaluate_type(ep_context)
    return types
