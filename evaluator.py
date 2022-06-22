from numbers import Number

from core import EPContext, Node, NodeToValueMapping, State


def evaluate(
    nodes: list[Node], ep_context: EPContext
) -> dict[str, None | Number | list[Number]]:
    node_to_value_mapping: NodeToValueMapping = {}
    ep_context.set_node_to_value_mapping(node_to_value_mapping)
    for node in nodes:
        node.evaluate(ep_context)
    ep_context.set_state(State.EVALUATION_COMPLETE)
    return {node.source(): node_to_value_mapping[node] for node in nodes}
