from typing import Callable, Iterable, Iterator, List, Sequence, Set, Tuple

import networkx as nx

from pycid.core.causal_bayesian_network import CausalBayesianNetwork


def _dfs_search_paths(start: str, end: str, successors: Callable[[List[str]], Iterable[str]]) -> Iterator[List[str]]:
    """Perform a depth-first search over paths from start to end.

    Successors is a function mapping a path to a set of possible successor states.
    """
    if start == end:
        yield [start]
        return

    path = [start]
    successor_stack = [iter(successors(path))]
    while successor_stack:
        try:
            next_ = next(successor_stack[-1])
        except StopIteration:
            # Exausted all successors for this path
            path.pop()
            successor_stack.pop()
            continue

        if next_ == end:
            yield path + [next_]
            continue

        path.append(next_)
        successor_stack.append(iter(successors(path)))


def _active_neighbours(cbn: CausalBayesianNetwork, path: Sequence[str], observed: Set[str]) -> Set[str]:
    """Find possibly active extensions of path conditional on the `observed' set of nodes."""
    end_of_path = path[-1]
    last_forward = len(path) > 1 and end_of_path in cbn.get_children(path[-2])
    possible_colliders: Set[str] = set().union(*[set(cbn._get_ancestors_of(e)) for e in observed])  # type: ignore

    # if going upward or at a possible collider, it's possible to continue to a parent
    if not last_forward or end_of_path in possible_colliders:
        active_parents = set(cbn.get_parents(end_of_path)) - observed
    else:
        active_parents = set()

    # it's possible to go downward if and only if not an observed node
    if end_of_path in observed:
        active_children = set()
    else:
        active_children = set(cbn.get_children(end_of_path))

    active_neighbours = active_parents.union(active_children)
    new_active_neighbours = active_neighbours - set(path)
    return new_active_neighbours


def find_active_path(
    cbn: CausalBayesianNetwork, start_node: str, end_node: str, observed: Set[str] = set()
) -> List[str]:
    """Find an active path from `start_node' to `end_node' given the `observed' set of nodes."""
    considered_nodes = observed.union({start_node}, {end_node})
    for node in considered_nodes:
        if node not in cbn.nodes():
            raise KeyError(f"The node {node} is not in the (MA)CID")

    if end_node in observed:
        raise ValueError("No active path")

    def successors(path: List[str]) -> Set[str]:
        return _active_neighbours(cbn, path, observed)

    try:
        return next(_dfs_search_paths(start_node, end_node, successors))
    except StopIteration:
        raise ValueError("No active path")


def get_motif(cbn: CausalBayesianNetwork, path: Sequence[str], idx: int) -> str:
    """
    Classify three node structure as a forward (chain), backward (chain), fork,
    collider, or endpoint at index 'idx' along path.
    """
    for node in path:
        if node not in cbn.nodes():
            raise KeyError(f"The node {node} is not in the (MA)CID")

    if idx > len(path) - 1:
        raise IndexError(f"The given index {idx} is not valid for the length of this path {len(path)}")

    if len(path) == idx + 1:
        return "endpoint"

    elif cbn.has_edge(path[idx - 1], path[idx]) and cbn.has_edge(path[idx], path[idx + 1]):
        return "forward"

    elif cbn.has_edge(path[idx + 1], path[idx]) and cbn.has_edge(path[idx], path[idx - 1]):
        return "backward"

    elif cbn.has_edge(path[idx - 1], path[idx]) and cbn.has_edge(path[idx + 1], path[idx]):
        return "collider"

    elif cbn.has_edge(path[idx], path[idx - 1]) and cbn.has_edge(path[idx], path[idx + 1]):
        return "fork"

    else:
        raise RuntimeError(f"unsure how to classify this path at index {idx}")


def get_motifs(cbn: CausalBayesianNetwork, path: Sequence[str]) -> List[str]:
    """classify the motif of all nodes along a path as a forward (chain), backward (chain), fork,
    collider or endpoint"""
    for node in path:
        if node not in cbn.nodes():
            raise KeyError(f"The node {node} is not in the (MA)CID")

    shapes = []
    for i in range(len(path)):
        if i == 0:
            if path[i] in cbn.get_parents(path[i + 1]):
                shapes.append("forward")
            else:
                shapes.append("backward")
        else:
            shapes.append(get_motif(cbn, path, i))
    return shapes


def find_all_dir_paths(cbn: CausalBayesianNetwork, start_node: str, end_node: str) -> Iterator[List[str]]:
    """Iterate over all directed paths from start node to end node that exist in the (MA)CID."""
    for node in [start_node, end_node]:
        if node not in cbn.nodes():
            raise KeyError(f"The node {node} is not in the (MA)CID")

    def successors(path: List[str]) -> Iterable[str]:
        return cbn.get_children(path[-1])  # type: ignore

    return _dfs_search_paths(start_node, end_node, successors)


def find_all_undir_paths(cbn: CausalBayesianNetwork, start_node: str, end_node: str) -> Iterable[List[str]]:
    """
    Finds all undirected paths from start node to end node that exist in the (MA)CID.
    """
    for node in [start_node, end_node]:
        if node not in cbn.nodes():
            raise KeyError(f"The node {node} is not in the (MA)CID")

    def successors(path: List[str]) -> Iterable[str]:
        neighbours = set(cbn.get_children(path[-1]))
        neighbours.update(cbn.get_parents(path[-1]))
        neighbours.difference_update(path)
        return neighbours

    return _dfs_search_paths(start_node, end_node, successors)


def directed_decision_free_path(cbn: CausalBayesianNetwork, start_node: str, end_node: str) -> bool:
    """
    Checks to see if a directed decision free path exists
    """
    for node in [start_node, end_node]:
        if node not in cbn.nodes():
            raise KeyError(f"The node {node} is not in the (MA)CID")

    # ignore path's start_node and end_node
    return any(cbn.decisions.isdisjoint(path[1:-1]) for path in find_all_dir_paths(cbn, start_node, end_node))


def _get_path_edges(cbn: CausalBayesianNetwork, path: Sequence[str]) -> List[Tuple[str, str]]:
    """
    Returns the structure of a path's edges as a list of pairs.
    In each pair, the first argument states where an edge starts and the second argument states
    where that same edge finishes. For example, if a (colliding) path is D1 -> X <- D2, this function
    returns: [('D1', 'X'), ('D2', 'X')]
    """
    structure = []
    for i in range(len(path) - 1):
        if path[i] in cbn.get_parents(path[i + 1]):
            structure.append((path[i], path[i + 1]))
        elif path[i + 1] in cbn.get_parents(path[i]):
            structure.append((path[i + 1], path[i]))
    return structure


def is_active_path(cbn: CausalBayesianNetwork, path: Sequence[str], observed: Set[str] = None) -> bool:
    """
    Check if a specifc path remains active given the 'observed' set of variables.
    """
    if observed is None:
        observed = set()
    considered_nodes = set(path).union(observed)
    for node in considered_nodes:
        if node not in cbn.nodes():
            raise KeyError(f"The node {node} is not in the (MA)CID")

    if len(path) < 3:
        return True

    for _, b, _ in zip(path[:-2], path[1:-1], path[2:]):
        structure = get_motif(cbn, path, path.index(b))

        if structure in {"fork", "forward", "backward"} and b in observed:
            return False

        if structure == "collider":
            descendants = nx.descendants(cbn, b).union({b})
            if not descendants.intersection(observed):
                return False

    return True


def is_active_indirect_frontdoor_trail(
    cbn: CausalBayesianNetwork, start_node: str, end_node: str, observed: Set[str] = None
) -> bool:
    """
    checks whether an active indirect frontdoor path exists given the 'observed' set of variables.
    - A frontdoor path between X and Z is a path in which the first edge comes
    out of the first node (X→···Z).
    - An indirect path contains at least one collider at some node from start_node to end_node.
    """
    if observed is None:
        observed = set()
    considered_nodes = observed.union({start_node}, {end_node})
    for node in considered_nodes:
        if node not in cbn.nodes():
            raise KeyError(f"The node {node} is not in the (MA)CID")

    for path in find_all_undir_paths(cbn, start_node, end_node):
        is_frontdoor_path = path[0] in cbn.get_parents(path[1])
        not_blocked_by_observed = is_active_path(cbn, path, observed)
        contains_collider = "collider" in get_motifs(cbn, path)
        # default is False since if w = [], any unobserved collider blocks path
        if is_frontdoor_path and not_blocked_by_observed and contains_collider:
            return True
    else:
        return False


def is_active_backdoor_trail(
    cbn: CausalBayesianNetwork, start_node: str, end_node: str, observed: Set[str] = None
) -> bool:
    """
    Returns true if there is a backdoor path that's active given the 'observed' set of nodes.
    - A backdoor path between X and Z is an (undirected) path in which the first edge goes into the first node (X←···Z)
    """
    if observed is None:
        observed = set()
    considered_nodes = observed.union({start_node}, {end_node})
    for node in considered_nodes:
        if node not in cbn.nodes():
            raise KeyError(f"The node {node} is not in the (MA)CID")

    for path in find_all_undir_paths(cbn, start_node, end_node):
        if len(path) > 1:  # must have path of at least 2 nodes
            is_backdoor_path = path[1] in cbn.get_parents(path[0])
            not_blocked_by_observed = is_active_path(cbn, path, observed)
            if is_backdoor_path and not_blocked_by_observed:
                return True
    else:
        return False
