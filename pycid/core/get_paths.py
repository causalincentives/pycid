# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.
from typing import List, Set, Tuple

import networkx as nx

from pycid.core.macid_base import MACIDBase


def _active_neighbours(mb: MACIDBase, path: List[str], observed: List[str]) -> Set[str]:
    """Find possibly active extensions of path conditional on the `observed' set of nodes."""
    end_of_path = path[-1]
    last_forward = len(path) > 1 and end_of_path in mb.get_children(path[-2])
    possible_colliders: Set[str] = set().union(*[set(mb._get_ancestors_of(e)) for e in observed])  # type: ignore

    # if going upward or at a possible collider, it's possible to continue to a parent
    if end_of_path in possible_colliders or not last_forward:
        active_parents = set(mb.get_parents(end_of_path)) - set(observed)
    else:
        active_parents = set()

    # it's possible to go downward if and only if not an observed node
    if end_of_path in observed:
        active_children = set()
    else:
        active_children = set(mb.get_children(end_of_path))

    active_neighbours = active_parents.union(active_children)
    new_active_neighbours = active_neighbours - set(path)
    return new_active_neighbours


def _find_active_path_recurse(mb: MACIDBase, path: List[str], end_node: str, observed: List[str]) -> List[str]:
    """Find active path from `path' to `end_node' given the `observed' set of nodes."""
    if path[-1] == end_node and end_node not in observed:
        return path
    else:
        neighbours = _active_neighbours(mb, path, observed)
        for neighbour in neighbours:
            ext = _find_active_path_recurse(mb, path + [neighbour], end_node, observed)
            if ext:
                return ext
    return []  # should never happen


def find_active_path(mb: MACIDBase, start_node: str, end_node: str, observed: List[str] = []) -> List[str]:
    """Find active path from `start_node' to `end_node' given the `observed' set of nodes."""
    considered_nodes = set(observed).union({start_node}, {end_node})
    for node in considered_nodes:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")

    return _find_active_path_recurse(mb, [start_node], end_node, observed)


def get_motif(mb: MACIDBase, path: List[str], idx: int) -> str:
    """
    Classify three node structure as a forward (chain), backward (chain), fork,
    collider, or endpoint at index 'idx' along path.
    """
    for node in path:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")

    if idx > len(path) - 1:
        raise Exception(f"The given index {idx} is not valid for the length of this path {len(path)}")

    if len(path) == idx + 1:
        return "endpoint"

    elif mb.has_edge(path[idx - 1], path[idx]) and mb.has_edge(path[idx], path[idx + 1]):
        return "forward"

    elif mb.has_edge(path[idx + 1], path[idx]) and mb.has_edge(path[idx], path[idx - 1]):
        return "backward"

    elif mb.has_edge(path[idx - 1], path[idx]) and mb.has_edge(path[idx + 1], path[idx]):
        return "collider"

    elif mb.has_edge(path[idx], path[idx - 1]) and mb.has_edge(path[idx], path[idx + 1]):
        return "fork"

    else:
        raise Exception(f"unsure how to classify this path at index {idx}")


def get_motifs(mb: MACIDBase, path: List[str]) -> List[str]:
    """classify the motif of all nodes along a path as a forward (chain), backward (chain), fork,
    collider or endpoint"""
    for node in path:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")

    shapes = []
    for i in range(len(path)):
        if i == 0:
            if path[i] in mb.get_parents(path[i + 1]):
                shapes.append("forward")
            else:
                shapes.append("backward")
        else:
            shapes.append(get_motif(mb, path, i))
    return shapes


def _find_all_dirpath_recurse(mb: MACIDBase, path: List[str], end_node: str) -> List[List[str]]:
    """Find all directed paths beginning with 'path' as a prefix and ending at the end node"""

    if path[-1] == end_node:
        return [path]
    path_extensions = []
    children = mb.get_children(path[-1])
    for child in children:
        path_extensions.extend(_find_all_dirpath_recurse(mb, path + [child], end_node))
    return path_extensions


def find_all_dir_paths(mb: MACIDBase, start_node: str, end_node: str) -> List[List[str]]:
    """
    Find all directed paths from start node to end node that exist in the (MA)CID.
    """
    for node in [start_node, end_node]:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")
    return _find_all_dirpath_recurse(mb, [start_node], end_node)


def _find_all_undirpath_recurse(mb: MACIDBase, path: List[str], end_node: str) -> List[List[str]]:
    """Find all undirected paths beginning with 'path' as a prefix and ending at the end node."""

    if path[-1] == end_node:
        return [path]
    path_extensions = []
    neighbours = list(mb.get_children(path[-1])) + list(mb.get_parents(path[-1]))
    neighbours_not_in_path = set(neighbours).difference(set(path))
    for child in neighbours_not_in_path:
        path_extensions.extend(_find_all_undirpath_recurse(mb, path + [child], end_node))
    return path_extensions


def find_all_undir_paths(mb: MACIDBase, start_node: str, end_node: str) -> List[List[str]]:
    """
    Finds all undirected paths from start node to end node that exist in the (MA)CID.
    """
    for node in [start_node, end_node]:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")
    return _find_all_undirpath_recurse(mb, [start_node], end_node)


def directed_decision_free_path(mb: MACIDBase, start_node: str, end_node: str) -> bool:
    """
    Checks to see if a directed decision free path exists
    """
    for node in [start_node, end_node]:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")

    start_to_end_paths = find_all_dir_paths(mb, start_node, end_node)
    dec_free_path_exists = any(
        set(mb.all_decision_nodes).isdisjoint(set(path[1:-1])) for path in start_to_end_paths
    )  # ignore path's start_node and end_node
    if start_to_end_paths and dec_free_path_exists:
        return True
    else:
        return False


def _get_path_edges(mb: MACIDBase, path: List[str]) -> List[Tuple[str, str]]:
    """
    Returns the structure of a path's edges as a list of pairs.
    In each pair, the first argument states where an edge starts and the second argument states
    where that same edge finishes. For example, if a (colliding) path is D1 -> X <- D2, this function
    returns: [('D1', 'X'), ('D2', 'X')]
    """
    structure = []
    for i in range(len(path) - 1):
        if path[i] in mb.get_parents(path[i + 1]):
            structure.append((path[i], path[i + 1]))
        elif path[i + 1] in mb.get_parents(path[i]):
            structure.append((path[i + 1], path[i]))
    return structure


def is_active_path(mb: MACIDBase, path: List[str], observed: List[str] = []) -> bool:
    """
    Check if a specifc path remains active given the 'observed' set of variables.
    """
    considered_nodes = set(path).union(set(observed))
    for node in considered_nodes:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")

    if len(path) < 3:
        return True

    for _, b, _ in zip(path[:-2], path[1:-1], path[2:]):
        structure = get_motif(mb, path, path.index(b))

        if structure in {"fork", "forward", "backward"} and b in observed:
            return False

        if structure == "collider":
            descendants = nx.descendants(mb, b).union({b})
            if not descendants.intersection(set(observed)):
                return False

    return True


def is_active_indirect_frontdoor_trail(mb: MACIDBase, start_node: str, end_node: str, observed: List[str] = []) -> bool:
    """
    checks whether an active indirect frontdoor path exists given the 'observed' set of variables.
    - A frontdoor path between X and Z is a path in which the first edge comes
    out of the first node (X→···Z).
    - An indirect path contains at least one collider at some node from start_node to end_node.
    """
    considered_nodes = set(observed).union({start_node}, {end_node})
    for node in considered_nodes:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")

    start_to_end_paths = find_all_undir_paths(mb, start_node, end_node)
    for path in start_to_end_paths:
        is_frontdoor_path = path[0] in mb.get_parents(path[1])
        not_blocked_by_observed = is_active_path(mb, path, observed)
        contains_collider = "collider" in get_motifs(mb, path)
        # default is False since if w = [], any unobserved collider blocks path
        if is_frontdoor_path and not_blocked_by_observed and contains_collider:
            return True
    else:
        return False


def is_active_backdoor_trail(mb: MACIDBase, start_node: str, end_node: str, observed: List[str] = []) -> bool:
    """
    Returns true if there is a backdoor path that's active given the 'observed' set of nodes.
    - A backdoor path between X and Z is an (undirected) path in which the first edge goes into the first node (X←···Z)
    """
    considered_nodes = set(observed).union({start_node}, {end_node})
    for node in considered_nodes:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")

    start_to_end_paths = find_all_undir_paths(mb, start_node, end_node)
    for path in start_to_end_paths:

        if len(path) > 1:  # must have path of at least 2 nodes
            is_backdoor_path = path[1] in mb.get_parents(path[0])
            not_blocked_by_observed = is_active_path(mb, path, observed)
            if is_backdoor_path and not_blocked_by_observed:
                return True
    else:
        return False
