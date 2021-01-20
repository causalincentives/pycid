# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.

from typing import List, Set
from pgmpy.models import BayesianModel


def _active_neighbours(bn: BayesianModel, path: List[str], observed: List[str]) -> Set[str]:
    """Find possibly active extensions of path conditional on observed"""
    end_of_path = path[-1]
    last_forward = len(path) > 1 and end_of_path in bn.get_children(path[-2])
    possible_colliders = set().union(*[set(bn._get_ancestors_of(e)) for e in observed])

    # if going upward or at a possible collider, it's possible to continue to a parent
    if end_of_path in possible_colliders or not last_forward:
        active_parents = set(bn.get_parents(end_of_path)) - set(observed)
    else:
        active_parents = set()

    # it's possible to go downward if and only if not an observed node
    if end_of_path in observed:
        active_children = set()
    else:
        active_children = set(bn.get_children(end_of_path))

    active_neighbours = active_parents.union(active_children)
    new_active_neighbours = active_neighbours - set(path)
    return new_active_neighbours


def _find_active_path_recurse(bn: BayesianModel, path: List[str],
                              end_node: str, observed: List[str]) -> List[str]:
    """Find active path from `path' to `end_node' given `observed'"""
    if path[-1] == end_node and end_node not in observed:
        return path
    else:
        neighbours = _active_neighbours(bn, path, observed)
        for neighbour in neighbours:
            ext = _find_active_path_recurse(bn, path + [neighbour], end_node, observed)
            if ext:
                return ext


def find_active_path(bn: BayesianModel, start_node: str, end_node: str, observed: List[str]) -> List[str]:
    """Find active path from `start_node' to `end_node' given `observed'"""
    return _find_active_path_recurse(bn, [start_node], end_node, observed)


def get_motifs(cid, path):
    shapes = []
    for i in range(len(path)):
        if i == 0:
            if path[i] in cid.get_parents(path[i+1]):
                shapes.append('forward')
            else:
                shapes.append('backward')
        else:
            shapes.append(get_motif(cid, path, i))
    return shapes


def get_motif(cid, path: List[str], i):
    """
    Classify three node structure as a forward (chain), backward (chain), fork or collider.
    """
    if len(path) == i+1:
        return "endpoint"

    elif cid.has_edge(path[i-1], path[i]) and cid.has_edge(path[i], path[i+1]):
        return "forward"

    elif cid.has_edge(path[i+1], path[i]) and cid.has_edge(path[i], path[i-1]):
        return "backward"

    elif cid.has_edge(path[i-1], path[i]) and cid.has_edge(path[i+1], path[i]):
        return "collider"

    elif cid.has_edge(path[i], path[i-1]) and cid.has_edge(path[i], path[i+1]):
        return "fork"

    else:
        ValueError(f"unsure how to classify this path at index {i}")
