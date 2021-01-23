# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.

from core.macid_base import MACIDBase
from typing import List, Set, Tuple
from pgmpy.models import BayesianModel
import networkx as nx


def _active_neighbours(mb: MACIDBase, path: List[str], observed: List[str]) -> Set[str]:
    """Find possibly active extensions of path conditional on observed"""
    end_of_path = path[-1]
    last_forward = len(path) > 1 and end_of_path in mb.get_children(path[-2])
    possible_colliders = set().union(*[set(mb._get_ancestors_of(e)) for e in observed])

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


def _find_active_path_recurse(mb: MACIDBase, path: List[str],
                              end_node: str, observed: List[str]) -> List[str]:
    """Find active path from `path' to `end_node' given `observed'"""
    if path[-1] == end_node and end_node not in observed:
        return path
    else:
        neighbours = _active_neighbours(mb, path, observed)
        for neighbour in neighbours:
            ext = _find_active_path_recurse(mb, path + [neighbour], end_node, observed)
            if ext:
                return ext


def find_active_path(mb: MACIDBase, start_node: str, end_node: str, observed: List[str] = []) -> List[str]:
    """Find active path from `start_node' to `end_node' given `observed'"""
    return _find_active_path_recurse(mb, [start_node], end_node, observed)


def get_motif(mb: MACIDBase, path: List[str], idx: int) -> str:
    """
    Classify three node structure as a forward (chain), backward (chain), fork or collider at index 'idx' along the path.
    """
    if len(path) == idx+1:
        return 'endpoint'

    elif mb.has_edge(path[idx-1], path[idx]) and mb.has_edge(path[idx], path[idx+1]):
        return 'forward'

    elif mb.has_edge(path[idx+1], path[idx]) and mb.has_edge(path[idx], path[idx-1]):
        return 'backward'

    elif mb.has_edge(path[idx-1], path[idx]) and mb.has_edge(path[idx+1], path[idx]):
        return 'collider'

    elif mb.has_edge(path[idx], path[idx-1]) and mb.has_edge(path[idx], path[idx+1]):
        return 'fork'

    else:
        ValueError(f"unsure how to classify this path at index {idx}")


def get_motifs(mb: MACIDBase, path: List[str]) -> List[str]:
    shapes = []
    for i in range(len(path)):
        if i == 0:
            if path[i] in mb.get_parents(path[i+1]):
                shapes.append('forward')
            else:
                shapes.append('backward')
        else:
            shapes.append(get_motif(mb, path, i))
    return shapes


def _find_dirpath_recurse(mb: MACIDBase, path: List[str], finish: str, all_paths: List[List[str]]) -> List[List[str]]:

    if path[-1] == finish:
        return path
    else:
        children = mb.get_children(path[-1])
        for child in children:
            ext = path + [child]
            ext = _find_dirpath_recurse(mb, ext, finish, all_paths)
            if ext and ext[-1] == finish:  # the "if ext" checks to see that it's a full directed path.
                all_paths.append(ext)
            else:
                continue
        return all_paths


def find_all_dir_paths(mb: MACIDBase, start: str, finish: str) -> List[List[str]]:
    """
    Finds all directed paths from start node to finish node that exist in the MAID.
    """
    all_paths = []
    return _find_dirpath_recurse(mb, [start], finish, all_paths)


def _find_undirpath_recurse(mb: MACIDBase, path: List[str], finish: str, all_paths: str) -> List[List[str]]:

    if path[-1] == finish:
        return path
    else:
        neighbours = list(mb.get_children(path[-1])) + list(mb.get_parents(path[-1]))
        new = set(neighbours).difference(set(path))
        for child in new:
            ext = path + [child]
            ext = _find_undirpath_recurse(mb, ext, finish, all_paths)
            if ext and ext[-1] == finish:  # the "if ext" checks to see that it's a full path.
                all_paths.append(ext)
            else:
                continue
        return all_paths


def find_all_undir_paths(mb: MACIDBase, start: str, finish: str) -> List[List[str]]:
    """
    Finds all paths from start node to end node that exist in the MAID
    """
    all_paths = []
    return _find_undirpath_recurse(mb, [start], finish, all_paths)


def directed_decision_free_path(mb: MACIDBase, start: str, finish: str) -> bool:
    """
    Checks to see if a directed decision free path exists
    """
    start_finish_paths = find_all_dir_paths(mb, start, finish)
    dec_free_path_exists = any(set(mb.all_decision_nodes).isdisjoint(set(path[1:-1])) for path in start_finish_paths)  # ignore path's start and finish node
    if start_finish_paths and dec_free_path_exists:
        return True
    else:
        return False

def _get_path_structure(mb: MACIDBase, path: List[str]) -> List[Tuple[str, str]]:
    """
    returns the path's structure (ie pairs showing the direction of the edges that make up this path)

    If a path is D1 -> X <- D2, this function returns: [('D1', 'X'), ('D2', 'X')]
    """
    structure = []
    for i in range(len(path)-1):
        if path[i] in mb.get_parents(path[i+1]):
            structure.append((path[i], path[i+1]))
        elif path[i+1] in mb.get_parents(path[i]):
            structure.append((path[i+1], path[i]))
    return structure


def path_d_separated_by_Z(mb: MACIDBase, path: List[str], Z: List[str] = []) -> bool:
    """
    Check if a path is d-separated by the set of variables Z.
    """
    if len(path) < 3:
        return False

    for _, b, _ in zip(path[:-2], path[1:-1], path[2:]):
        structure = get_motif(mb, path, path.index(b))

        if structure in {'fork', 'forward', 'backward'} and b in Z:
            return True

        if structure == "collider":
            descendants = nx.descendants(mb, b).union({b})
            if not descendants.intersection(set(Z)):
                return True

    return False


def frontdoor_indirect_path_not_blocked_by_W(mb: MACIDBase, start: str, finish: str, W: List[str] = []) -> bool:
    """
    checks whether an indirect frontdoor path exists that isn't blocked by the nodes in set W.  
    - A frontdoor path between X and Z is an (undirected) path in which the first edge comes out of the first node (X→···Z).
    """
    
    start_finish_paths = find_all_undir_paths(mb, start, finish)
    for path in start_finish_paths:
        is_frontdoor_path = path[0] in mb.get_parents(path[1])
        not_blocked_by_W = not path_d_separated_by_Z(mb, path, W)
        contains_collider = "collider" in get_motifs(mb, path)
        # default is False since if w = [], any unobserved collider blocks path
        if is_frontdoor_path and not_blocked_by_W and contains_collider:   
            return True
    else:
        return False


def parents_of_Y_not_descended_from_X(mb: MACIDBase, Y: str, X: str) -> List[str]:
    """
    Finds the parents of Y not descended from X
    """
    Y_parents = mb.get_parents(Y)
    X_descendants = list(nx.descendants(mb, X))
    return list(set(Y_parents).difference(set(X_descendants)))


def backdoor_path_active_when_conditioning_on_W(mb: MACIDBase, start: str, finish: str, W: List[str] = []) -> bool:
    """
    Returns true if there is a backdoor path that's active when conditioning on nodes in set W.
    - A backdoor path between X and Z is an (undirected) path in which the first edge goes into the first node (X←···Z)
    """
    start_finish_paths = find_all_undir_paths(mb, start, finish)
    for path in start_finish_paths:

        if len(path) > 1:   # must have path of at least 2 nodes
            is_backdoor_path = path[1] in mb.get_parents(path[0])
            not_blocked_by_W = not path_d_separated_by_Z(mb, path, W)
            if is_backdoor_path and not_blocked_by_W:
                return True
    else:
        return False

