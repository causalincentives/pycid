# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.

from core.macid_base import MACIDBase
from typing import List, Set, Tuple
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
    considered_nodes = set(observed).union({start_node}, {end_node})
    for node in considered_nodes:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")
    
    return _find_active_path_recurse(mb, [start_node], end_node, observed)


def get_motif(mb: MACIDBase, path: List[str], idx: int) -> str:
    """
    Classify three node structure as a forward (chain), backward (chain), fork or collider at index 'idx' along path.
    """
    for node in path:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")

    if idx > len(path) - 1:
        raise Exception(f"The given index {idx} is not valid for the length of this path {len(path)}")

    if len(path) == idx + 1:
        return 'endpoint'

    elif mb.has_edge(path[idx - 1], path[idx]) and mb.has_edge(path[idx], path[idx + 1]):
        return 'forward'

    elif mb.has_edge(path[idx + 1], path[idx]) and mb.has_edge(path[idx], path[idx - 1]):
        return 'backward'

    elif mb.has_edge(path[idx - 1], path[idx]) and mb.has_edge(path[idx + 1], path[idx]):
        return 'collider'

    elif mb.has_edge(path[idx], path[idx - 1]) and mb.has_edge(path[idx], path[idx + 1]):
        return 'fork'

    else:
        ValueError(f"unsure how to classify this path at index {idx}")


def get_motifs(mb: MACIDBase, path: List[str]) -> List[str]:
    for node in path:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")
    
    shapes = []
    for i in range(len(path)):
        if i == 0:
            if path[i] in mb.get_parents(path[i + 1]):
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
    considered_nodes = {start}.union({finish})
    for node in considered_nodes:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")
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
    considered_nodes = {start}.union({finish})
    for node in considered_nodes:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")
    all_paths = []
    return _find_undirpath_recurse(mb, [start], finish, all_paths)


def directed_decision_free_path(mb: MACIDBase, start: str, finish: str) -> bool:
    """
    Checks to see if a directed decision free path exists
    """
    considered_nodes = {start}.union({finish})
    for node in considered_nodes:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")

    start_finish_paths = find_all_dir_paths(mb, start, finish)
    dec_free_path_exists = any(set(mb.all_decision_nodes).isdisjoint(set(path[1:-1]))
                               for path in start_finish_paths)  # ignore path's start and finish node
    if start_finish_paths and dec_free_path_exists:
        return True
    else:
        return False


def _get_path_structure(mb: MACIDBase, path: List[str]) -> List[Tuple[str, str]]:
    """
    returns the path's structure (ie pairs showing the direction of the edges that make up this path)
    If a path is D1 -> X <- D2, this function returns: [('D1', 'X'), ('D2', 'X')]
    """
    # TODO Tom: Is the docstring wrong? Should it be [('D1', 'X'), ('X', 'D2')]?
    #           In that case, would a name like _get_path_edges be clearer?
    structure = []
    for i in range(len(path) - 1):
        if path[i] in mb.get_parents(path[i + 1]):
            structure.append((path[i], path[i + 1]))
        elif path[i + 1] in mb.get_parents(path[i]):
            structure.append((path[i + 1], path[i]))
    return structure


def path_d_separated_by_Z(mb: MACIDBase, path: List[str], Z: List[str] = []) -> bool:
    """
    Check if a path is d-separated by the set of variables Z.
    """
    # TODO Tom: Terminology. Paths are active or not, while nodes are d-separated or d-connected.
    #           A better name might be path_is_active. Also, a better name for Z would be
    #           "observed".
    considered_nodes = set(path).union(set(Z))
    for node in considered_nodes:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")

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
    - A frontdoor path between X and Z is an (undirected) path in which the first edge comes
    out of the first node (X→···Z).
    """
    # TODO Tom: What does it mean that the frontdoor path is indirect?
    #           It seems similar to mb.is_active_trail(), so would be good if the name reflected that.
    considered_nodes = set(W).union({start}, {finish})
    for node in considered_nodes:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")

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
    # TODO Tom: This method doesn't appear to be heavily used, and is anyway basically a one
    #           one line operation list(set(Y_parents).difference(set(nx.descendants(mb, X))).
    #           I'm not sure it's justified to keep this as a separate method. Might be better
    #           to delete, and just use that oneliner when needed.
    considered_nodes = {Y}.union({X})
    for node in considered_nodes:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")

    Y_parents = mb.get_parents(Y)
    X_descendants = list(nx.descendants(mb, X))
    return list(set(Y_parents).difference(set(X_descendants)))


def backdoor_path_active_when_conditioning_on_W(mb: MACIDBase, start: str, finish: str, W: List[str] = []) -> bool:
    """
    Returns true if there is a backdoor path that's active when conditioning on nodes in set W.
    - A backdoor path between X and Z is an (undirected) path in which the first edge goes into the first node (X←···Z)
    """
    # TODO Tom: This is similar to mb.is_active_trail(). It would be good if the name reflected that,
    #           e.g. is_active_backdoor_trail().
    considered_nodes = set(W).union({start}, {finish})
    for node in considered_nodes:
        if node not in mb.nodes():
            raise Exception(f"The node {node} is not in the (MA)CID")

    start_finish_paths = find_all_undir_paths(mb, start, finish)
    for path in start_finish_paths:

        if len(path) > 1:   # must have path of at least 2 nodes
            is_backdoor_path = path[1] in mb.get_parents(path[0])
            not_blocked_by_W = not path_d_separated_by_Z(mb, path, W)
            if is_backdoor_path and not_blocked_by_W:
                return True
    else:
        return False
