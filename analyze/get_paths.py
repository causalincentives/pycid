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



# TODO add tests for these
# -------- Methods for finding MACID graphical properties --------------------

def _find_dirpath_recurse(self, path: List[str], finish: str, all_paths):

    if path[-1] == finish:
        return path
    else:
        children = self.get_children(path[-1])
        for child in children:
            ext = path + [child]
            ext = self._find_dirpath_recurse(ext, finish, all_paths)
            if ext and ext[-1] == finish:  # the "if ext" checks to see that it's a full directed path.
                all_paths.append(ext)
            else:
                continue
        return all_paths

def find_all_dir_path(self, start, finish):
    """
    finds all direct paths from start node to end node that exist in the MAID
    """
    all_paths = []
    return self._find_dirpath_recurse([start], finish, all_paths)



def _find_undirpath_recurse(self, path: List[str], finish: str, all_paths: str):

    if path[-1] == finish:
        return path
    else:
        neighbours = list(self.get_children(path[-1])) + list(self.get_parents(path[-1]))
        new = set(neighbours).difference(set(path))
        for child in new:
            ext = path + [child]
            ext = self._find_undirpath_recurse(ext, finish, all_paths)
            if ext and ext[-1] == finish:  # the "if ext" checks to see that it's a full directed path.
                all_paths.append(ext)
            else:
                continue
        return all_paths

def find_all_undir_path(self, start: str, finish: str):
    """
    finds all direct paths from start node to end node that exist in the MAID
    """
    all_paths = []
    return self._find_undirpath_recurse([start], finish, all_paths)


def _directed_decision_free_path(self, start: str, finish: str):
    """
    checks to see if a directed decision free path exists
    """
    start_finish_paths = self.find_all_dir_path(start, finish)
    dec_free_path_exists = any(set(self.all_decision_nodes).isdisjoint(set(path[1:-1])) for path in start_finish_paths)  # ignore path's start and finish node
    if start_finish_paths and dec_free_path_exists:
        return True
    else:
        return False


def _get_path_structure(self, path:List[str]):
    """
    returns the path's structure (ie the direction of the edges that make up this path)
    """
    structure = []
    for i in range(len(path)-1):
        if path[i] in self.get_parents(path[i+1]):
            structure.append((path[i], path[i+1]))
        elif path[i+1] in self.get_parents(path[i]):
            structure.append((path[i+1], path[i]))
    return structure

def path_d_separated_by_Z(self, path:List[str], Z:List[str]=[]):
    """
    Check if a path is d-separated by set of variables Z.
    """
    if len(path) < 3:
        return False

    for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
        structure = get_motif(self, path, path.index(b))

        if structure in ("chain", "fork") and b in Z:
            return True

        if structure == "collider":
            descendants = (nx.descendants(self, b) | {b})
            if not descendants & set(Z):
                return True

    return False

def frontdoor_indirect_path_not_blocked_by_W(self, start: str, finish: str, W:List[str]=[]):
    """checks whether an indirect frontdoor path exists that isn't blocked by the nodes in set W."""
    start_finish_paths = self.find_all_undir_path(start, finish)
    for path in start_finish_paths:
        is_frontdoor_path = path[0] in self.get_parents(path[1])
        not_blocked_by_W = not self.path_d_separated_by_Z(path, W)
        contains_collider = "collider" in get_motifs(self, path)
        if is_frontdoor_path and not_blocked_by_W and contains_collider:   #default (if w = [] is going to be false since any unobserved collider blocks path
            return True
    else:
        return False


def parents_of_Y_not_descended_from_X(self, X: str,Y: str):
    """finds the parents of Y not descended from X"""
    Y_parents = self.get_parents(Y)
    X_descendants = list(nx.descendants(self, X))
    print(f" desc of {X} are {X_descendants}")
    return list(set(Y_parents).difference(set(X_descendants)))


def get_key_node(self, path:List[str]):
    """ The key node of a path is the first "fork" node in the path"""
    for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
        structure = get_motif(self, path, path.index(b))
        if structure == "fork":
            return b

def backdoor_path_active_when_conditioning_on_W(self, start: str, finish: str, W:List[str]=[]):
    """
    returns true if there is a backdoor path that's active when conditioning on nodes in set W.
    """
    start_finish_paths = self.find_all_undir_path(start, finish)
    for path in start_finish_paths:

        if len(path) > 1:   #must have path of at least 2 nodes
            is_backdoor_path = path[1] in self.get_parents(path[0])
            not_blocked_by_W = not self.path_d_separated_by_Z(path, W)
            if is_backdoor_path and not_blocked_by_W:
                return True

    else:
        return False

def backdoor_path_active_when_conditioning_on_W2(self, start: str, finish: str, W:List[str]=[]):
    """
    returns true if there is a backdoor path that's active when conditioning on nodes in set W.
    """

    start_finish_paths = self.find_all_undir_path(start, finish)
    for path in start_finish_paths:
        #print(f"path1 = {path}")
        if len(path) > 1:   #must have path of at least 2 nodes
            is_backdoor_path = path[1] in self.get_parents(path[0])
            #print(f"is_bd_path {is_backdoor_path}")
            not_blocked_by_W = not self.path_d_separated_by_Z(path, W)
            #print(f"not_blocked = {not_blocked_by_W}")
            if is_backdoor_path and not_blocked_by_W:
                #print(f"path is {path}")

                return True

    else:
        return False