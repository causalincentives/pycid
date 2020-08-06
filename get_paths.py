#Licensed to the Apache Software Foundation (ASF) under one or more contributor license 
#agreements; and to You under the Apache License, Version 2.0.

from typing import List
import numpy as np

def get_motifs(cid, path):
    shapes = []
    for i in range(len(path)):
        if i==0:
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
            ValueError(f"unsure how to calssify this path at index {i}")
    

def _find_dirpath_recurse(bn, path: List[str], B: str):
    if path[-1]==B:
        return path
    else:
        children = bn.get_children(path[-1])
        for child in children:
            ext = path + [child]
            ext = _find_dirpath_recurse(bn, ext, B)
            if ext and ext[-1]==B:
                return ext

def find_dirpath(bn, A, B):
    return _find_dirpath_recurse(bn, [A], B)

def _active_neighbours(bn, path: List[str], E: List[str]):
    #find possibly active extensions of path conditional on E
    A = path[-1]
    last_forward = len(path) > 1 and A in bn.get_children(path[-2])
    
    if A in E: #implies that last step was forward
        active_children= []
        active_parents = [i for i in bn.get_parents(A) if i not in E]
    elif last_forward and not np.any([A in bn._get_ancestors_of(e) for e in E]):
        active_children = bn.get_children(A)
        active_parents = []
    else:
        active_children = bn.get_children(A)
        active_parents = [i for i in bn.get_parents(A) if i not in E]
        
    active_neighbours = active_parents + active_children
    new_active_neighbours = [i for i in active_neighbours if i not in path]
    return new_active_neighbours
    

def find_active_path_recurse(bn, path:List, B:str, E:List):
    #find active path from `path' to `B' given `E'
    if path[-1]==B and B not in E:
        return path
    else:
        neighbours = _active_neighbours(bn, path, E)
        for neighbour in neighbours:
            ext = path + [neighbour]
            ext = find_active_path_recurse(bn, ext, B, E)
            if ext and ext[-1]==B and B not in E:
                return ext
            
#def find_active_path(bn, A, B, E):
#    return _find_active_path_recurse(bn, [A], B, E)

def _get_path_pair(cid, D, X):
    assert (X,D) in cid.edges, '{} is not a parent of {}'.format(X, D)
    #remove observations of previous decisions
    cid = cid.copy()
    decisions = cid._get_valid_order(cid._get_decisions())
    ind = np.argmax(np.array(decisions)==D)
    prior_decisions = decisions[:ind]
    for edge in list(cid.edges).copy(): #avoid dict size changed during iteration error
        if edge[1] in prior_decisions:
            cid.remove_edge(*edge)

    #find infopath and control path to a downstream utility
    downstream_utilities = set([i for i in cid.utilities if D in cid._get_ancestors_of(i)])
    parents = cid.get_parents(D)
    other_parents = list(set(parents+[D]) - set([X]))
    for utility in downstream_utilities:
        info_path = find_active_path_recurse(cid, [X], utility, [D] + other_parents)
        if info_path:
            control_path = find_dirpath(cid, D, utility)
            obs_paths = _get_obs_paths(cid, info_path, D)
            return {'control':control_path, 'info':info_path, 'obs_paths':obs_paths}
    return #if no path present

def _get_active_dirpath(cid, A:List, D):
        A_to_D = _find_dirpath_recurse(cid, A, D)
        for i, W in enumerate(A_to_D):
            if W in cid.get_parents(D):
                return A_to_D


def _get_obs_paths(cid, info_path, D):
    #find active paths from other colliders to D
    motifs = get_motifs(cid, info_path)
    obs_paths = []
    for i in range(len(info_path)):
        motif = motifs[i]
        if motif=='c':
            coll_to_pa = _get_active_dirpath(cid, [info_path[i]], D)
            obs_paths.append(coll_to_pa[:len(coll_to_pa)])
    return obs_paths

def get_infolinks(cid, path):
    #extract infolinks from a path
    infolinks = []
    for start, end in zip(path[:-1], path[1:]):
        if end in cid._get_decisions():
            if start in cid.get_parents(end):
                infolinks.append((start, end))
    return infolinks

def choose_all_paths(cid, decision, obs):
    paths = {}
    pair = _get_path_pair(cid, decision, obs)
    assert pair is not None, "paths not found from ({}->{}) to {}".format(obs, decision, cid.utilities)
    paths[(obs, decision)] = pair
    infolinks = get_infolinks(cid, pair['control']) + get_infolinks(cid, pair['info'])
    new_infolinks = set(infolinks) - set(paths)
    while new_infolinks:
        for X, D in new_infolinks:
            if (X,D) not in paths.keys():
                paths[(X,D)] = _get_path_pair(cid, D, X)
        infolinks = get_infolinks(cid, pair['control']) + get_infolinks(cid, pair['info'])
        new_infolinks = set(infolinks) - set(paths)
    return paths


#TODO: why does step (1) advise removing all other nodes?
#TODO: why does step (1) advise choosing the info path so that X \neq S?


