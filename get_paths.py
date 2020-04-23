#Licensed to the Apache Software Foundation (ASF) under one or more contributor license 
#agreements; and to You under the Apache License, Version 2.0.

from typing import List
import numpy as np


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
            return {'control':control_path, 'info':info_path}
    return #if no path present

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


