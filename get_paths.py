#Licensed to the Apache Software Foundation (ASF) under one or more contributor license 
#agreements; and to You under the Apache License, Version 2.0.


def _find_path_recurse(bn, path: List[str], B: str):
    if path[-1]==B:
        return path
    else:
        children = bn.get_children(path[-1])
        for child in children:
            ext = path + [child]
            ext = _find_path_recurse(bn, ext, B)
            if ext and ext[-1]==B:
                return ext

def find_path_from(bn, A, B):
    return _find_path_recurse(bn, [A], B)

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
    

def _find_active_path_recurse(bn, path, B, E):
    if path[-1]==B and B not in E:
        return path
    else:
        neighbours = _active_neighbours(bn, path, E)
        for neighbour in neighbours:
            ext = path + [neighbour]
            ext = _find_active_path_recurse(bn, ext, B, E)
            if ext and ext[-1]==B and B not in E:
                return ext
            
def find_active_path(bn, A, B, E):
    return _find_active_path_recurse(bn, [A], B, E)

def choose_paths(cid, decision, X, utility):
    control_path = find_path_from(cid, decision, utility)
    other_parents = [i for i in cid.get_parents(decision) if i!=X]
    info_path = find_active_path(cid, [decision, X], utility, other_parents)
    return control_path, info_path


