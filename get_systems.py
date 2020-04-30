#Licensed to the Apache Software Foundation (ASF) under one or more contributor license 
#agreements; and to You under the Apache License, Version 2.0.

from get_paths import _get_path_pair, find_dirpath, _find_dirpath_recurse
import matplotlib.pyplot as plt

def is_backdoor(cid, info_path):
    return info_path[1] in cid.get_parents(info_path[0])

def is_directed(cid, info_path):
    for i in range(1,len(info_path)):
        A, B = info_path[i-1:i+1]
        if A not in cid.get_parents(B):
            return False
    return True

def get_c_index(cid, history_fragment, info_path):
    i_C = 0
    assert info_path[0]==history_fragment[-1]
    
    for i in range(min(len(info_path), len(history_fragment))):
        if info_path[i]==history_fragment[-i-1]:
            return i_C
        else:
            i_C+= 1
    return i_C

def augment_paths(cid, history_path, D, info_path, i_C0):
    X0 = info_path[0]
    C0 = info_path[i_C0]
    
    #follow the procedure for changing the path pair
    if is_backdoor(cid, info_path[i_C0:]) or is_directed(cid, info_path[i_C0:]):
        paths = {'history':history_path, 'info': info_path, 'i_C':i_C0}
        return paths
    else:
        #find C (first collider on infopath)
        for i in range(i_C0, len(info_path)):
            A, B = info_path[i:i+2]
            back_step = B in cid.get_parents(A)
            if back_step:
                i_C = i
                break
        #find path from C to parent of D
        C = info_path[i_C]
        C_to_D = _find_dirpath_recurse(cid, [C], D)
        for i, W in enumerate(C_to_D):
            if W in cid.get_parents(D):
                active_C_to_Pa = C_to_D[:i+1]
                break
        
        history_final = history_path[:-i_C0] + info_path[i_C0:i_C] + active_C_to_Pa
        info_final = active_C_to_Pa[::-1] + info_path[i_C+1:]
        paths = {'history':history_final, 'info':info_final, 'i_C':len(active_C_to_Pa)-1}
        return paths

def get_system(cid, history_fragment, D, X):
    # takes a history_fragment for (D,X) then
    # chooses good (control/info)paths and returns them
    paths = _get_path_pair(cid, D, X)
    i_C = get_c_index(cid, history_fragment, paths['info'])
    new_paths = augment_paths(cid, history_fragment, D, paths['info'], i_C)
    new_paths['control'] = paths['control']
    return new_paths

def _find_systems_along_history(cid, systems, full_history):
    new_history = []
    new_infolinks = []
    i = 0 #start of history fragment
    for j in range(1, len(full_history)):
        X, D = full_history[j-1:j+1]
        if D in cid._get_decisions() and X in cid.get_parents(D):
            history_fragment = full_history[i:j]
            paths = get_system(cid, history_fragment, D, X)
            new_history += paths.pop('history')
            X_new = paths['info'][0]
            systems.append(paths)
            new_infolinks.append((X_new,D))
            i = j
    if i: #decision encountered
        new_history += paths['control']
        return new_history, new_infolinks
    else:
        return full_history, new_infolinks

def _choose_systems_recurse(cid, systems, system_idx, history_pointer):
    systems[-1]['h_pointer'] = history_pointer
    for name in ['control', 'info']:
        history = systems[system_idx][name]
        new_history_path, new_infolinks = _find_systems_along_history(cid, systems, history)
        systems[system_idx][name] = new_history_path
        history_pointer = (system_idx, name)

        for j in range(1, len(new_infolinks) + 1):
            _choose_systems_recurse(cid, systems, system_idx+j, history_pointer)
    
def choose_systems(cid, decision, obs):
    #recursively choose paths where infopaths are directed or backdoor from combiner node C
    systems = []
    system = _get_path_pair(cid, decision, obs)
    assert system is not None, "paths not found from ({}->{}) to {}".format(obs, decision, cid.utilities)
    systems.append(system)
    
    _choose_systems_recurse(cid, systems, 0, None)
    return systems



def check_systems(cid, paths, decision, obs): 
    #TODO: also check that i_C is the node where the history encounters the string
    #check that all infolinks have their own paths, and that they're directed or backdoor from C
    paths = paths.copy()
    #check that all infolinks have their own paths
    for ptype in ['control', 'info']:
        for path in paths:
            for i in range(len(path)-3):
                X, D = path[ptype][i:i+2]
                if X in cid.get_parents(D) and D in cid._get_decisions():
                    if (X,D) not in [(path['info'][0],path['control'][0]) for path in paths]:
                        print("{} not in {}".format((X,D),paths.keys()))
                        return False
    paths.pop(0) #TODO: remove this (and args decision, obs) when (obs, decision) has an i_C
    for i, path in enumerate(paths):
        segment = path['info'][path['i_C']:]
        if not is_backdoor(cid, segment) and not is_directed(cid, segment):
            print("{} system:{} is front-door but undirected".format(i, segment))
            return False
    return True
        #check that all infopaths except that of (X,D) are directed or backdoor from C
