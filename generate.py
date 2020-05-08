#Licensed to the Apache Software Foundation (ASF) under one or more contributor license 
#agreements; and to You under the Apache License, Version 2.0.

import random
from typing import List, Tuple
from cid import CID, NullCPD
from get_paths import find_active_path_recurse
import networkx as nx


def get_node_names(n_all: int, n_decisions: int, n_utilities: int):
    n_structural = n_all - n_decisions - n_utilities
    snames = ['S{}'.format(i) for i in range(n_structural)]
    dnames = ['D{}'.format(i) for i in range(n_decisions)]
    unames = ['U{}'.format(i) for i in range(n_utilities)]
    allnames = snames+dnames+unames
    random.shuffle(allnames)
    return allnames, dnames, unames

def get_edges(names: List[str], edge_density: float, seed=None):
    random.seed(seed)
    edges = []
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            if random.random()<edge_density:
                edges.append((name1, name2))

        #connect any nodes lacking edges
        nodes_with_edges = [i for j in names[i+1:] for i in j]
        if name1 not in nodes_with_edges:
            other_node_indices = list(set(range(len(names)))-set([i]))
            j = random.choice(other_node_indices)
            if i<j:
                edges.append((name1, names[j]))
            else:
                edges.append((names[j], name1))
    return edges

def random_cid(
        n_all:int, 
        n_decisions:int, 
        n_utilities:int, 
        edge_density:float=0.4,
        seed:int=None):
    
    allnames, dnames, unames = get_node_names(n_all, n_decisions, n_utilities)
    edges = get_edges(allnames, edge_density, seed=seed)
    
    cid = CID(edges, unames)
    nullcpds = [NullCPD(dname, 0) for dname in dnames]
    cid.add_cpds(*nullcpds)
    return cid


def random_cids(
        ns_range:Tuple[int, int]=(14,20),
        nd_range:Tuple[int, int]=(4,7),
        nu_range:Tuple[int, int]=(4,7),
        edge_density:float=.4,
        n_cids:int=10,
        seed:int=None,
        add_sr_edges=True,
        ):
    # generates a bunch of CID skeletons with sufficient recall
    # if add_sr_edges=True, then sufficient recall is ensured by adding edges
    # otherwise it is ensured by resampling graphs
    cids = []    

    while len(cids) < n_cids:
        n_all = random.randint(*ns_range)
        n_decisions = random.randint(*nd_range)
        n_utilities = random.randint(*nu_range)

        cid = random_cid(n_all, n_decisions, n_utilities, edge_density, seed=seed)

        if add_sr_edges:
            cid = add_sufficient_recalls(cid)
            cids.append(cid)
        else:
            if cid.check_sufficient_recall():
                cids.append(cid)

    return cids

def _add_sufficient_recall(cid, dec1, dec2, utility):
    #adds edges to a cid until `dec2` has recall of `dec1` that is
    #sufficient to optimize `utility`
    #this is done by adding edges from non-collider nodes until recall is adequate

    if dec2 in cid._get_ancestors_of(dec1):
        raise ValueError('{} is an ancestor of {}'.format(dec2, dec1))

    
    cid2 = cid.copy()
    cid2.add_edge('pi',dec1)
    if not cid2.is_active_trail('pi', utility, observed=cid2.get_parents(dec2) + [dec2]): #recall is already sufficient
        cid2.remove_node('pi')
        return cid2

    while cid2.is_active_trail('pi', utility, observed=cid2.get_parents(dec2) + [dec2]):
        path = find_active_path_recurse(cid2, ['pi'], utility, cid2.get_parents(dec2) + [dec2])
        i = random.randrange(1, len(path)-1)
        #print('consider {}--{}--{}'.format(path[i-1], path[i], path[i+1]),end='')
        chain_or_fork = ((path[i], path[i-1]) in cid2.edges) or ((path[i], path[i+1]) in cid2.edges)
        if chain_or_fork:
            if dec2 not in cid2._get_ancestors_of(path[i]):
                #print('add {}->{}'.format(path[i], dec2), end=' ')
                cid2.add_edge(path[i], dec2)

    #remove \Pi and (\Pi,D)
    cid2.remove_node('pi')
    return cid2

def add_sufficient_recalls(cid):
    #adds edges to a cid until all decisions have sufficient recall of all prior decisions
    for utility in cid.utilities:
        decisions = [d for d in cid._get_ancestors_of(utility) if d in cid._get_decisions()]
        decisions = cid._get_valid_order(decisions)
        for i, dec1 in enumerate(decisions):
            for dec2 in decisions[i+1:]:
                cid = _add_sufficient_recall(cid, dec1, dec2, utility)
    return cid
