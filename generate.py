import random
from typing import List, Tuple
from cid import CID, NullCPD


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
        seed:int=None):
    # generates a bunch of CID skeletons with sufficient recall
    cids = []    

    while len(cids) < n_cids:
        n_all = random.randint(*ns_range)
        n_decisions = random.randint(*nd_range)
        n_utilities = random.randint(*nu_range)

        cid = random_cid(n_all, n_decisions, n_utilities, edge_density, seed=seed)

        if cid.check_sufficient_recall():
            cids.append(cid)

    return cids
