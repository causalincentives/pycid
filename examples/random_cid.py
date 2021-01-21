# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.

import random
from typing import List, Tuple
from core.cid import CID
from analyze.get_paths import find_active_path


def random_cid(
        n_all: int,
        n_decisions: int,
        n_utilities: int,
        edge_density: float = 0.4,
        add_sr_edges: bool = True,
        seed: int = None):
    """examples a random cid with the specified number of nodes and edges"""

    allnames, dnames, unames = get_node_names(n_all, n_decisions, n_utilities)
    edges = get_edges(allnames, unames, edge_density, seed=seed, allow_u_edges=False)
    cid = CID(edges, dnames, unames)

    for uname in unames:
        for edge in edges:
            assert uname != edge[0]

    for i, d1 in enumerate(dnames):
        for j, d2 in enumerate(dnames[i+1:]):
            if d2 in cid._get_ancestors_of(d1):
                raise Exception("misordered decisions")

    if add_sr_edges:
        add_sufficient_recalls(cid)

    return cid


def random_cids(
        ns_range: Tuple[int, int] = (14, 20),
        nd_range: Tuple[int, int] = (4, 7),
        nu_range: Tuple[int, int] = (4, 7),
        edge_density: float = .4,
        n_cids: int = 10,
        seed: int = None,
        add_sr_edges=True,
        ):
    """generates a bunch of CIDs with sufficient recall

    if add_sr_edges=True, then sufficient recall is ensured by adding edges
    otherwise it is ensured by resampling graphs"""
    cids = []

    while len(cids) < n_cids:
        n_all = random.randint(*ns_range)
        n_decisions = random.randint(*nd_range)
        n_utilities = random.randint(*nu_range)

        cid = random_cid(n_all, n_decisions, n_utilities, edge_density,
                         add_sr_edges=add_sr_edges, seed=seed)

        for uname in cid.utility_nodes:
            for edge in cid.edges:
                assert uname != edge[0]
        if cid.check_sufficient_recall():
            cids.append(cid)

    return cids


def get_node_names(n_all: int, n_decisions: int, n_utilities: int):
    """examples lists of node names for decision, utility, and chance nodes"""
    n_structural = n_all - n_decisions - n_utilities
    snames = ['S{}'.format(i) for i in range(n_structural)]
    dnames = ['D{}'.format(i) for i in range(n_decisions)]
    unames = ['U{}'.format(i) for i in range(n_utilities)]
    nonunames = snames + dnames

    # scramble decision and structure nodes, without upsetting internal orders
    nonunames = []
    si = 0
    di = 0
    while si < len(snames) - 1 and di < len(dnames):
        if random.random() > 0.5:
            nonunames.append(snames[si])
            si += 1
        else:
            nonunames.append(dnames[di])
            di += 1
    nonunames = nonunames + dnames[di:] + snames[si:]
    return nonunames + unames, dnames, unames


def get_edges(
        names: List[str],
        unames: List[str],
        edge_density: float,
        seed=None,
        allow_u_edges=False,
        ):
    random.seed(seed)
    edges = []
    nodes_with_edges = set()
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            if random.random() < edge_density:
                if allow_u_edges or name1 not in unames:
                    edges.append((name1, name2))
                    nodes_with_edges.add(name1)
                    nodes_with_edges.add(name2)

        while name1 not in nodes_with_edges:
            other_node_indices = list(set(range(len(names))) - {i})
            j = random.choice(other_node_indices)
            if i < j:
                if allow_u_edges or (name1 not in unames):
                    edges.append((name1, names[j]))
                    nodes_with_edges.add(name1)
            else:
                if allow_u_edges or (names[j] not in unames):
                    edges.append((names[j], name1))
                    nodes_with_edges.add(name1)
    return edges


def _add_sufficient_recall(cid: CID, dec1: str, dec2: str, utility_node: str) -> None:
    """Add edges to a cid until `dec2` has sufficient recall of `dec1` (to optimize utility)

    this is done by adding edges from non-collider nodes until recall is adequate
    """

    if dec2 in cid._get_ancestors_of(dec1):
        raise ValueError('{} is an ancestor of {}'.format(dec2, dec1))

    cid2 = cid.copy()
    cid2.add_edge('pi', dec1)

    while cid2.is_active_trail('pi', utility_node, observed=cid.get_parents(dec2) + [dec2]):
        path = find_active_path(cid2, 'pi', utility_node, cid.get_parents(dec2) + [dec2])
        if path is None:
            raise Exception("couldn't find path even though there should be an active trail")
        while True:
            i = random.randrange(1, len(path) - 1)
            # print('consider {}--{}--{}'.format(path[i-1], path[i], path[i+1]),end='')
            collider = ((path[i-1], path[i]) in cid2.edges) and ((path[i+1], path[i]) in cid2.edges)
            if not collider:
                if dec2 not in cid2._get_ancestors_of(path[i]):
                    # print('add {}->{}'.format(path[i], dec2), end=' ')
                    cid.add_edge(path[i], dec2)
                    cid2.add_edge(path[i], dec2)
                    break


def add_sufficient_recalls(cid: CID) -> None:
    """add edges to a cid until all decisions have sufficient recall of all prior decisions"""
    for utility_node in cid.utility_nodes:
        #decisions = cid._get_valid_order(cid.decision_nodes)  # cannot be trusted...
        for i, dec1 in enumerate(cid.decision_nodes):
            for dec2 in cid.decision_nodes[i+1:]:
                if dec1 in cid._get_ancestors_of(dec2):
                    _add_sufficient_recall(cid, dec1, dec2, utility_node)
                else:
                    _add_sufficient_recall(cid, dec2, dec1, utility_node)

