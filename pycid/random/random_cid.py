import random
from typing import List, Tuple

from pycid.core.cid import CID
from pycid.core.cpd import DecisionDomain
from pycid.core.get_paths import find_active_path
from pycid.random.random_cpd import random_cpd

# TODO add a random_macid function


def random_cid(
    n_all: int,
    n_decisions: int,
    n_utilities: int,
    edge_density: float = 0.4,
    add_sr_edges: bool = True,
    add_cpds: bool = True,
    seed: int = None,
) -> CID:
    """Generates a random Cid with the specified number of nodes and edges"""

    all_names, decision_names, utility_names = get_node_names(n_all, n_decisions, n_utilities)
    edges = get_edges(all_names, utility_names, edge_density, seed=seed, allow_u_edges=False)
    cid = CID(edges, decision_names, utility_names)

    for uname in utility_names:
        for edge in edges:
            assert uname != edge[0]

    for i, d1 in enumerate(decision_names):
        for j, d2 in enumerate(decision_names[i + 1 :]):
            assert d2 not in cid._get_ancestors_of(d1)

    if add_sr_edges:
        add_sufficient_recalls(cid)

    if add_cpds:
        for node in cid.nodes:
            if node in cid.decisions:
                cid.add_cpds(DecisionDomain(node, [0, 1]))
            else:
                cid.add_cpds(random_cpd(node))
    return cid


def random_cids(
    ns_range: Tuple[int, int] = (14, 20),
    nd_range: Tuple[int, int] = (4, 7),
    nu_range: Tuple[int, int] = (4, 7),
    edge_density: float = 0.4,
    n_cids: int = 10,
    seed: int = None,
    add_sr_edges: bool = True,
) -> List[CID]:
    """generates a bunch of CIDs with sufficient recall

    if add_sr_edges=True, then sufficient recall is ensured by adding edges
    otherwise it is ensured by resampling graphs"""
    cids: List[CID] = []

    while len(cids) < n_cids:
        n_all = random.randint(*ns_range)
        n_decisions = random.randint(*nd_range)
        n_utilities = random.randint(*nu_range)

        cid = random_cid(n_all, n_decisions, n_utilities, edge_density, add_sr_edges=add_sr_edges, seed=seed)

        for uname in cid.utilities:
            for edge in cid.edges:
                assert uname != edge[0]
        if cid.sufficient_recall():
            cids.append(cid)

    return cids


def get_node_names(n_all: int, n_decisions: int, n_utilities: int) -> Tuple[List[str], List[str], List[str]]:
    """random lists of node names for decision, utility, and chance nodes"""
    n_structural = n_all - n_decisions - n_utilities
    structure_names = ["S{}".format(i) for i in range(n_structural)]
    decision_names = ["D{}".format(i) for i in range(n_decisions)]
    utility_names = ["U{}".format(i) for i in range(n_utilities)]

    # scramble decision and structure nodes, without upsetting internal orders
    non_utility_names = []
    si = 0
    di = 0
    while si < len(structure_names) - 1 and di < len(decision_names):
        if random.random() > 0.5:
            non_utility_names.append(structure_names[si])
            si += 1
        else:
            non_utility_names.append(decision_names[di])
            di += 1
    non_utility_names = non_utility_names + decision_names[di:] + structure_names[si:]
    return non_utility_names + utility_names, decision_names, utility_names


def get_edges(
    names: List[str], utility_names: List[str], edge_density: float, seed: int = None, allow_u_edges: bool = False
) -> List[Tuple[str, str]]:
    random.seed(seed)
    edges = []
    nodes_with_edges = set()
    for i, name1 in enumerate(names):
        for name2 in names[i + 1 :]:
            if random.random() < edge_density:
                if allow_u_edges or name1 not in utility_names:
                    edges.append((name1, name2))
                    nodes_with_edges.add(name1)
                    nodes_with_edges.add(name2)

        while name1 not in nodes_with_edges:
            other_node_indices = list(set(range(len(names))) - {i})
            j = random.choice(other_node_indices)
            if i < j:
                if allow_u_edges or (name1 not in utility_names):
                    edges.append((name1, names[j]))
                    nodes_with_edges.add(name1)
            else:
                if allow_u_edges or (names[j] not in utility_names):
                    edges.append((names[j], name1))
                    nodes_with_edges.add(name1)
    return edges


def _add_sufficient_recall(cid: CID, dec1: str, dec2: str, utility_node: str) -> None:
    """Add edges to a cid until `dec2` has sufficient recall of `dec1` (to optimize utility)

    this is done by adding edges from non-collider nodes until recall is adequate
    """

    if dec2 in cid._get_ancestors_of(dec1):
        raise ValueError("{} is an ancestor of {}".format(dec2, dec1))

    cid2 = cid.copy()
    cid2.add_edge("pi", dec1)

    # pygmpy is_active_trail does not accept a set (and then it does membership checks on the list...)
    while cid2.is_active_trail("pi", utility_node, observed=cid.get_parents(dec2) + [dec2]):
        path = find_active_path(cid2, "pi", utility_node, {dec2, *cid.get_parents(dec2)})
        if path is None:
            raise RuntimeError("couldn't find path even though there should be an active trail")
        while True:
            i = random.randrange(1, len(path) - 1)
            # print('consider {}--{}--{}'.format(path[i-1], path[i], path[i+1]),end='')
            collider = ((path[i - 1], path[i]) in cid2.edges) and ((path[i + 1], path[i]) in cid2.edges)
            if not collider:
                if dec2 not in cid2._get_ancestors_of(path[i]):
                    # print('add {}->{}'.format(path[i], dec2), end=' ')
                    cid.add_edge(path[i], dec2)
                    cid2.add_edge(path[i], dec2)
                    break


def add_sufficient_recalls(cid: CID) -> None:
    """add edges to a cid until all decisions have sufficient recall of all prior decisions"""
    decisions = list(cid.decisions)
    for utility_node in cid.utilities:
        # decisions = cid._get_valid_order(cid.decisions)  # cannot be trusted...
        for i, dec1 in enumerate(decisions):
            for dec2 in decisions[i + 1 :]:
                if dec1 in cid._get_ancestors_of(dec2):
                    _add_sufficient_recall(cid, dec1, dec2, utility_node)
                else:
                    _add_sufficient_recall(cid, dec2, dec1, utility_node)
