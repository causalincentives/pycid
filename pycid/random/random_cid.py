from __future__ import annotations

import random
from typing import List, Tuple

import networkx as nx
from pgmpy.base.DAG import DAG

from pycid.core.cid import CID
from pycid.random.random_macidbase import add_random_cpds, random_macidbase


def random_cid(
    number_of_nodes: int = 8,
    number_of_decisions: int = 1,
    number_of_utilities: int = 1,
    add_cpds: bool = True,
    sufficient_recall: bool = False,
    edge_density: float = 0.4,
    max_in_degree: int = 4,
    max_resampling_attempts: int = 100,
) -> CID:
    """
    Generate a random CID.

    Parameters:
    -----------
    number_of nodes: The total number of nodes in the CID.

    number_of_decisions: The number of decisions in the CID.

    number_of_utilities: The number of utilities in the CID.

    add_cpds: True if we should pararemeterise the CID as a model.
    This adds [0,1] domains to every decision node and RandomCPDs to every utility and chance node in the CID.

    sufficient_recall: True the agent should have sufficient recall of all of its previous decisions.
    An Agent has sufficient recall in a CID if the relevance graph is acyclic.

    edge_density: The density of edges in the CID's DAG as a proportion of the maximum possible number of nodes
    in the DAG - n*(n-1)/2

    max_in_degree: The maximal number of edges incident to a node in the CID's DAG.

    max_resampling_attempts: The maxmimum number of resampling of random DAGs attempts in order to try
    to satisfy all constraints.

    Returns
    -------
    A CID that satisfies the given constraints or a ValueError if it was unable to meet the constraints in the
    specified number of attempts.

    """
    mb = random_macidbase(
        number_of_nodes=number_of_nodes,
        agent_decisions_num=(number_of_decisions,),
        agent_utilities_num=(number_of_utilities,),
        add_cpds=False,
        sufficient_recall=sufficient_recall,
        edge_density=edge_density,
        max_in_degree=max_in_degree,
        max_resampling_attempts=max_resampling_attempts,
    )

    dag = DAG(mb.edges)
    decision_nodes = mb.decisions
    utility_nodes = mb.utilities

    # change the naming style of decision and utility nodes
    dec_name_change = {old_dec_name: "D" + str(i) for i, old_dec_name in enumerate(decision_nodes)}
    util_name_change = {old_util_name: "U" + str(i) for i, old_util_name in enumerate(utility_nodes)}
    node_name_change_map = {**dec_name_change, **util_name_change}
    dag = nx.relabel_nodes(dag, node_name_change_map)

    cid = CID(dag.edges, decisions=list(dec_name_change.values()), utilities=list(util_name_change.values()))

    if add_cpds:
        add_random_cpds(cid)
    return cid


def random_cids(
    total_nodes_range: Tuple[int, int] = (10, 15),
    num_decs_range: Tuple[int, int] = (2, 4),
    num_utils_range: Tuple[int, int] = (2, 4),
    add_cpds: bool = True,
    sufficient_recall: bool = True,
    edge_density: float = 0.4,
    n_cids: int = 10,
) -> List[CID]:
    """Generates a number of CIDs with sufficient recall"""
    cids: List[CID] = []

    while len(cids) < n_cids:
        n_all = random.randint(*total_nodes_range)
        n_decisions = random.randint(*num_decs_range)
        n_utilities = random.randint(*num_utils_range)

        cid = random_cid(
            number_of_nodes=n_all,
            number_of_decisions=n_decisions,
            number_of_utilities=n_utilities,
            add_cpds=add_cpds,
            sufficient_recall=sufficient_recall,
            edge_density=edge_density,
        )

        if cid.sufficient_recall():
            cids.append(cid)

    return cids
