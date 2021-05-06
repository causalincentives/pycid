from __future__ import annotations

import random
from typing import List, Tuple

from pycid.core.macid import MACID
from pycid.random.random_macidbase import add_random_cpds, random_macidbase


def random_macid(
    number_of_nodes: int = 10,
    agent_decisions_num: Tuple[int, ...] = (1, 2),
    agent_utilities_num: Tuple[int, ...] = (2, 1),
    add_cpds: bool = False,
    sufficient_recall: bool = False,
    edge_density: float = 0.4,
    max_in_degree: int = 4,
    max_resampling_attempts: int = 5000,
) -> MACID:
    """
    Generate a random MACID.

    Parameters:
    -----------
    number_of nodes: The total number of nodes in the MACID.

    number_of_agents: The number of agents in the MACID.

    agent_decisions_num: A Tuple giving the number of decision nodes for each agent. For example, (1,2,1)
    means that the first agent has 1 decision, the second agent has two, and the third agent has 1.

    agent_utilities_num: A Tuple giving the number of utilities for each agent. For example, (1,2,1)
    means that the first agent has 1 utility node, the second agent has two, and the third agent has 1.

    add_cpds: True if we should pararemeterise the MACID as a model.
    This adds [0,1] domains to every decision node and RandomCPDs to every utility and chance node in the MACID.

    sufficient_recall: True if all of the agents should have sufficient recall of all of their previous
    decisions. Agent i has sufficient recall in a MACID if the relevance graph restricted to
    just agent i's decision nodes is acyclic.

    edge_density: The density of edges in the MACID's DAG as a proportion of the maximum possible number
     of nodes in the DAG - n*(n-1)/2

    max_in_degree: The maximal number of edges incident to a node in the MACID's DAG.

    max_resampling_attempts: The maxmimum number of resampling of random DAGs attempts in order to try to
     satisfy all constraints.

    Returns
    -------
    A MACID that satisfies the given constraints or a ValueError if it was unable to meet the constraints in the
    specified number of attempts.

    """

    mb = random_macidbase(
        number_of_nodes=number_of_nodes,
        agent_decisions_num=agent_decisions_num,
        agent_utilities_num=agent_utilities_num,
        add_cpds=False,
        sufficient_recall=sufficient_recall,
        edge_density=edge_density,
        max_in_degree=max_in_degree,
        max_resampling_attempts=max_resampling_attempts,
    )
    macid = MACID(mb.edges, agent_decisions=mb.agent_decisions, agent_utilities=mb.agent_utilities)

    if add_cpds:
        add_random_cpds(macid)

    return macid


def random_macids(
    total_nodes_range: Tuple[int, int] = (13, 17),
    num_decs_range: Tuple[int, int] = (1, 3),
    num_utils_range: Tuple[int, int] = (1, 3),
    add_cpds: bool = True,
    sufficient_recall: bool = False,
    edge_density: float = 0.4,
    n_macids: int = 10,
) -> List[MACID]:
    """Generates a number of CIDs with sufficient recall"""
    macids: List[MACID] = []

    while len(macids) < n_macids:
        n_all = random.randint(*total_nodes_range)
        num_agents = random.randint(1, 3)
        agent_n_decisions = tuple(random.choices(range(*num_decs_range), k=num_agents))
        agent_n_utilities = tuple(random.choices(range(*num_utils_range), k=num_agents))

        macid = random_macid(
            number_of_nodes=n_all,
            agent_decisions_num=agent_n_decisions,
            agent_utilities_num=agent_n_utilities,
            add_cpds=add_cpds,
            sufficient_recall=sufficient_recall,
            edge_density=edge_density,
        )

        if macid.sufficient_recall():
            macids.append(macid)

    return macids
