from __future__ import annotations

import random
from typing import Dict, Iterable, List, Mapping, Set, Tuple

import networkx as nx
import numpy as np
from pgmpy.base.DAG import DAG

from pycid.core.cid import CID
from pycid.core.cpd import DecisionDomain
from pycid.core.get_paths import find_active_path, get_motif
from pycid.core.macid import MACID
from pycid.core.macid_base import AgentLabel, MACIDBase, MechanismGraph
from pycid.random.random_cpd import RandomCPD
from pycid.random.random_dag import random_dag


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
        _add_random_cpds(cid)

    return cid


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
        _add_random_cpds(macid)

    return macid


def random_macidbase(
    number_of_nodes: int = 10,
    agent_decisions_num: Tuple[int, ...] = (1, 2),
    agent_utilities_num: Tuple[int, ...] = (2, 1),
    add_cpds: bool = False,
    sufficient_recall: bool = False,
    edge_density: float = 0.4,
    max_in_degree: int = 4,
    max_resampling_attempts: int = 5000,
) -> MACIDBase:
    """
    Generate a random MACIDBase

    Returns
    -------
    A MACIDBase that satisfies the given constraints or a ValueError if it was unable to meet
     the constraints in the specified number of attempts.

    """
    for _ in range(max_resampling_attempts):

        dag = random_dag(number_of_nodes=number_of_nodes, edge_density=edge_density, max_in_degree=max_in_degree)

        # assign utility nodes to each agent based on the barren nodes in the random dag
        barren_nodes = [node for node in dag.nodes if not list(dag.successors(node))]
        if sum(agent_utilities_num) > len(barren_nodes):
            # there are not enough barren_nodes: resample a new random DAG.
            continue
        np.random.shuffle(barren_nodes)  # randomise
        util_node_candidates = iter(barren_nodes)
        agent_utilities_old_name = {
            agent: [next(util_node_candidates) for _ in range(num)] for agent, num in enumerate(agent_utilities_num)
        }

        used_nodes = set()  # type: Set[str]
        agent_decisions: Mapping[AgentLabel, Iterable[str]] = {}
        agent_utilities: Mapping[AgentLabel, Iterable[str]] = {}
        node_name_change_map: Dict[str, str] = {}

        for agent in agent_utilities_old_name.keys():
            # assign decision nodes to agent
            num_decs = agent_decisions_num[agent]
            agent_utils = agent_utilities_old_name[agent]
            possible_dec_nodes: Set[str] = (
                set().union(*[set(dag._get_ancestors_of(node)) for node in agent_utils])
                - set(agent_utils)
                - used_nodes  # type: ignore
            )
            if num_decs > len(possible_dec_nodes):
                break
            agent_decs = random.sample(possible_dec_nodes, num_decs)
            used_nodes.update(agent_decs)

            # rename decision and utility nodes
            agent_util_name_change = {
                old_util_name: "U^" + str(agent) + "_" + str(i) for i, old_util_name in enumerate(agent_utils)
            }
            agent_dec_name_change = {
                old_dec_name: "D^" + str(agent) + "_" + str(i) for i, old_dec_name in enumerate(agent_decs)
            }
            agent_utilities[agent] = list(agent_util_name_change.values())  # type: ignore
            agent_decisions[agent] = list(agent_dec_name_change.values())  # type: ignore
            node_name_change_map.update(**agent_util_name_change, **agent_dec_name_change)

        else:
            # rename chance nodes
            chance_nodes = [node for node in dag.nodes if node not in node_name_change_map.keys()]
            chance_name_change = {old_chance_name: "X_" + str(i) for i, old_chance_name in enumerate(chance_nodes)}
            node_name_change_map.update(chance_name_change)

            dag = nx.relabel_nodes(dag, node_name_change_map)
            mb = MACIDBase(dag.edges, agent_decisions=agent_decisions, agent_utilities=agent_utilities)

            if sufficient_recall:
                add_sufficient_recalls(mb)
                if not _check_max_in_degree(mb, max_in_degree):
                    # adding edges for sufficient recall requirement violates max_in_degree: resample a new random DAG
                    continue

            if add_cpds:
                _add_random_cpds(mb)

            return mb
        continue
    else:
        raise ValueError(
            f"Could not create a MACID satisfying all constraints in {max_resampling_attempts} sampling attempts"
        )


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


def _check_max_in_degree(mb: MACIDBase, max_in_degree: int) -> bool:
    """
    check that the degree of each vertex in the DAG is less than the set maximum.
    """
    for node in mb.nodes:
        if mb.in_degree(node) > max_in_degree:
            return False
    else:
        return True


def _add_random_cpds(mb: MACIDBase) -> None:
    """
    add cpds to the random (MA)CID.
    """
    for node in mb.nodes:
        if node in mb.decisions:
            mb.add_cpds(DecisionDomain(node, [0, 1]))
        else:
            mb.add_cpds(RandomCPD(node))


def _add_sufficient_recall(mb: MACIDBase, d1: str, d2: str, utility_node: str) -> None:
    """Add edges to a (MA)CID until an agent at `d2` has sufficient recall of `d1` to optimise utility_node.

    d1, d2 and utility node all belong to the same agent.

    `d2' has sufficient recall of `d1' if d2 does not strategically rely on d1. This means
    that d1 is not s-reachable from d2.

    edges are added from non-collider nodes along an active path from the mechanism of `d1' to
    somue utilty node descended from d2 until recall is sufficient.
    """

    if d2 in mb._get_ancestors_of(d1):
        raise ValueError("{} is an ancestor of {}".format(d2, d1))

    mg = MechanismGraph(mb)
    while mg.is_active_trail(d1 + "mec", utility_node, observed=mg.get_parents(d2) + [d2]):
        path = find_active_path(mg, d1 + "mec", utility_node, {d2, *mg.get_parents(d2)})
        if path is None:
            raise RuntimeError("couldn't find path even though there should be an active trail")
        while True:
            idx = random.randrange(1, len(path) - 1)
            if get_motif(mg, path, idx) != "collider":
                if d2 not in mg._get_ancestors_of(path[idx]):  # to prevent cycles
                    mb.add_edge(path[idx], d2)
                    mg.add_edge(path[idx], d2)
                    break


def add_sufficient_recalls(mb: MACIDBase) -> None:
    """add edges to a macid until all agents have sufficient recall of all of their previous decisions"""
    agents = mb.agents
    for agent in agents:
        decisions = mb.agent_decisions[agent]
        for utility_node in mb.agent_utilities[agent]:
            for i, dec1 in enumerate(decisions):
                for dec2 in decisions[i + 1 :]:
                    if dec1 in mb._get_ancestors_of(dec2):
                        if utility_node in nx.descendants(mb, dec2):
                            _add_sufficient_recall(mb, dec1, dec2, utility_node)
                    else:
                        if utility_node in nx.descendants(mb, dec1):
                            _add_sufficient_recall(mb, dec2, dec1, utility_node)
