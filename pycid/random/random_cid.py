from __future__ import annotations

import random
from typing import Dict, Iterable, List, Mapping, Set, Tuple

import networkx as nx
import numpy as np

from pycid.core.cid import CID
from pycid.core.cpd import DecisionDomain
from pycid.core.get_paths import find_active_path, get_motif
from pycid.core.macid_base import AgentLabel, MACIDBase, MechanismGraph
from pycid.random.random_cpd import RandomCPD
from pycid.random.random_dag import random_dag


def random_cid(
    number_of_nodes: int = 8,
    number_of_decisions: int = 1,
    number_of_utilities: int = 1,
    edge_density: float = 0.4,
    max_in_degree: int = 4,
    max_resampling_attempts: int = 100,
    add_cpds: bool = True,
    sufficient_recall: bool = False,
) -> CID:
    """
    Generate a random CID with the specified number of nodes, decisions utilities.
    """

    for _ in range(max_resampling_attempts):

        dag = random_dag(number_of_nodes=number_of_nodes, edge_density=edge_density, max_in_degree=max_in_degree)

        barren_nodes = [node for node in dag.nodes if not list(dag.successors(node))]
        if number_of_utilities > len(barren_nodes):
            # there are not enough barren_nodes: resample a new random DAG.
            continue
        util_nodes = random.sample(barren_nodes, number_of_utilities)

        # a decision node must be an ancestor of a utility node
        ancestors = set()  # type: Set[str]
        possible_dec_nodes = ancestors.union(*[set(dag._get_ancestors_of(node)) for node in util_nodes]) - set(
            util_nodes
        )

        if number_of_decisions > len(possible_dec_nodes):
            # there are not enough possible decision nodes: resample a new random DAG
            continue
        dec_nodes = random.sample(possible_dec_nodes, number_of_decisions)

        dec_name_change = {old_dec_name: "D" + str(i) for i, old_dec_name in enumerate(dec_nodes)}
        decision_nodes = list(dec_name_change.values())
        util_name_change = {old_util_name: "U" + str(i) for i, old_util_name in enumerate(util_nodes)}
        utility_nodes = list(util_name_change.values())
        node_name_change_map = {**dec_name_change, **util_name_change}
        dag = nx.relabel_nodes(dag, node_name_change_map)

        cid = CID(dag.edges, decisions=decision_nodes, utilities=utility_nodes)

        if sufficient_recall:
            add_sufficient_recalls(cid)
            if not _check_max_in_degree(cid, max_in_degree):
                # adding edges to meet sufficient recall requirement violates max_in_degree: resample a new random DAG
                continue
        if add_cpds:
            _add_random_cpds(cid)

        return cid  # random CID has satisfied all of the requirements.
    else:
        raise ValueError(
            f"Could not create a CID that satisfied all constraints in {max_resampling_attempts} sampling attempts"
        )


def random_macidbase(
    number_of_nodes: int = 8,
    number_of_agents: int = 2,
    max_decision_nodes_for_agent: int = 1,
    max_utility_nodes_for_agent: int = 1,
    max_resampling_attempts: int = 100,
    edge_density: float = 0.4,
    max_in_degree: int = 4,
    add_cpds: bool = True,
    sufficient_recall: bool = False,
) -> MACIDBase:
    """
    Generate a random MACIDBAse with the specified number of nodes, number of agents, and a maximum number of decision
    and utility nodes for each agent.
    """

    for _ in range(max_resampling_attempts):

        dag = random_dag(number_of_nodes=number_of_nodes, edge_density=edge_density, max_in_degree=max_in_degree)

        barren_nodes = [node for node in dag.nodes if not list(dag.successors(node))]
        if max_utility_nodes_for_agent * number_of_agents > len(barren_nodes):
            # there are not enough barren_nodes: resample a new random DAG.
            continue

        agent_utilities, util_nodes_name_change = _create_random_utility_nodes(
            number_of_agents, max_decision_nodes_for_agent, barren_nodes
        )
        dag = nx.relabel_nodes(dag, util_nodes_name_change)

        agent_decisions: Mapping[AgentLabel, Iterable[str]] = {}
        all_dec_name_change: Dict[str, str] = {}
        used_nodes = set()  # type: Set[str]
        for agent in range(number_of_agents):
            agent_utils = agent_utilities[agent]

            ancestors = set()  # type: Set[str]
            possible_dec_nodes = (
                ancestors.union(*[set(dag._get_ancestors_of(node)) for node in agent_utils])
                - set(agent_utils)
                - used_nodes
            )
            if not possible_dec_nodes:
                # there are no possible decision nodes for this agent: resample a new random DAG.
                break
            sample_dec_nodes = random.sample(
                possible_dec_nodes, min(len(possible_dec_nodes), max_decision_nodes_for_agent)
            )
            used_nodes.update(sample_dec_nodes)

            agent_dec_name_change = {
                old_dec_name: "D^" + str(agent) + "_" + str(i) for i, old_dec_name in enumerate(sample_dec_nodes)
            }
            agent_decisions[agent] = list(agent_dec_name_change.values())  # type: ignore
            all_dec_name_change.update(agent_dec_name_change)

        else:
            dag = nx.relabel_nodes(dag, all_dec_name_change)
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
    n_all_range: Tuple[int, int] = (10, 15),
    nd_range: Tuple[int, int] = (2, 4),
    nu_range: Tuple[int, int] = (2, 4),
    edge_density: float = 0.4,
    n_cids: int = 10,
    sufficient_recall: bool = True,
) -> List[CID]:
    """generates a bunch of CIDs with sufficient recall
    if add_sr_edges=True, then sufficient recall is ensured by adding edges
    otherwise it is ensured by resampling graphs"""
    cids: List[CID] = []

    while len(cids) < n_cids:
        n_all = random.randint(*n_all_range)
        n_decisions = random.randint(*nd_range)
        n_utilities = random.randint(*nu_range)

        cid = random_cid(n_all, n_decisions, n_utilities, edge_density, sufficient_recall=sufficient_recall)

        if cid.sufficient_recall():
            cids.append(cid)

    return cids


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


def _create_random_utility_nodes(
    number_of_agents: int, max_utility_nodes_for_agent: int, barren_nodes: List[str]
) -> Tuple[Mapping[AgentLabel, Iterable[str]], Dict[str, str]]:
    """
    Create random utility nodes for each agent based on the barren nodes available, a fixed number of agents,
    and a maximum number of utility nodes for each agent.
    """
    sample_util_nodes = random.sample(barren_nodes, max_utility_nodes_for_agent * number_of_agents)
    sample_util_nodes_partition = np.array_split(sample_util_nodes, number_of_agents)

    agent_utilities: Mapping[AgentLabel, Iterable[str]] = {}
    all_util_name_change = {}
    for agent, nodes_to_be_utils in enumerate(sample_util_nodes_partition):
        agent_util_name_change = {
            old_util_name: "U^" + str(agent) + "_" + str(i) for i, old_util_name in enumerate(list(nodes_to_be_utils))
        }
        agent_utilities[agent] = list(agent_util_name_change.values())  # type: ignore
        all_util_name_change.update(agent_util_name_change)

    return agent_utilities, all_util_name_change


def _add_sufficient_recall(cid: CID, d1: str, d2: str, utility_node: str) -> None:
    """Add edges to a cid until `d2` has sufficient recall of `d1` (to optimize utility)

    `d2' has sufficient recall of `d1' if d2 does not strategically rely on d1. This means
    that d1 is not s-reachable from d2.

    edges are added from non-collider nodes along an active path from the mechanism of `d1' to
    somue utilty node descended from d2 until recall is sufficient.
    """

    if d2 in cid._get_ancestors_of(d1):
        raise ValueError("{} is an ancestor of {}".format(d2, d1))

    mg = MechanismGraph(cid)
    while mg.is_active_trail(d1 + "mec", utility_node, observed=mg.get_parents(d2) + [d2]):
        path = find_active_path(mg, d1 + "mec", utility_node, {d2, *mg.get_parents(d2)})
        if path is None:
            raise RuntimeError("couldn't find path even though there should be an active trail")
        while True:
            idx = random.randrange(1, len(path) - 1)
            if get_motif(mg, path, idx) != "collider":
                if d2 not in mg._get_ancestors_of(path[idx]):  # to prevent cycles
                    cid.add_edge(path[idx], d2)
                    mg.add_edge(path[idx], d2)
                    break


def add_sufficient_recalls(mb: MACIDBase) -> None:
    """add edges to a macid until all agents have sufficient recall of all previous decisions"""
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
