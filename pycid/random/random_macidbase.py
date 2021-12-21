from __future__ import annotations

import random
from typing import Callable, Dict, Iterable, Mapping, Set, Tuple, Union

import networkx as nx
import numpy as np

from pycid.core.cpd import Outcome, TabularCPD
from pycid.core.get_paths import find_active_path, get_motif
from pycid.core.macid_base import AgentLabel, MACIDBase, MechanismGraph
from pycid.random.random_cpd import RandomCPD
from pycid.random.random_dag import random_dag

Relationship = Union[TabularCPD, Dict[Outcome, float], Callable[..., Union[Outcome, Dict[Outcome, float]]]]


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
    if len(agent_decisions_num) != len(agent_utilities_num):
        raise ValueError(
            f"The number of agents specified for agent_decisions_num {len(agent_decisions_num)} does not match \
        the number of agents specified for agent_utilities_num {len(agent_utilities_num)}"
        )

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
            agent_decs = random.sample(list(possible_dec_nodes), num_decs)
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
                _add_sufficient_recalls(mb)
                if not _check_max_in_degree(mb, max_in_degree):
                    # adding edges for sufficient recall requirement violates max_in_degree: resample a new random DAG
                    continue

            if add_cpds:
                add_random_cpds(mb)

            return mb
        continue
    else:
        raise ValueError(
            f"Could not create a MACID satisfying all constraints in {max_resampling_attempts} sampling attempts"
        )


def add_random_cpds(mb: MACIDBase) -> None:
    """
    add cpds to the random (MA)CID.
    """
    node_cpds: Dict[str, Relationship] = {}
    for node in mb.nodes:
        if node in mb.decisions:
            node_cpds[node] = [0, 1]
        else:
            node_cpds[node] = RandomCPD()
    mb.add_cpds(**node_cpds)


def _check_max_in_degree(mb: MACIDBase, max_in_degree: int) -> bool:
    """
    check that the degree of each vertex in the DAG is less than the set maximum.
    """
    for node in mb.nodes:
        if mb.in_degree(node) > max_in_degree:
            return False
    else:
        return True


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


def _add_sufficient_recalls(mb: MACIDBase) -> None:
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
