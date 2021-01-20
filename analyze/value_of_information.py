"""Value of Information

Criterion for Information incentive on X:
    (i) X is not a descendent of the decison node, D.
    (ii) U∈Desc(D) (U must be a descendent of D)
    (iii) X is d-connected to U | Fa_D\{X}"""
from typing import List

import networkx as nx

from cid import CID
from macid import MACID


def admits_voi(cid: CID, decision: str, node: str, agent=None) -> bool:
    """Return True if cid admits value of information for node and decision"""

    if agent:
        assert isinstance(cid, MACID)
        agent_utils = cid.utility_nodes[agent]  # this agent's utility nodes
    else:
        agent_utils = cid.utility_nodes  # this agent's utility nodes

    if not agent_utils:  # if the agent has no decision or no utility nodes, no node will have VoI
        return False
    if node not in cid.nodes:
        raise ValueError(f"{node} is not present in the cid")

    # condition (i)
    elif node == decision or node in nx.descendants(cid, decision):
        return False

    for util in agent_utils:
        if util in nx.descendants(cid, decision):  # condition (ii)
            con_nodes = [decision] + cid.get_parents(decision)  # nodes to be conditioned on
            if node in con_nodes:  # remove node from condition nodes
                con_nodes.remove(node)
            if cid.is_active_trail(node, util, con_nodes):  # condition (iv)
                return True
    else:
        return False


def admits_voi_list(cid: CID, decision: str, agent=None) -> List[str]:
    """Return list of nodes with possible value of information for decision"""
    if agent:
        assert isinstance(cid, MACID)
        agent_utils = cid.utility_nodes[agent]  # this agent's utility nodes
    else:
        agent_utils = cid.utility_nodes  # this agent's utility nodes

    if not agent_utils:  # if the agent has no utility nodes, there's no VoI
        return []
    else:
        return [x for x in list(cid.nodes) if admits_voi(cid, decision, x, agent=agent)]


def voi(cid: CID, decision: str, variable: str):
    # TODO test this method
    new = cid.copy()
    new.add_edge(variable, decision)
    new.impute_optimal_policy()
    ev1 = new.expected_utility({})
    new = cid.copy()
    new.remove_edge(variable, decision)
    new.impute_optimal_policy()
    ev2 = new.expected_utility({})
    return ev1 - ev2