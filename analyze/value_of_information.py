"""Value of Information

Criterion for Information incentive on X:
    (i) X is not a descendant of the decision node, D.
    (ii) U∈Desc(D) (U must be a descendant of D)
    (iii) X is d-connected to U | Fa_D\{X}"""
from typing import List

import networkx as nx

from core.cid import CID
from core.macid_base import MACIDBase


def admits_voi(cid: MACIDBase, decision: str, node: str) -> bool:
    """Return True if cid admits value of information for node and decision"""

    agent_utils = cid.utility_nodes_agent[cid.whose_node[decision]]  # this agent's utility nodes

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


def admits_voi_list(cid: MACIDBase, decision: str) -> List[str]:
    """
    Return list of nodes with possible value of information for decision
    """
    return [x for x in list(cid.nodes) if admits_voi(cid, decision, x)]


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
