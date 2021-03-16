from typing import List

import networkx as nx

from pycid.core.macid_base import MACIDBase


def requisite(cid: MACIDBase, decision: str, node: str) -> bool:
    r"""Check if a CID node is a requisite observation for a decision.

    A node is a requisite observation if it is possibly material.
    A node can be material if:
        i) it is a parent of D.
        ii) X is d-connected to (U ∩ Desc(D)) given Fa_D \ {X}
    "A note about redundancy in influence diagrams" Fagiuoli and Zaffalon, 1998.

    Returns True if the node is requisite.
    """
    if node not in cid.get_parents(decision):
        raise KeyError(f"{node} is not a parent of {decision}")

    agent_utilities = cid.agent_utilities[cid.decision_agent[decision]]
    descended_agent_utilities = set(agent_utilities).intersection(nx.descendants(cid, decision))
    family_d = [decision] + cid.get_parents(decision)
    conditioning_nodes = [i for i in family_d if i != node]
    return any(cid.is_active_trail(node, u_node, conditioning_nodes) for u_node in descended_agent_utilities)


def requisite_list(cid: MACIDBase, decision: str) -> List[str]:
    """Returns list of requisite nodes for decision"""
    return [node for node in cid.get_parents(decision) if requisite(cid, decision, node)]


def requisite_graph(cid: MACIDBase) -> MACIDBase:
    """The requiste graph of the original CID.

    The requisite graph is also called a minimal reduction, d reduction, or the trimmed graph.

    The requisite graph G∗ of a multi-decision CID G is the result of repeatedely
    removing from G all nonrequisite observation links.
    ("Representing and Solving Decision Problems with Limited Information", Lauritzen and Nielsen, 2001)
    """
    requisite_graph = cid.copy()
    decisions = cid.get_valid_order()

    for decision in reversed(decisions):
        non_requisite_nodes = set(cid.get_parents(decision)) - set(requisite_list(requisite_graph, decision))
        for nr in non_requisite_nodes:
            requisite_graph.remove_edge(nr, decision)
    return requisite_graph
