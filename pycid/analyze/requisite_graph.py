from typing import List

import networkx as nx

from pycid.core.macid_base import MACIDBase


def requisite(cid: MACIDBase, decision: str, node: str) -> bool:
    """Return True if cid node is requisite observation for decision (i.e. possibly material).
    - A node can be material if:
    i) it is a parent of D.
    ii) X is d-connected to (U ∩ Desc(D)) given Fa_D \ {X}
    "A note about redundancy in influence diagrams" Fagiuoli and Zaffalon, 1998.
    """
    if decision not in cid.nodes:
        raise Exception(f"{decision} is not present in the cid")
    if node not in cid.get_parents(decision):
        raise Exception(f"{node} is not a parent of {decision}")

    agent_utilities = cid.utility_nodes_agent[cid.whose_node[decision]]
    descended_agent_utilities = set(agent_utilities).intersection(nx.descendants(cid, decision))
    family_d = [decision] + cid.get_parents(decision)
    conditioning_nodes = [i for i in family_d if i != node]
    return any([cid.is_active_trail(node, u_node, conditioning_nodes)
                for u_node in descended_agent_utilities])


def requisite_list(cid: MACIDBase, decision: str) -> List[str]:
    """Returns list of requisite nodes for decision"""
    return [node for node in cid.get_parents(decision) if requisite(cid, decision, node)]


def requisite_graph(cid: MACIDBase) -> MACIDBase:
    """Return the requisite graph of the original CID, also called
    a minimal_reduction, d_reduction, or the trimmed graph.
    - The requisite graph G∗ of a multi-decision CID G is the result of repeatedely
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
