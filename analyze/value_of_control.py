from core.cid import CID
import networkx as nx
from analyze.requisite_graph import requisite_graph
from typing import List


def admits_voc(cid: CID, decision: str, node: str) -> bool:
    """
    Return True if a single-decision cid admits positive value of control for node.
    - A single-decision CID G admits positive value of control for a node X ∈ V \ {D}
    if and only if there is a directed path X --> U in the reduced graph G∗.
    """
    req_graph = requisite_graph(cid)
    agent_utilities = cid.all_utility_nodes

    if node not in cid.nodes:
        raise Exception(f"{node} is not present in the cid")
    if decision not in cid.nodes:
        raise Exception(f"{decision} is not present in the cid")

    if node == decision:
        return False

    for util in agent_utilities:
        if node == util or util in nx.descendants(req_graph, node):
            return True

    return False


def admits_voc_list(cid: CID, decision: str) -> List[str]:
    """
    Return list of nodes in single-decision cid with positive value of control.
    """
    return [x for x in list(cid.nodes) if admits_voc(cid, decision, x)]
