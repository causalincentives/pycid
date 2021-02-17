from typing import List
import networkx as nx
from core.cid import CID


def admits_voi(cid: CID, decision: str, node: str) -> bool:
    """Return True if cid admits value of information for node.
    - A CID admits value of information for a node X if:
    i) X is not a descendant of the decision node, D.
    ii) X is d-connected to U given Fa_D \ {X}, where U ∈ U ∩ Desc(D)
    ("Agent Incentives: a Causal Perspective" by Everitt, Carey, Langlois, Ortega, and Legg, 2020)
    """
    agent_utilities = cid.all_utility_nodes

    if node not in cid.nodes:
        raise Exception(f"{node} is not present in the cid")
    if decision not in cid.nodes:
        raise Exception(f"{decision} is not present in the cid")

    # condition (i)
    elif node == decision or node in nx.descendants(cid, decision):
        return False
    # condition (ii)
    descended_agent_utilities = [util for util in agent_utilities if util in nx.descendants(cid, decision)]
    d_family = [decision] + cid.get_parents(decision)
    con_nodes = [i for i in d_family if i != node]
    voi = any([cid.is_active_trail(node, u_node, con_nodes) for u_node in descended_agent_utilities])
    return voi


def admits_voi_list(cid: CID, decision: str) -> List[str]:
    """
    Return the list of nodes with possible value of information for decision.
    """
    return [x for x in list(cid.nodes) if admits_voi(cid, decision, x)]


def voi(cid: CID, decision: str, variable: str) -> float:
    # TODO test this method
    new = cid.copy()
    new.add_edge(variable, decision)
    new.impute_optimal_policy()
    ev1: float = new.expected_utility({})
    new = cid.copy()
    new.remove_edge(variable, decision)
    new.impute_optimal_policy()
    ev2: float = new.expected_utility({})
    return ev1 - ev2
