from typing import List

import networkx as nx

from pycid.analyze.requisite_graph import requisite_graph
from pycid.core.cid import CID


def admits_voi(cid: CID, decision: str, node: str) -> bool:
    r"""Return True if cid admits value of information for node.

    - A CID admits value of information for a node X if:
    i) X is not a descendant of the decision node, D.
    ii) X is d-connected to U given Fa_D \ {X}, where U ∈ U ∩ Desc(D)
    ("Agent Incentives: a Causal Perspective" by Everitt, Carey, Langlois, Ortega, and Legg, 2020)
    """
    if len(cid.agents) > 1:
        raise ValueError(
            f"This CID has {len(cid.agents)} agents. This incentive is currently only valid for CIDs with one agent."
        )

    if node not in cid.nodes:
        raise KeyError(f"{node} is not present in the cid")
    if decision not in cid.nodes:
        raise KeyError(f"{decision} is not present in the cid")
    if not cid.sufficient_recall():
        raise ValueError("Voi only implemented graphs with sufficient recall")
    if node in nx.descendants(cid, decision) or node == decision:
        return False

    cid2 = cid.copy_without_cpds()
    cid2.add_edge(node, decision)
    req_graph = requisite_graph(cid2)
    return node in req_graph.get_parents(decision)


def admits_voi_list(cid: CID, decision: str) -> List[str]:
    """
    Return the list of nodes with possible value of information for decision.
    """
    non_descendants = set(cid.nodes) - set(nx.descendants(cid, decision)) - {decision}
    return [x for x in non_descendants if admits_voi(cid, decision, x)]


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
