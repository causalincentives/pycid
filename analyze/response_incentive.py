from core.cid import CID
from core.get_paths import find_all_dir_paths
from analyze.requisite_graph import requisite_graph
from typing import List


def admits_ri(cid: CID, decision: str, node: str) -> bool:
    """
    Return True if a single-decision cid admits a response incentive on node.
     - A single decision CID G admits a response incentive on X ∈ V \ {D} if
    and only if the reduced graph G* min has a directed path X --> D.
    ("Agent Incentives: a Causal Perspective" by Everitt, Carey, Langlois, Ortega, and Legg, 2020)
    """
    if node not in cid.nodes:
        raise Exception(f"{node} is not present in the cid")
    if decision not in cid.nodes:
        raise Exception(f"{decision} is not present in the cid")
    if not cid.sufficient_recall():
        raise Exception("Voi only implemented graphs with sufficient recall")
    if node == decision:
        return False

    req_graph = requisite_graph(cid)
    if find_all_dir_paths(req_graph, node, decision):
        return True

    return False


def admits_ri_list(cid: CID, decision: str) -> List[str]:
    """
    Return the list of nodes in single-decision cid that admit a response incentive.
    """
    return [x for x in list(cid.nodes) if admits_ri(cid, decision, x)]
