from core.cid import CID
from core.macid_base import MACIDBase
from core.get_paths import find_all_dir_paths
import networkx as nx
from analyze.d_reduction import d_reduction
from typing import List

def admits_ri(cid: MACIDBase, decision: str, node: str) -> bool:
    """
    Return True if a single-decision cid admits a response incentive on node.
     - A single decision CID G admits a response incentive on X ∈ V \ {D} if
    and only if the reduced graph G* min has a directed path X --> D.
    """
    if node == decision:
        return False
        
    reduced_cid = d_reduction(cid)
    if find_all_dir_paths(reduced_cid, node, decision): 
        return True

    return False


def admits_ri_list(cid: MACIDBase, decision: str) -> List[str]:
    """
    Return list of nodes in single-decision cid that admit a response incentive.
    """
    return [x for x in list(cid.nodes) if admits_ri(cid, decision, x)]
