from core.cid import CID
from core.get_paths import find_all_dir_paths
from typing import List


def admits_ici(cid: CID, decision: str, node: str) -> bool:
    """
    Return True if a single-decision cid admits an instrumental control incentive on node.
    - A single-decision CID G admits an instrumental control incentive on X ∈ V
        if and only if G has a directed path from the decision D to a utility node U ∈ U that passes through X,
        i.e. a directed path D --> X --> U.
    """
    if len(cid.agents) > 1:
        raise Exception(f"This CID has {len(cid.agents)} agents. This incentive is currently only \
                        valid for CIDs with one agent.")

    if node not in cid.nodes:
        raise Exception(f"{node} is not present in the cid")
    if decision not in cid.nodes:
        raise Exception(f"{decision} is not present in the cid")

    agent_utilities = cid.all_utility_nodes
    d_u_paths = [path for util in agent_utilities for path in find_all_dir_paths(cid, decision, util)]
    if any(node in path for path in d_u_paths):
        return True

    return False


def admits_ici_list(cid: CID, decision: str) -> List[str]:
    """
    Return list of nodes in single-decision cid that admit an instrumental control incentive.
    """
    return [x for x in list(cid.nodes) if admits_ici(cid, decision, x)]
