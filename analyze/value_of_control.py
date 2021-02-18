from core.cid import CID
import networkx as nx
from analyze.requisite_graph import requisite_graph
from typing import List
from core.get_paths import is_active_backdoor_trail, find_all_dir_paths


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


def admits_indir_voc(cid: CID, decision: str, node: str) -> bool:
    """
    Return True if a single-decision cid admits indirect positive value of control for node.
    - A single-decision CID G admits positive value of control for a node X ∈ V \ {D} if and
    only if there is a directed path X --> U in the reduced graph G∗.
    - The path X --> U may or may not pass through D.
    - The agent has a direct value of control incentive on D if the path does not pass through D.
    - The agent has an indirect value of control incentive on D if the path does pass through D
    and there is also a backdoor path X--U that begins backwards from X (...<- X) and is
    active when conditioning on Fa_D \ {X}
    """
    if node not in cid.nodes:
        raise Exception(f"{node} is not present in the cid")
    if decision not in cid.nodes:
        raise Exception(f"{decision} is not present in the cid")

    agent_utilities = cid.all_utility_nodes
    req_graph = requisite_graph(cid)
    d_family = [decision] + cid.get_parents(decision)
    con_nodes = [i for i in d_family if i != node]
    if not admits_voc(cid, decision, node):
        return False

    for util in agent_utilities:
        if node == util or util in nx.descendants(req_graph, node):
            backdoor_exists = is_active_backdoor_trail(req_graph, node, util, con_nodes)
            x_u_paths = find_all_dir_paths(req_graph, node, util)
            if any(decision in paths for paths in x_u_paths) and backdoor_exists:
                return True

    return False


def admits_indir_voc_list(cid: CID, decision: str) -> List[str]:
    """
    Return list of nodes in single-decision cid with indirect positive value of control.
    """
    return [x for x in list(cid.nodes) if admits_indir_voc(cid, decision, x)]


def admits_dir_voc(cid: CID, decision: str, node: str) -> bool:
    """
    Return True if a single-decision cid admits direct positive value of control for node.
    - A single-decision CID G admits positive value of control for a node X ∈ V \ {D} if and
    only if there is a directed path X --> U in the reduced graph G∗.
    - The path X --> U may or may not pass through D.
    - The agent has a direct value of control incentive on D if the path does not pass through D.
    - The agent has an indirect value of control incentive on D if the path does pass through D
    and there is also a backdoor path X--U that begins backwards from X (...<- X) and is
    active when conditioning on Fa_D \ {X}
    """
    if node not in cid.nodes:
        raise Exception(f"{node} is not present in the cid")
    if decision not in cid.nodes:
        raise Exception(f"{decision} is not present in the cid")

    agent_utilities = cid.all_utility_nodes
    req_graph = requisite_graph(cid)

    if not admits_voc(cid, decision, node):
        return False

    for util in agent_utilities:
        if node == util or util in nx.descendants(req_graph, node):
            x_u_paths = find_all_dir_paths(req_graph, node, util)
            if any([decision not in path for path in x_u_paths]):
                return True

    return False


def admits_dir_voc_list(cid: CID, decision: str) -> List[str]:
    """
    Return list of nodes in single-decision cid with indirect positive value of control.
    """
    return [x for x in list(cid.nodes) if admits_dir_voc(cid, decision, x)]
