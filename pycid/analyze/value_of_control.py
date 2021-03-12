from typing import List

import networkx as nx

from pycid.analyze.requisite_graph import requisite_graph
from pycid.core.cid import CID
from pycid.core.get_paths import find_all_dir_paths, is_active_backdoor_trail


def admits_voc(cid: CID, node: str) -> bool:
    """Check if a CID admits postiive value of control for a node.

    A CID G admits positive value of control for a node X ∈ V
    if and only if X is not a decision node and there is a directed path X --> U
    in the reduced graph G∗.
    """
    if len(cid.agents) > 1:
        raise Exception(
            f"This CID has {len(cid.agents)} agents. This incentive is currently only \
                        valid for CIDs with one agent."
        )

    if node not in cid.nodes:
        raise Exception(f"{node} is not present in the cid")
    if not cid.sufficient_recall():
        raise Exception("VoC only implemented graphs with sufficient recall")
    if node in cid.all_decision_nodes:
        return False

    req_graph = requisite_graph(cid)
    agent_utilities = cid.all_utility_nodes

    for util in agent_utilities:
        if node == util or util in nx.descendants(req_graph, node):
            return True

    return False


def admits_voc_list(cid: CID) -> List[str]:
    """
    Return list of nodes in cid with positive value of control.
    """
    return [x for x in list(cid.nodes) if admits_voc(cid, x)]


def admits_indir_voc(cid: CID, decision: str, node: str) -> bool:
    r"""Check if a single-decision CID admits indirect positive value of control for a node.

    - A single-decision CID G admits positive value of control for a node X ∈ V \ {D} if and
    only if there is a directed path X --> U in the reduced graph G∗.
    - The path X --> U may or may not pass through D.
    - The agent has a direct value of control incentive on D if the path does not pass through D.
    - The agent has an indirect value of control incentive on D if the path does pass through D
    and there is also a backdoor path X--U that begins backwards from X (...<- X) and is
    active when conditioning on Fa_D \ {X}
    """
    if len(cid.agents) > 1:
        raise Exception(
            f"This CID has {len(cid.agents)} agents. This incentive is currently only \
                        valid for CIDs with one agent."
        )

    if node not in cid.nodes:
        raise Exception(f"{node} is not present in the cid")
    if decision not in cid.nodes:
        raise Exception(f"{decision} is not present in the cid")
    if not cid.sufficient_recall():
        raise Exception("VoC only implemented graphs with sufficient recall")

    agent_utilities = cid.all_utility_nodes
    req_graph = requisite_graph(cid)
    d_family = [decision] + cid.get_parents(decision)
    con_nodes = [i for i in d_family if i != node]
    if not admits_voc(cid, node):
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
    Return list of nodes in cid with indirect positive value of control.
    """
    return [x for x in list(cid.nodes) if admits_indir_voc(cid, decision, x)]


def admits_dir_voc(cid: CID, node: str) -> bool:
    r"""Check if a CID admits direct positive value of control for a node.

    - A CID G admits positive value of control for a node X ∈ V \ {D} if and
    only if there is a directed path X --> U in the requisite graph G∗.
    - The path X --> U may or may not pass through decisions.
    - The agent has a direct value of control incentive on D if the path does not pass through a decision.
    - The agent has an indirect value of control incentive on D if the path does pass through any decision
    and there is also a backdoor path X--U that begins backwards from X (...<- X) and is
    active when conditioning on Fa_D \ {X}
    """
    if len(cid.agents) > 1:
        raise Exception(
            f"This CID has {len(cid.agents)} agents. This incentive is currently only \
                        valid for CIDs with one agent."
        )

    if node not in cid.nodes:
        raise Exception(f"{node} is not present in the cid")

    agent_utilities = cid.all_utility_nodes
    req_graph = requisite_graph(cid)

    if not admits_voc(cid, node):
        return False

    for util in agent_utilities:
        if node == util or util in nx.descendants(req_graph, node):
            x_u_paths = find_all_dir_paths(req_graph, node, util)
            for path in x_u_paths:
                if not set(path).intersection(cid.all_decision_nodes):
                    return True

    return False


def admits_dir_voc_list(cid: CID) -> List[str]:
    """
    Return list of nodes in cid with direct positive value of control.
    """
    return [x for x in list(cid.nodes) if admits_dir_voc(cid, x)]
