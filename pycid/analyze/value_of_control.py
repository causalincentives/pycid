from typing import List

import networkx as nx

from pycid.analyze.requisite_graph import requisite_graph
from pycid.core.cid import CID
from pycid.core.get_paths import find_all_dir_paths, is_active_backdoor_trail


def admits_voc(cid: CID, node: str) -> bool:
    """Check if a CID admits positive value of control for a node.

    A CID G admits positive value of control for a node X ∈ V
    if and only if X is not a decision node and there is a directed path X --> U
    in the reduced graph G∗.
    """
    if len(cid.agents) > 1:
        raise ValueError(
            f"This CID has {len(cid.agents)} agents. This incentive is currently only valid for CIDs with one agent."
        )

    if node not in cid.nodes:
        raise KeyError(f"{node} is not present in the cid")
    if not cid.sufficient_recall():
        raise ValueError("VoC only implemented graphs with sufficient recall")
    if node in cid.decisions:
        return False

    req_graph = requisite_graph(cid)
    agent_utilities = cid.utilities

    for util in agent_utilities:
        if node == util or util in nx.descendants(req_graph, node):
            return True

    return False


def admits_voc_list(cid: CID) -> List[str]:
    """
    Return list of nodes in cid with positive value of control.
    """
    return [x for x in list(cid.nodes) if admits_voc(cid, x)]


def quantitative_voc(cid: CID, node: str) -> float:
    r"""
    Returns the quantitative value of control (voc) of a variable corresponding to a node in a parameterised CID.

    A node X ∈ V \ {D} in a single-decision CID has quantitative voi equal to
    max_EU_(π, g^x)[M_g^x] - max_EU_(π)[M]
    ie the maximum utility attainable under any policy π and any soft intervention g^x in M_g^x minus the maximum
    utility attainable under any policy π in M where:
    - M is the original CID
    - M_g^x is the the original CID modified with a new soft intervention g^x
    (ie a new function g^x: dom(Pa^X) -> dom(X)) on variable X.

    ("Agent Incentives: a Causal Perspective" by Everitt, Carey, Langlois, Ortega, and Legg, 2020)
    """
    if node not in cid.nodes:
        raise KeyError(f"{node} is not present in the cid")

    # optimal policy in the original CID.
    cid.impute_optimal_policy()
    ev1: float = cid.expected_utility({})
    cid.make_decision(node)
    # optimal policy in the modified CID where the agent can now decide the CPD for node.
    cid.impute_optimal_policy()
    ev2: float = cid.expected_utility({})
    return ev2 - ev1


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
        raise ValueError(
            f"This CID has {len(cid.agents)} agents. This incentive is currently only valid for CIDs with one agent."
        )

    if node not in cid.nodes:
        raise KeyError(f"{node} is not present in the cid")
    if decision not in cid.nodes:
        raise KeyError(f"{decision} is not present in the cid")
    if not cid.sufficient_recall():
        raise ValueError("VoC only implemented graphs with sufficient recall")

    agent_utilities = cid.utilities
    req_graph = requisite_graph(cid)
    d_family = [decision] + cid.get_parents(decision)
    con_nodes = {i for i in d_family if i != node}
    if not admits_voc(cid, node):
        return False

    req_graph_node_descendants = set(nx.descendants(req_graph, node))
    for util in agent_utilities:
        if util != node and util not in req_graph_node_descendants:
            continue
        if not is_active_backdoor_trail(req_graph, node, util, con_nodes):
            continue
        if any(decision in path for path in find_all_dir_paths(req_graph, node, util)):
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
        raise ValueError(
            f"This CID has {len(cid.agents)} agents. This incentive is currently only valid for CIDs with one agent."
        )

    if node not in cid.nodes:
        raise KeyError(f"{node} is not present in the cid")

    agent_utilities = cid.utilities
    req_graph = requisite_graph(cid)

    if not admits_voc(cid, node):
        return False

    req_graph_node_descendants = set(nx.descendants(req_graph, node))
    for util in agent_utilities:
        if util != node and util not in req_graph_node_descendants:
            continue
        for path in find_all_dir_paths(req_graph, node, util):
            if not set(path).intersection(cid.decisions):
                return True

    return False


def admits_dir_voc_list(cid: CID) -> List[str]:
    """
    Return list of nodes in cid with direct positive value of control.
    """
    return [x for x in list(cid.nodes) if admits_dir_voc(cid, x)]
