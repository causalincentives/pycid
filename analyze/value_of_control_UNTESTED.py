from core.cid import CID
from core.macid_base import MACIDBase
from core.get_paths import find_all_dir_paths
import networkx as nx
from analyze.value_of_information import admits_voi
from analyze.d_reduction import d_reduction
from typing import List

def admits_voc(cid: MACIDBase, decision: str, node: str) -> bool:
    """
    Return True if a single-decision cid admits positive value of control on node.
    - A single-decision CID G admits positive value of control for a node X ∈ V \ {D} if 
    and only if there is a directed path X --> U in the reduced graph G∗.
    """
    reduced_cid = d_reduction(cid)
    agent_utils = cid.all_utility_nodes

    if node == decision:
        return False

    for util in agent_utils:
        if node == util or util in nx.descendants(reduced_cid, node):  # condition (ii)
            return True

    return False


def admits_voc_list(cid: MACIDBase, decision: str) -> List[str]:
    """
    Return list of nodes in single-decision cid with possible value of control.
    """
    return [x for x in list(cid.nodes) if admits_voc(cid, decision, x)]


def admits_ici(cid: MACIDBase, decision: str, node: str) -> bool:
    """
    Return True if a single-decision cid admits an instrumental control incentive on node.
    - A single-decision CID G admits an instrumental control incentive on X ∈ V 
        if and only if G has a directed path from the decision D to a utility node U ∈ U that passes through X,
        i.e. a directed path D --> X --> U.
    """
    agent_utils = cid.all_utility_nodes
    d_u_paths = [path for util in agent_utils for path in find_all_dir_paths(cid, decision, util)]

    if any(node in path for path in d_u_paths):
        return True

    return False


def admits_ici_list(cid: MACIDBase, decision: str) -> List[str]:
    """
    Return list of nodes in single-decision cid that admit an instrumental control incentive.
    """
    return [x for x in list(cid.nodes) if admits_ici(cid, decision, x)]



# def has_indir_control_inc(self, node, agent):
#     """
#     returns True if a node faces an indirect control incentive
#     """
#     agent_dec = self.all_decision_nodes[
#         agent]  # decision made by this agent (this incentive is currently only proven to hold for the single decision case)
#     agent_utils = self.all_utility_nodes[agent]  # this agent's utility nodes
#     trimmed_MACID = self.dreduction(agent)

#     for util in agent_utils:
#         if trimmed_MACID.has_control_inc(node, agent):

#             Fa_d = trimmed_MACID.get_parents(*agent_dec) + agent_dec
#             con_nodes = [i for i in Fa_d if i != node]
#             backdoor_exists = trimmed_MACID.backdoor_path_active_when_conditioning_on_W(node, util, con_nodes)
#             x_u_paths = trimmed_MACID.find_all_dir_path(node, util)
#             if any(agent_dec[0] in paths for paths in
#                    x_u_paths) and backdoor_exists:  # agent_dec[0] as it should only have one entry because we've currently restricted it to the single dec case
#                 return True

#     return False


# def has_dir_control_inc(self, node, agent):
#     """
#     returns True if a node faces a direct control incentive
#     """
#     agent_dec = self.all_decision_nodes[
#         agent]  # decision made by this agent (this incentive is currently only proven to hold for the single decision case)
#     agent_utils = self.all_utility_nodes[agent]  # this agent's utility nodes
#     trimmed_MACID = self.dreduction(agent)

#     for util in agent_utils:
#         if trimmed_MACID.has_control_inc(node, agent):
#             x_u_paths = trimmed_MACID.find_all_dir_path(node, util)
#             for path in x_u_paths:
#                 if set(agent_dec).isdisjoint(set(path)):
#                     return True
#     return False
