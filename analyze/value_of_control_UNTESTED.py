from core.cid import CID
from core.macid_base import MACIDBase
import networkx as nx
from analyze.value_of_information import admits_voi



def dreduction(cid: CID, agent):
        """
        returns the DAG which has been trimmed of all irrelevant information links.
        """
        #assert (len(cid.all_decision_nodes) == 1)"The theory currently only works for the single-decision case!"  
        agent_dec = cid.all_decision_nodes[agent] #decision made by this agent (this incentive is currently only proven to hold for the single decision case)
        #agent_utils = cid.all)utility_nodes[agent] #this agent's utility nodes

        trimmed_graph = cid.copy()
        d_par = cid.get_parents(*agent_dec)
        nonrequisite_nodes = [n for n in d_par if not admits_voi(cid, 'P', n)]

        for node in nonrequisite_nodes:
            trimmed_graph.remove_edge(node, *agent_dec)
        return trimmed_graph


def has_control_inc(cid: CID, node: str, agent) -> bool:
    """
    A single-decision CID G admits positive value of control for a node X ∈ V \ {D} ifand only ifthere is a directed path X  U in the reduced graph G∗.
    returns True if a node faces a control incentive or "positive value of control"
    """
    agent_dec = cid.all_decision_nodes[agent]
    # decision made by this agent (this incentive is currently only proven to hold for the single decision case)
    agent_utils = cid.all_utility_nodes[agent]  # this agent's utility nodes

    if not agent_dec or not agent_utils:  # if the agent has no decision or no utility nodes, no node will face a control incentive
        return False

    if len(agent_dec) > 1:
        return "This incentive currently only applies to the single decision case"

    if [node] == agent_dec:  # condition (i)
        return False

    trimmed_MACID = dreduction(cid, agent)

    for util in agent_utils:
        if node == util or util in nx.descendants(trimmed_MACID, node):  # condition (ii)
            return True

    return False


def all_control_inc_nodes(cid: CID, agent):

        return [x for x in list(cid.nodes) if has_control_inc(cid, x, agent)]


def has_indir_control_inc(self, node, agent):
    """
    returns True if a node faces an indirect control incentive
    """
    agent_dec = self.all_decision_nodes[
        agent]  # decision made by this agent (this incentive is currently only proven to hold for the single decision case)
    agent_utils = self.all_utility_nodes[agent]  # this agent's utility nodes
    trimmed_MACID = self.dreduction(agent)

    for util in agent_utils:
        if trimmed_MACID.has_control_inc(node, agent):

            Fa_d = trimmed_MACID.get_parents(*agent_dec) + agent_dec
            con_nodes = [i for i in Fa_d if i != node]
            backdoor_exists = trimmed_MACID.backdoor_path_active_when_conditioning_on_W(node, util, con_nodes)
            x_u_paths = trimmed_MACID.find_all_dir_path(node, util)
            if any(agent_dec[0] in paths for paths in
                   x_u_paths) and backdoor_exists:  # agent_dec[0] as it should only have one entry because we've currently restricted it to the single dec case
                return True

    return False


def has_dir_control_inc(self, node, agent):
    """
    returns True if a node faces a direct control incentive
    """
    agent_dec = self.all_decision_nodes[
        agent]  # decision made by this agent (this incentive is currently only proven to hold for the single decision case)
    agent_utils = self.all_utility_nodes[agent]  # this agent's utility nodes
    trimmed_MACID = self.dreduction(agent)

    for util in agent_utils:
        if trimmed_MACID.has_control_inc(node, agent):
            x_u_paths = trimmed_MACID.find_all_dir_path(node, util)
            for path in x_u_paths:
                if set(agent_dec).isdisjoint(set(path)):
                    return True
    return False
