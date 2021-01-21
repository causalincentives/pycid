def has_control_inc(self, node: str, agent):
    """
    returns True if a node faces a control incentive or "positive value of control"
    """
    agent_dec = self.all_decision_nodes[
        agent]  # decision made by this agent (this incentive is currently only proven to hold for the single decision case)
    agent_utils = self.all_utility_nodes[agent]  # this agent's utility nodes

    if not agent_dec or not agent_utils:  # if the agent has no decision or no utility nodes, no node will face a control incentive
        return False

    if len(agent_dec) > 1:
        return "This incentive currently only applies to the single decision case"

    if [node] == agent_dec:  # condition (i)
        return False

    trimmed_MACID = self.dreduction(agent)

    for util in agent_utils:
        if node == util or util in nx.descendants(trimmed_MACID, node):  # condition (ii)
            return True

    return False


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
