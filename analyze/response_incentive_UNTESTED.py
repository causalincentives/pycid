
"""Response incentive
Criterion for response incentive on X:
(i) there is a directed path from X--> D in the reduced graph G*
"""

def has_response_inc(self, node: str, agent):
    """
    returns True if a node faces a response incentive"
    """
    agent_dec = self.all_decision_nodes[agent] #decision made by this agent (this incentive is currently only proven to hold for the single decision case)
    agent_utils = self.all_utility_nodes[agent] #this agent's utility nodes

    if len(agent_dec) > 1:
        return "This incentive currently only applies to the single decision case"

    trimmed_MACID = self.dreduction(agent)

    if agent_dec[0] in nx.descendants(trimmed_MACID, node):
            return True

    return False

def all_response_inc_nodes(self, agent):

    return [x for x in list(self.nodes) if self.has_response_inc(x, agent)]
