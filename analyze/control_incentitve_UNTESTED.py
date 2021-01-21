# Control Incentive
    """Control incentive
    Criterion for a control incentive on X:
    (i) X is not a decision node
    (ii) iff there is a directed path X --> U in the reduced graph G*

    The reduced graph G* of a MACID G is the result of removing from G information links Y -> D from all non-requisite
    observations Y ∈ Pa_D

    The control incentive is (understanding agent incentives using CIDs):
    A) Direct: if the directed path X --> U does not pass through D
    B) Indirect: if the directed path X --> U does pass through D and there is a backdoor path X -- U that begins
    backwards from X(···←X) and is active when conditioning on Fa_D \ {X}


    (The incentives that shape behaviour)
    Positive value of control for a node X ∈ V \ {D} iff there exists a directed path x --> U in the reduced graph (so this is the same as vanilla control incentive above)
    A feasible control incentive exists iff there exists a directed path D --> X --> U
    """

    def dreduction(self, agent):
        """
        returns the DAG which has been trimmed of all irrelevant information links.
        """
        assert (len(self.all_decision_nodes) ==1) ,"The theory currently only works for the single-decision case!"
        agent_dec = self.all_decision_nodes[agent] #decision made by this agent (this incentive is currently only proven to hold for the single decision case)
        agent_utils = self.all_utility_nodes[agent] #this agent's utility nodes

        trimmed_graph = self.copy()
        d_par = self.get_parents(*agent_dec)
        nonrequisite_nodes = [n for n in d_par if not self.has_info_inc(n, agent)]

        for node in nonrequisite_nodes:
            trimmed_graph.remove_edge(node, *agent_dec)
        return trimmed_graph




    def has_feasible_control_inc(self, node, agent):
        """
        returns True if a node faces a feasible control incentive
        """
        agent_dec = self.all_decision_nodes[agent] #decision made by this agent (this incentive is currently only proven to hold for the single decision case)
        agent_utils = self.all_utility_nodes[agent] #this agent's utility nodes

        if [node] == agent_dec:  #ignore decision node
            return False

        for util in agent_utils:
            D_u_paths = self.find_all_dir_path(agent_dec[0], util)
            if any(node in path for path in D_u_paths):
                return True
        return False


    def all_control_inc_nodes(self, agent):

        return [x for x in list(self.nodes) if self.has_control_inc(x, agent)]

    def all_dir_control_inc_nodes(self, agent):

        return [x for x in list(self.nodes) if self.has_dir_control_inc(x, agent)]

    def all_indir_control_inc_nodes(self, agent):

        return [x for x in list(self.nodes) if self.has_indir_control_inc(x, agent)]

    def all_feasible_control_inc_nodes(self, agent):

        return [x for x in list(self.nodes) if self.has_feasible_control_inc(x, agent)]
