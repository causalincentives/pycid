# ---------- Extra Graphical criterion methods for finding Pfeffer and Gal's Reasoning patterns -----------------------

def _directed_decision_free_path(self, start: str, finish: str):
    """
    checks to see if a directed decision free path exists
    """
    start_finish_paths = self.find_all_dir_path(start, finish)
    dec_free_path_exists = any(set(self.all_decision_nodes).isdisjoint(set(path[1:-1])) for path in start_finish_paths)  # ignore path's start and finish node
    if start_finish_paths and dec_free_path_exists:
        return True
    else:
        return False


def effective_dir_path_exists(self, start: str, finish: str, effective_set: List[str]):
    """
    checks whether an effective directed path exists

    """
    start_finish_paths = self.find_all_dir_path(start, finish)
    for path in start_finish_paths:
        if self._path_is_effective(path, effective_set):
            return True
    else:
        return False

def effective_undir_path_exists(self, start: str, finish: str, effective_set: List[str]):
    """
    checks whether an effective undirected path exists
    """
    start_finish_paths = self.find_all_undir_path(start, finish)
    for path in start_finish_paths:
        if self._path_is_effective(path, effective_set):
            return True
    else:
        return False


def _path_is_effective(self, path:List[str], effective_set: List[str]):
    """
    checks whether a path is effective
    """
    dec_nodes_in_path = set(self.all_decision_nodes).intersection(set(path[1:]))  #exclude first node of the path
    all_dec_nodes_effective = all(dec_node in effective_set for dec_node in dec_nodes_in_path)   #all([]) evaluates to true => this covers case where path has no decision nodes
    if all_dec_nodes_effective:
        return True
    else:
        return False


def directed_effective_path_not_through_Y(self, start: str, finish: str, effective_set: List[str], Y:List[str]=[]):
    """
    checks whether a directed effective path exists that doesn't pass through any of the nodes in the set Y.
    """
    start_finish_paths = self.find_all_dir_path(start, finish)
    for path in start_finish_paths:
        path_not_through_Y = set(Y).isdisjoint(set(path))
        if self._path_is_effective(path, effective_set) and path_not_through_Y:
            return True
    else:
        return False


def effective_backdoor_path_not_blocked_by_W(self, start: str, finish: str, effective_set: List[str], W:List[str]=[]):
    """
    returns the effective backdoor path not blocked if we condition on nodes in set W. If no such path exists, this returns false
    """
    start_finish_paths = self.find_all_undir_path(start, finish)
    for path in start_finish_paths:
        is_backdoor_path = path[1] in self.get_parents(path[0])
        not_blocked_by_W = not self.path_d_separated_by_Z(path, W)
        if is_backdoor_path and self._path_is_effective(path, effective_set) and not_blocked_by_W:
            return path
    else:
        return False


def effective_undir_path_not_blocked_by_W(self, start: str, finish: str, effective_set: List[str], W:List[str]=[]):
    """
    returns an effective undirected path not blocked if we condition on nodes in set W. If no such path exists, this returns false.
    """
    start_finish_paths = self.find_all_undir_path(start, finish)
    for path in start_finish_paths:
        not_blocked_by_W = not self.path_d_separated_by_Z(path, W)
        if self._path_is_effective(path, effective_set) and not_blocked_by_W:
            return path
    else:
        return False

# ------------ Pfeffer and Gal's Reasoning Patterns ----------------


def direct_effect(self, dec: str, effective_set: List[str]):
    """checks to see whether this decision is motivated by an incentive for a direct effect

    Graphical Criterion:
    1) There is a directed decision free path from D_A to a utility node U_A

    """
    agent = self._get_dec_agent(dec)
    print(agent)
    agent_utils = self.all_utility_nodes[agent]
    print(agent_utils)
    for u in agent_utils:
        if self._directed_decision_free_path(dec,u):
            return True
    else:
        return False

def manipulation(self, dec: str, effective_set: List[str]):
    """checks to see whether this decision is motivated by an incentive for manipulation

    Graphical Criterion:
    1) There is a directed decision-free path from D_A to an effective decision node D_B.
    2) There is a directed, effective path from D_B to U_A (an effective path is a path in which all decision nodes, except possibly the initial node, and except fork nodes, are effective)
    3) There is a directed, effective path from D_A to U_B that does not pass trhough D_B.
    """
    agent = self._get_dec_agent(dec)
    agent_utils = self.all_utility_nodes[agent]
    reachable_decisions = []    #set of possible D_B
    list_decs = copy.deepcopy(self.all_decision_nodes)
    list_decs.remove(dec)
    for dec_reach in list_decs:
        if dec_reach in effective_set:
            if self._directed_decision_free_path(dec, dec_reach):
                reachable_decisions.append(dec_reach)

    for dec_B in reachable_decisions:
        agentB = self._get_dec_agent(dec_B)
        agentB_utils = self.all_utility_nodes[agentB]

        for u in agent_utils:
            if self.effective_dir_path_exists(dec_B, u, effective_set):

                for u_B in agentB_utils:
                    if self.directed_effective_path_not_through_Y(dec, u_B, effective_set, [dec_B]):
                        return True
    else:
        return False


def signaling(self, dec: str, effective_set: List[str]):
    """checks to see whether this decision is motivated by an incentive for signaling

    Graphical Criterion:
    1) There is a directed decision-free path from D_A to an effective decision node D_B.
    2) There is a directed, effective path from D_B to U_A.
    3) There is an effective back-door path π from D_A to U_B that is not blocked by D_B U W^{D_A}_{D_B}.
    4) If C is the key node in π, there is an effective path from C to U_A that is not blocked by D_A U W^{C}_{D_A}

    """

    agent = self._get_dec_agent(dec)
    agent_utils = self.all_utility_nodes[agent]
    reachable_decisions = []    #set of possible D_B
    list_decs = copy.deepcopy(self.all_decision_nodes)
    list_decs.remove(dec)
    for dec_reach in list_decs:
        if dec_reach in effective_set:
            if self._directed_decision_free_path(dec, dec_reach):
                reachable_decisions.append(dec_reach)

    for dec_B in reachable_decisions:
        agentB = self._get_dec_agent(dec_B)
        agentB_utils = self.all_utility_nodes[agentB]
        for u in agent_utils:
            if self.effective_dir_path_exists(dec_B, u, effective_set):

                for u_B in agentB_utils:
                    D_B_parents_not_desc_dec = self.parents_of_Y_not_descended_from_X(dec, dec_B)
                    cond_nodes = [dec_B] + D_B_parents_not_desc_dec

                    if self.effective_backdoor_path_not_blocked_by_W(dec, u_B, effective_set, cond_nodes):
                        path = self.effective_backdoor_path_not_blocked_by_W(dec, u_B, effective_set, cond_nodes)
                        key_node = self.get_key_node(path)
                        dec_parents_not_desc_key = self.parents_of_Y_not_descended_from_X(key_node, dec)
                        cond_nodes2 = [dec] + dec_parents_not_desc_key

                        if self.effective_undir_path_not_blocked_by_W(key_node, u, effective_set, cond_nodes2):
                            return True
    else:
        return False



def revealing_or_denying(self, dec: str, effective_set: List[str]):
    """checks to see whether this decision is motivated by an incentive for revealing or denying

    Graphical Criterion:
    1) There is a directed decision-free path from D_A to an effective decision node D_B.
    2) There is a direced, effective path from D_B to U_A.
    3) There is an effective indirect front-door path π from D_A to U_B that is not blocked by D_B U W^{D_A}_{D_B}.
    """
    agent = self._get_dec_agent(dec)
    agent_utils = self.all_utility_nodes[agent]
    reachable_decisions = []    #set of possible D_B
    list_decs = copy.deepcopy(self.all_decision_nodes)
    list_decs.remove(dec)
    for dec_reach in list_decs:
        if dec_reach in effective_set:
            if self._directed_decision_free_path(dec, dec_reach):
                reachable_decisions.append(dec_reach)

    for dec_B in reachable_decisions:
        agentB = self._get_dec_agent(dec_B)
        agentB_utils = self.all_utility_nodes[agentB]

        for u in agent_utils:
            if self.effective_dir_path_exists(dec_B, u, effective_set):

                for u_B in agentB_utils:
                    D_B_parents_not_desc_dec = self.parents_of_Y_not_descended_from_X(dec, dec_B)
                    cond_nodes = [dec_B] + D_B_parents_not_desc_dec

                    if self.frontdoor_indirect_path_not_blocked_by_W(dec, u_B, cond_nodes):
                        return True
    else:
        return False

def find_motivations(self):
    """ This finds all of the circumstances under which an agent in a MAID has a reason to prefer one strategy over another, when all
    other agents are playing WD strategies (Pfeffer and Gal, 2007: On the Reasoning patterns of Agents in Games).
    """
    motivations = {'dir_effect':[], 'sig':[], 'manip':[], 'rev_den':[]}
    effective_set = list(self.all_decision_nodes)
    while True:
        new_set = [D for D in effective_set if self.direct_effect(D, effective_set) or self.manipulation(D, effective_set) \
            or self.signaling(D, effective_set) or self.revealing_or_denying(D, effective_set)]

        if len(new_set)==len(effective_set):
            break
        effective_set = new_set


    for decision in effective_set:
        if self.direct_effect(decision, effective_set):
            motivations['dir_effect'].append(decision)
        elif self.signaling(decision, effective_set):
            motivations['sig'].append(decision)
        elif self.manipulation(decision, effective_set):
            motivations['manip'].append(decision)
        elif self.revealing_or_denying(decision, effective_set):
            motivations['rev_den'].append(decision)

    return motivations






