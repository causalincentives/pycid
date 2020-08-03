from pgmpy.models import BayesianModel
import networkx as nx
from macid import MACID
from typing import List
import copy


class Reasoning():
    """This class contains:
    -  Methods for identifying graphical criteria on MACIDs
    - Method for determining reasonsing patterns in MACIDs (Pfeffer and Gal, 2007: On the Reasoning patterns of Agents in Games)
    - A method that finds all of the circumstances under which an agent in a MAID has a reason to prefer one strategy over another, when all
        other agents are playing WD strategies (Pfeffer and Gal, 2007: On the Reasoning patterns of Agents in Games).
    """

    def __init__(self, model):
        self.model = model

        try:
            assert isinstance(model, BayesianModel)
        except AssertionError:
            raise AssertionError("Reasoning instantiation's first argument must be a BayesianModel")

    # ------------------------- Methods for identifying graphical criteria on MACIDs -----------------------
    
    def _get_dec_agent(self, dec):
        """
        finds which agent this decision node belongs to
        """
        for agent, decisions in self.model.decision_nodes.items():
            if dec in decisions:
                return agent

    def _get_util_agent(self, util):
        """
        finds which agent this utility node belongs to
        """
        for agent, utilities in self.model.utility_nodes.items():
            if util in utilities:
                return agent

    

    def _find_dirpath_recurse(self, path: List[str], finish: str, all_paths: str):    
        if path[-1] == finish:
            return path
        else:
            children = self.model.get_children(path[-1])
            for child in children:
                ext = path + [child]
                ext = _find_dirpath_recurse(ext, finish, all_paths)
                if ext and ext[-1] == finish:  # the "if ext" checks to see that it's a full directed path.
                    all_paths.append(ext)
                else:
                    continue
            return all_paths

    def find_all_dir_path(self, start: str, finish: str):
        """
        finds all direct paths from start node to end node that exist in the MAID
        """
        all_paths = []
        return _find_dirpath_recurse([start], finish, all_paths)


    def _find_undirpath_recurse(self, path: List[str], finish: str, all_paths: str):      
        if path[-1] == finish:
            return path
        else:
            neighbours = list(self.model.get_children(path[-1])) + list(self.model.get_parents(path[-1]))
            new = set(neighbours).difference(set(path))
            for child in new:
                ext = path + [child]
                ext = _find_undirpath_recurse(ext, finish, all_paths)
                if ext and ext[-1] == finish:  # the "if ext" checks to see that it's a full directed path.
                    all_paths.append(ext)
                else:
                    continue
            return all_paths

    def find_all_undir_path(self, start: str, finish: str):
        """
        finds all direct paths from start node to end node that exist in the MAID
        """
        all_paths = []
        return _find_undirpath_recurse([start], finish, all_paths)



    def _directed_decision_free_path(self, start: str, finish: str):
        """
        checks to see if a directed decision free path exists
        """
        start_finish_paths = find_all_dir_path(start, finish)
        dec_free_path_exists = any(set(self.model.all_decision_nodes).isdisjoint(set(path[1:-1])) for path in start_finish_paths)  # ignore path's start and finish node
        if start_finish_paths and dec_free_path_exists:
            return True
        else:
            return False

    


    def effective_dir_path_exists(self, start: str, finish: str, effective_set: List[str]):
        """
        checks whether an effective directed path exists
        """
        start_finish_paths = find_all_dir_path(start, finish)
        for path in start_finish_paths:
            if _path_is_effective(path, effective_set):
                return True        
        else:
            return False

    def effective_undir_path_exists(self, start: str, finish: str, effective_set: List[str]):
        """
        checks whether an effective undirected path exists
        """
        start_finish_paths = find_all_undir_path(start, finish)
        for path in start_finish_paths:
            if _path_is_effective(path, effective_set):
                return True
        else:
            return False


    def _path_is_effective(self, path:List[str], effective_set: List[str]):
        """
        checks whether a path is effective
        """
        dec_nodes_in_path = set(self.model.all_decision_nodes).intersection(set(path[1:]))  #exclude first node of the path
        all_dec_nodes_effective = all(dec_node in effective_set for dec_node in dec_nodes_in_path)   #all([]) evaluates to true => this covers case where path has no decision nodes
        if all_dec_nodes_effective:
            return True
        else:
            return False


    def directed_effective_path_not_through_Y(self, start: str, finish: str, effective_set: List[str], Y:List[str]=[]):
        """
        checks whether a directed effective path exists that doesn't pass through any of the nodes in the set Y.
        """
        start_finish_paths = find_all_dir_path(start, finish)
        for path in start_finish_paths:
            path_not_through_Y = set(Y).isdisjoint(set(path))
            if _path_is_effective(path, effective_set) and path_not_through_Y:
                return True
        else:
            return False


    def backdoor_path_not_blocked_by_W(self, start: str, finish: str, effective_set: List[str], W:List[str]=[]):
        """
        returns the effective backdoor path not blocked if we condition on nodes in set W. If no such path exists, this returns false
        """
        start_finish_paths = find_all_undir_path(start, finish)
        for path in start_finish_paths:
            is_backdoor_path = path[1] in self.model.get_parents(path[0])
            not_blocked_by_W = not path_d_separated_by_Z(path, W)
            if is_backdoor_path and _path_is_effective(path, effective_set) and not_blocked_by_W:
                return path
        else:
            return False


    def effective_undir_path_not_blocked_by_W(self, start: str, finish: str, effective_set: List[str], W:List[str]=[]):
        """
        returns an effective undirected path not blocked if we condition on nodes in set W. If no such path exists, this returns false.
        """
        start_finish_paths = find_all_undir_path(start, finish)
        for path in start_finish_paths:
            not_blocked_by_W = not path_d_separated_by_Z(path, W)
            if _path_is_effective(path, effective_set) and not_blocked_by_W:
                return path
        else:
            return False
                
    
    def _get_path_structure(self, path:List[str]):
        """
        returns the path's structure (ie the direction of the edges that make up this path)
        """
        structure = []
        for i in range(len(path)-1):
            if path[i] in self.model.get_parents(path[i+1]):
                structure.append((path[i], path[i+1]))
            elif path[i+1] in self.model.get_parents(path[i]):    
                structure.append((path[i+1], path[i]))
        return structure


    def path_d_separated_by_Z(self, path:List[str], Z:List[str]=[]):
        """
        Check if a path is d-separated by set of variables Z.
        """

        if len(path) < 3:
            return False

        for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
            structure = _classify_three_structure(a, b, c)

            if structure in ("chain", "fork") and b in Z:
                return True

            if structure == "collider":
                descendants = (nx.descendants(self.model, b) | {b})
                if not descendants & set(Z):
                    return True

        return False


    def _classify_three_structure(self, a: str, b: str, c: str):
        """
        Classify three node structure as a chain, fork or collider.
        """
        if self.model.has_edge(a, b) and self.model.has_edge(b, c):
            return "chain"

        if self.model.has_edge(c, b) and self.model.has_edge(b, a):
            return "chain"

        if self.model.has_edge(a, b) and self.model.has_edge(c, b):
            return "collider"

        if self.model.has_edge(b, a) and self.model.has_edge(b, c):
            return "fork"

        raise ValueError(f"Unsure how to classify ({a},{b},{c})")

    

    def frontdoor_indirect_path_not_blocked_by_W(self, start: str, finish: str, W:List[str]=[]):
        """checks whether an indirect frontdoor path exists that isn't blocked by the nodes in set W."""
        start_finish_paths = find_all_undir_path(start, finish)
        for path in start_finish_paths:
            is_frontdoor_path = path[0] in self.model.get_parents(path[1])
            not_blocked_by_W = not path_d_separated_by_Z(path, W)
            contains_collider = _path_contains_collider(path)
            if is_frontdoor_path and not_blocked_by_W and contains_collider:   #default (if w = [] is going to be false since any unobserved collider blocks path
                return True
        else:
            return False


    def _path_contains_collider(self, path:List[str]):
        """checks whether the path contains a collider"""
        if len(path) < 3:
            return False

        for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
            structure = _classify_three_structure(a, b, c)
            if structure == "collider":
                return True
        else:
            return False

    def parents_of_Y_not_descended_from_X(self, X: str,Y: str):
        """finds the parents of Y not descended from X"""
        Y_parents = self.model.get_parents(Y)
        X_descendants = list(nx.descendants(self.model, X))
        print(f" desc of {X} are {X_descendants}")
        return list(set(Y_parents).difference(set(X_descendants)))


    def get_key_node(self, path:List[str]):
        """ The key node of a path is the first "fork" node in the path"""
        for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
            structure = _classify_three_structure(a, b, c)
            if structure == "fork":
                return b


    # ----------------- reasoning patterns in MACIDs  -------------

    def direct_effect(self, dec: str, effective_set: List[str]):
        """checks to see whether this decision is motivated by an incentive for a direct effect"""
        agent = _get_dec_agent(dec)
        print(agent)
        agent_utils = self.model.utility_nodes[agent]
        print(agent_utils)
        for u in agent_utils:
            if _directed_decision_free_path(dec,u):
                return True
        else:
            return False

    def manipulation(self, dec: str, effective_set: List[str]):
        """checks to see whether this decision is motivated by an incentive for manipulation"""
        agent = _get_dec_agent(dec)
        agent_utils = self.model.utility_nodes[agent]
        reachable_decisions = []    #set of possible D_B
        list_decs = copy.deepcopy(self.model.all_decision_nodes)
        list_decs.remove(dec)
        for dec_reach in list_decs:
            if dec_reach in effective_set:
                if _directed_decision_free_path(dec, dec_reach):
                    reachable_decisions.append(dec_reach)

        for dec_B in reachable_decisions:
            agentB = _get_dec_agent(dec_B)
            agentB_utils = self.model.utility_nodes[agentB]

            for u in agent_utils:
                if effective_dir_path_exists(dec_B, u, effective_set):

                    for u_B in agentB_utils:
                        if directed_effective_path_not_through_Y(dec, u_B, effective_set, [dec_B]):
                            return True
        else:
            return False
                

    def signaling(self, dec: str, effective_set: List[str]):
        """checks to see whether this decision is motivated by an incentive for signaling"""
        
        agent = _get_dec_agent(dec)
        agent_utils = self.model.utility_nodes[agent]
        reachable_decisions = []    #set of possible D_B
        list_decs = copy.deepcopy(self.model.all_decision_nodes)
        list_decs.remove(dec)
        for dec_reach in list_decs:
            if dec_reach in effective_set:
                if _directed_decision_free_path(dec, dec_reach):
                    reachable_decisions.append(dec_reach)

        for dec_B in reachable_decisions:
            agentB = _get_dec_agent(dec_B)
            agentB_utils = self.model.utility_nodes[agentB]
            for u in agent_utils:
                if effective_dir_path_exists(dec_B, u, effective_set):

                    for u_B in agentB_utils:
                        D_B_parents_not_desc_dec = parents_of_Y_not_descended_from_X(dec, dec_B)
                        cond_nodes = [dec_B] + D_B_parents_not_desc_dec

                        if backdoor_path_not_blocked_by_W(dec, u_B, effective_set, cond_nodes):
                            path = backdoor_path_not_blocked_by_W(dec, u_B, effective_set, cond_nodes)
                            key_node = get_key_node(path)
                            dec_parents_not_desc_key = parents_of_Y_not_descended_from_X(key_node, dec)
                            cond_nodes2 = [dec] + dec_parents_not_desc_key
                            
                            if effective_undir_path_not_blocked_by_W(key_node, u, effective_set, cond_nodes2):
                                return True
        else:
            return False



    def revealing_or_denying(self, dec: str, effective_set: List[str]):
        """checks to see whether this decision is motivated by an incentive for revealing or denying"""
        agent = _get_dec_agent(dec)
        agent_utils = self.model.utility_nodes[agent]
        reachable_decisions = []    #set of possible D_B
        list_decs = copy.deepcopy(self.model.all_decision_nodes)
        list_decs.remove(dec)
        for dec_reach in list_decs:
            if dec_reach in effective_set:
                if _directed_decision_free_path(dec, dec_reach):
                    reachable_decisions.append(dec_reach)
        
        for dec_B in reachable_decisions:
            agentB = _get_dec_agent(dec_B)
            agentB_utils = self.model.utility_nodes[agentB]
            
            for u in agent_utils:
                if effective_dir_path_exists(dec_B, u, effective_set):
                    
                    for u_B in agentB_utils:
                        D_B_parents_not_desc_dec = parents_of_Y_not_descended_from_X(dec, dec_B)
                        cond_nodes = [dec_B] + D_B_parents_not_desc_dec
                        
                        if frontdoor_indirect_path_not_blocked_by_W(dec, u_B, cond_nodes):
                            return True
        else:
            return False

    def find_motivations(self):
        """ This finds all of the circumstances under which an agent in a MAID has a reason to prefer one strategy over another, when all
        other agents are playing WD strategies (Pfeffer and Gal, 2007: On the Reasoning patterns of Agents in Games).
        """     
        motivations = {'dir_effect':[], 'sig':[], 'manip':[], 'rev_den':[]}
        effective_set = list(self.model.all_decision_nodes)
        while True:
            nodes_not_effective = []
            for node in effective_set:
                if not direct_effect(node, effective_set) and not manipulation(node, effective_set) \
                    and not signaling(node, effective_set) and not revealing_or_denying(node, effective_set):
                    nodes_not_effective.append(node)
            if len(nodes_not_effective) > 0:
                effective_set = [node for node in effective_set if node not in nodes_not_effective]
                print(f"ne {nodes_not_effective}")
                print(f"e is {effective_set}")
            elif len(nodes_not_effective) ==0:
                print(f"e is {effective_set}")
                break


        for node in effective_set:
            if direct_effect(node, effective_set):
                motivations['dir_effect'].append(node)
            elif signaling(node, effective_set):
                motivations['sig'].append(node)
            elif manipulation(node, effective_set):
                motivations['manip'].append(node)
            elif revealing_or_denying(node, effective_set):
                motivations['rev_den'].append(node)

        return motivations