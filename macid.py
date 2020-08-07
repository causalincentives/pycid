#Licensed to the Apache Software Foundation (ASF) under one or more contributor license 
#agreements; and to You under the Apache License, Version 2.0.

import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.factors.discrete import TabularCPD
import logging
import typing
from typing import List, Tuple, Dict
import itertools
from pgmpy.inference import BeliefPropagation
import functools
import networkx as nx
from cpd import NullCPD
import matplotlib.pyplot as plt
import operator
from collections import defaultdict
from collections import deque
import copy
import matplotlib.cm as cm
from itertools import compress
from get_paths import get_motifs, get_motif




class MACID(BayesianModel):
    def __init__(self, ebunch:List[Tuple[str, str]]=None, node_types:Dict=None, utility_domains:Dict=None ):
        super(MACID, self).__init__(ebunch=ebunch)
        self.node_types = node_types
        self.utility_domains = utility_domains
        self.utility_nodes = {i:node_types[i]['U'] for i in node_types if i != 'C'}     # this gives a dictionary matching each agent with their decision and utility nodes
        self.decision_nodes = {i:node_types[i]['D'] for i in node_types if i != 'C'}     #  eg {'A': ['U1', 'U2'], 'B': ['U3', 'U4']}    
        self.chance_nodes = node_types['C']     # list of chance nodes
        self.agents = [agent for agent in node_types if agent != 'C']   # gives a list of the MAID's agents
        self.all_utility_nodes = list(itertools.chain(*self.utility_nodes.values()))        
        self.all_decision_nodes = list(itertools.chain(*self.decision_nodes.values()))        
        self.reversed_acyclic_ordering = list(reversed(self.get_acyclic_topological_ordering()))     
        self.numDecisions = len(self.reversed_acyclic_ordering)  

    def copy(self):
        model_copy = MACID(node_types=self.node_types)
        model_copy.add_nodes_from(self.nodes())
        model_copy.add_edges_from(self.edges())
        if self.cpds:
            model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        return model_copy


    def add_cpds(self, *cpds):
        # this method adds conditional probability distributions to the MACID.
        for cpd in cpds:
            if not isinstance(cpd, (TabularCPD, ContinuousFactor, NullCPD)):
                raise ValueError("Only TabularCPD, ContinuousFactor, or NullCPD can be added.")

            if set(cpd.scope()) - set(cpd.scope()).intersection(set(self.nodes())):
                raise ValueError("CPD defined on variable not in the model", cpd)

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    logging.warning(
                        "Replacing existing CPD for {var}".format(var=cpd.variable)
                    )
                    self.cpds[prev_cpd_index] = cpd
                    break
            else:
                self.cpds.append(cpd)


    def check_model(self, allow_null=True):
        """
        Check the model for various errors. This method checks for the following
        errors.

        * Checks if the sum of the probabilities for each state is equal to 1 (tol=0.01).
        * Checks if the CPDs associated with nodes are consistent with their parents.

        Returns
        -------
        check: boolean
            True if all the checks are passed
        """
        for node in self.nodes():
            cpd = self.get_cpds(node=node)

            if cpd is None:
                raise ValueError("No CPD associated with {}".format(node))
            elif isinstance(cpd, (TabularCPD, ContinuousFactor)):
                evidence = cpd.get_evidence()
                parents = self.get_parents(node)
                if set(evidence if evidence else []) != set(parents if parents else []):
                    raise ValueError(
                        "CPD associated with {node} doesn't have "
                        "proper parents associated with it.".format(node=node)
                    )
                if not cpd.is_valid_cpd():
                    raise ValueError(
                        "Sum or integral of conditional probabilites for node {node}"
                        " is not equal to 1.".format(node=node)
                    )
            elif isinstance(cpd, (NullCPD)):
                if not allow_null:
                    raise ValueError(
                        "CPD associated with {node} is nullcpd".format(node=node)
                    )
        return True


# --------------   methods for plotting MACID ----------------

    def _set_single_agent_node_color(self, node: str):
        """ 
        Colour codes the decision, chance nodes to match the single-agent conventions:
        - decision nodes are blue
        - utility nodes are yellow
        """
        if self.get_node_type(node) == 'p':  # 'p' is a player/decision node to match with gambit's notation
            return 'lightblue'
        if self.get_node_type(node) == 'u':  # utility node
            return 'yellow'

    def _get_shape(self, node: str):
        """ 
        Colour codes the decision, chance and utility nodes of each agent
        """
        for i in self.node_types:
            if i == 'C':
                if node in self.node_types['C']:
                    return 'o'
            else:
                if node in self.node_types[i]['D']:
                    return 's'
                if node in self.node_types[i]['U']:
                    return 'D'

   
    def _set_multi_agent_node_color(self, node):
        """
        This matches a unique colour with each new agent's decision and utility nodes 
        """
        colors = cm.rainbow(np.linspace(0, 1, len(self.node_types)))
        if self.get_node_type(node) == 'p':  # 'p' is a player/decision node to match with gambit's notation
            return colors[self._get_dec_agent(node)]
        if self.get_node_type(node) == 'u':  # utility node
            return colors[self._get_util_agent(node)]


    def _get_cmap(self, n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n) 


    def _get_node_shape(self, node):
        """
        finds a node's shape
        """
        if self.get_node_type(node) == 'p':  #decision nodes should be squares ('p' matches gambit's player node notation)
            return 's'
        elif self.get_node_type(node) == 'u': #utility nodes should be diamonds 
            return 'D'


    def draw(self):
        """
        This draws the DAG for the MACID
        """
        l = nx.kamada_kawai_layout(self)
        G = self.to_undirected()
        nx.draw_networkx(self, pos=l, node_size=400, arrowsize=20, node_color='lightgray', node_shape='o')  #chance nodes should be gray circles
        if len(self.agents) == 1:
            for node in self.nodes:
                if self.get_node_type(node) != 'c':
                    nx.draw_networkx(G.subgraph([node]), pos=l, node_size=400, node_color=self._set_single_agent_node_color(node), node_shape=self._get_node_shape(node))

        else:
            for node in self.nodes:
                if self.get_node_type(node) != 'c':
                    nx.draw_networkx(G.subgraph([node]), pos=l, node_size=400, node_color=self._set_multi_agent_node_color(node).reshape(1,-1), node_shape=self._get_node_shape(node))
       
            
# ---------- methods setting up MACID for probabilistic inference ------



    def random_instantiation_dec_nodes(self):
        """
        imputes random uniform policy to all decision nodes (NullCPDs) - arbitrary fully mixed strategy profile for MACID   #perhaps add something checking whether it's "isinstance(cpd, NullCPD)" is true
        """
        for dec in self.all_decision_nodes:
            dec_card = self.get_cardinality(dec)
            parents = self.get_parents(dec)
            parents_card = [self.get_cardinality(par) for par in parents]
            table = np.ones((dec_card, np.product(parents_card).astype(int))) / dec_card
            uniform_cpd = TabularCPD(variable=dec, variable_card=dec_card,
                            values=table, evidence=parents,
                            evidence_card=parents_card
                            )
            print(uniform_cpd)
            self.add_cpds(uniform_cpd)
    
        
        

# -------- Methods for finding MACID graphical properties --------------------

    def _find_dirpath_recurse(self, path: List[str], finish: str, all_paths):
       
        if path[-1] == finish:
            return path
        else:
            children = self.get_children(path[-1])
            for child in children:
                ext = path + [child]
                ext = self._find_dirpath_recurse(ext, finish, all_paths)
                if ext and ext[-1] == finish:  # the "if ext" checks to see that it's a full directed path.
                    all_paths.append(ext)
                else:
                    continue
            return all_paths

    def find_all_dir_path(self, start, finish):
        """
        finds all direct paths from start node to end node that exist in the MAID
        """
        all_paths = []    
        return self._find_dirpath_recurse([start], finish, all_paths)
        


    def _find_undirpath_recurse(self, path: List[str], finish: str, all_paths: str):

        if path[-1] == finish:
            return path
        else:
            neighbours = list(self.get_children(path[-1])) + list(self.get_parents(path[-1]))
            new = set(neighbours).difference(set(path))
            for child in new:
                ext = path + [child]
                ext = self._find_undirpath_recurse(ext, finish, all_paths)
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
        return self._find_undirpath_recurse([start], finish, all_paths)


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


    def _get_path_structure(self, path:List[str]):
        """
        returns the path's structure (ie the direction of the edges that make up this path)
        """
        structure = []
        for i in range(len(path)-1):
            if path[i] in self.get_parents(path[i+1]):
                structure.append((path[i], path[i+1]))
            elif path[i+1] in self.get_parents(path[i]):    
                structure.append((path[i+1], path[i]))
        return structure

    def path_d_separated_by_Z(self, path:List[str], Z:List[str]=[]):
        """
        Check if a path is d-separated by set of variables Z.
        """
        if len(path) < 3:
            return False

        for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
            structure = get_motif(self, path, path.index(b))

            if structure in ("chain", "fork") and b in Z:
                return True

            if structure == "collider":
                descendants = (nx.descendants(self, b) | {b})
                if not descendants & set(Z):
                    return True

        return False

    def frontdoor_indirect_path_not_blocked_by_W(self, start: str, finish: str, W:List[str]=[]):
        """checks whether an indirect frontdoor path exists that isn't blocked by the nodes in set W."""
        start_finish_paths = self.find_all_undir_path(start, finish)
        for path in start_finish_paths:
            is_frontdoor_path = path[0] in self.get_parents(path[1])
            not_blocked_by_W = not self.path_d_separated_by_Z(path, W)
            contains_collider = "collider" in get_motifs(self, path)
            if is_frontdoor_path and not_blocked_by_W and contains_collider:   #default (if w = [] is going to be false since any unobserved collider blocks path
                return True
        else:
            return False
        

    def parents_of_Y_not_descended_from_X(self, X: str,Y: str):
        """finds the parents of Y not descended from X"""
        Y_parents = self.get_parents(Y)
        X_descendants = list(nx.descendants(self, X))
        print(f" desc of {X} are {X_descendants}")
        return list(set(Y_parents).difference(set(X_descendants)))


    def get_key_node(self, path:List[str]):
        """ The key node of a path is the first "fork" node in the path"""
        for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
            structure = get_motif(self, path, path.index(b))
            if structure == "fork":
                return b

    def backdoor_path_active_when_conditioning_on_W(self, start: str, finish: str, W:List[str]=[]):
        """
        returns true if there is a backdoor path that's active when conditioning on nodes in set W. 
        """
        start_finish_paths = self.find_all_undir_path(start, finish)
        for path in start_finish_paths:
            
            if len(path) > 1:   #must have path of at least 2 nodes
                is_backdoor_path = path[1] in self.get_parents(path[0]) 
                not_blocked_by_W = not self.path_d_separated_by_Z(path, W) 
                if is_backdoor_path and not_blocked_by_W:
                    return True

        else:
            return False

    def backdoor_path_active_when_conditioning_on_W2(self, start: str, finish: str, W:List[str]=[]):
        """
        returns true if there is a backdoor path that's active when conditioning on nodes in set W. 
        """

        start_finish_paths = self.find_all_undir_path(start, finish)
        for path in start_finish_paths:
            #print(f"path1 = {path}")
            if len(path) > 1:   #must have path of at least 2 nodes
                is_backdoor_path = path[1] in self.get_parents(path[0])
                #print(f"is_bd_path {is_backdoor_path}")
                not_blocked_by_W = not self.path_d_separated_by_Z(path, W)
                #print(f"not_blocked = {not_blocked_by_W}")
                if is_backdoor_path and not_blocked_by_W:
                    #print(f"path is {path}")

                    return True

        else:
            return False


# ----------- methods for finding MACID properties -----------

    def _get_dec_agent(self, dec: str):
        """ 
        finds which agent a decision node belongs to
        """
        for agent, decisions in self.decision_nodes.items():
            if dec in decisions:
                return agent

    def _get_util_agent(self, util: str):
        """
        finds which agent a utility node belongs to
        """
        for agent, utilities in self.utility_nodes.items():
            if util in utilities:
                return agent

    def get_node_type(self, node):
        """
        finds a node's type
        """
        if node in self.chance_nodes:
            return 'c'
        elif node in self.all_decision_nodes:
            return 'p'  #gambit calls decision nodes player nodes
        elif node in self.all_utility_nodes:
            return 'u'
        else:
            return "node is not in MACID"




# ------------- Methods for Single agent incentives (using graphical criterion) -----------------------



## Information Incentive 
    """Criterion for Information incentive on X:
    (i) X is a node on the MACID which is not a decision node, D
    (ii) X is not a descendent of the decison node, D.
    (iii) U∈Desc(D) (U must be a descendent of D)
    (iv) X is d-connected to U | Fa_D\{X}
    
    

    """

    def has_info_inc(self, node: str, agent):
        """
        returns True if a node faces an information incentive
        """
        agent_dec = self.decision_nodes[agent] #decision made by this agent (this incentive is currently only proven to hold for the single decision case)
        agent_utils = self.utility_nodes[agent] #this agent's utility nodes
        
        if node not in self.nodes:
            raise ValueError(f"{node} is not present in the macid's graph")

        # #condition (i)
        if node in agent_dec or node in agent_utils:
            return False

        # condition (ii)
        elif node in nx.descendants(self, *agent_dec):
            return False
        
        for util in agent_utils:
            if util in nx.descendants(self, *agent_dec):  #condition (iii)
                con_nodes = agent_dec + self.get_parents(*agent_dec)  # nodes to be conditioned on
                if node in con_nodes:  # remove node from condition nodes
                    con_nodes.remove(node)
                if self.is_active_trail(node, util, con_nodes): #condition (iv)
                    return True
        else:
            return False

            
    
    def all_info_inc_nodes(self, agent):
        """
        returns all nodes which this agent has an information incentive for according to the single agent information incentive graphical criterion
        """
        agent_dec = self.decision_nodes[agent] #decision made by this agent (this incentive is currently only proven to hold for the single decision case)
        agent_utils = self.utility_nodes[agent] #this agent's utility nodes

        if not agent_dec or not agent_utils: #if the agent has no decision or no utility nodes, no node will face an information incentive
            return []
        
        elif len(agent_dec) > 1:
            return "This incentive currently only applies to the single decision case"

        else:
            return [x for x in list(self.nodes) if self.has_info_inc(x, agent)]
    ##



## Response Incentive

    """Response incentive
    Criterion for response incentive on X: 
    (i) there is a directed path from X--> D in the reduced graph G* 
    """



    def has_response_inc(self, node: str, agent):
        """
        returns True if a node faces a control incentive or "positive value of control"
        """
        agent_dec = self.decision_nodes[agent] #decision made by this agent (this incentive is currently only proven to hold for the single decision case)
        agent_utils = self.utility_nodes[agent] #this agent's utility nodes
        
        if len(agent_dec) > 1:
            return "This incentive currently only applies to the single decision case"

        trimmed_MACID = self.dreduction(agent)
        
        if agent_dec[0] in nx.descendants(trimmed_MACID, node): 
                return True
        
        return False

    def all_response_inc_nodes(self, agent):

        return [x for x in list(self.nodes) if self.has_response_inc(x, agent)]



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
        agent_dec = self.decision_nodes[agent] #decision made by this agent (this incentive is currently only proven to hold for the single decision case)
        agent_utils = self.utility_nodes[agent] #this agent's utility nodes

        trimmed_graph = self.copy()
        d_par = self.get_parents(*agent_dec)
        nonrequisite_nodes = [n for n in d_par if not self.has_info_inc(n, agent)]

        for node in nonrequisite_nodes:
            trimmed_graph.remove_edge(node, *agent_dec)
        return trimmed_graph

    
    def has_control_inc(self, node: str, agent):
        """
        returns True if a node faces a control incentive or "positive value of control"
        """
        agent_dec = self.decision_nodes[agent] #decision made by this agent (this incentive is currently only proven to hold for the single decision case)
        agent_utils = self.utility_nodes[agent] #this agent's utility nodes
        
        if not agent_dec or not agent_utils: #if the agent has no decision or no utility nodes, no node will face a control incentive
            return False

        if len(agent_dec) > 1:
            return "This incentive currently only applies to the single decision case"

        if [node] == agent_dec:  #condition (i)
            return False

        trimmed_MACID = self.dreduction(agent)
        
        for util in agent_utils:       
            if node == util or util in nx.descendants(trimmed_MACID, node): # condition (ii)
                return True
        
        return False

    def has_indir_control_inc(self, node, agent):
        """
        returns True if a node faces an indirect control incentive 
        """
        agent_dec = self.decision_nodes[agent] #decision made by this agent (this incentive is currently only proven to hold for the single decision case)
        agent_utils = self.utility_nodes[agent] #this agent's utility nodes
        trimmed_MACID = self.dreduction(agent)

        for util in agent_utils:
            if trimmed_MACID.has_control_inc(node, agent):

                Fa_d = trimmed_MACID.get_parents(*agent_dec) + agent_dec
                con_nodes = [i for i in Fa_d if i != node]
                backdoor_exists = trimmed_MACID.backdoor_path_active_when_conditioning_on_W(node, util, con_nodes)
                x_u_paths = trimmed_MACID.find_all_dir_path(node, util)
                if any(agent_dec[0] in paths for paths in x_u_paths) and backdoor_exists:  #agent_dec[0] as it should only have one entry because we've currently restricted it to the single dec case
                    return True
        
        return False

    def has_dir_control_inc(self, node, agent):
        """
        returns True if a node faces a direct control incentive
        """
        agent_dec = self.decision_nodes[agent] #decision made by this agent (this incentive is currently only proven to hold for the single decision case)
        agent_utils = self.utility_nodes[agent] #this agent's utility nodes  
        trimmed_MACID = self.dreduction(agent)
        
        for util in agent_utils:
            if trimmed_MACID.has_control_inc(node, agent):
                x_u_paths = trimmed_MACID.find_all_dir_path(node, util)
                for path in x_u_paths:
                    if set(agent_dec).isdisjoint(set(path)):
                        return True
        return False


    def has_feasible_control_inc(self, node, agent):
        """
        returns True if a node faces a feasible control incentive
        """
        agent_dec = self.decision_nodes[agent] #decision made by this agent (this incentive is currently only proven to hold for the single decision case)
        agent_utils = self.utility_nodes[agent] #this agent's utility nodes

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
        agent_utils = self.utility_nodes[agent]
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
        agent_utils = self.utility_nodes[agent]
        reachable_decisions = []    #set of possible D_B
        list_decs = copy.deepcopy(self.all_decision_nodes)
        list_decs.remove(dec)
        for dec_reach in list_decs:
            if dec_reach in effective_set:
                if self._directed_decision_free_path(dec, dec_reach):
                    reachable_decisions.append(dec_reach)

        for dec_B in reachable_decisions:
            agentB = self._get_dec_agent(dec_B)
            agentB_utils = self.utility_nodes[agentB]

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
        agent_utils = self.utility_nodes[agent]
        reachable_decisions = []    #set of possible D_B
        list_decs = copy.deepcopy(self.all_decision_nodes)
        list_decs.remove(dec)
        for dec_reach in list_decs:
            if dec_reach in effective_set:
                if self._directed_decision_free_path(dec, dec_reach):
                    reachable_decisions.append(dec_reach)

        for dec_B in reachable_decisions:
            agentB = self._get_dec_agent(dec_B)
            agentB_utils = self.utility_nodes[agentB]
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
        agent_utils = self.utility_nodes[agent]
        reachable_decisions = []    #set of possible D_B
        list_decs = copy.deepcopy(self.all_decision_nodes)
        list_decs.remove(dec)
        for dec_reach in list_decs:
            if dec_reach in effective_set:
                if self._directed_decision_free_path(dec, dec_reach):
                    reachable_decisions.append(dec_reach)
        
        for dec_B in reachable_decisions:
            agentB = self._get_dec_agent(dec_B)
            agentB_utils = self.utility_nodes[agentB]
            
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









# -------------Methods for returning all pure strategy subgame perfect NE ------------------------------------------------------------


    def _instantiate_initial_tree(self):
        #creates a tree (a nested dictionary) which we use to fill up with the subgame perfect NE of each sub-tree. 
        cardinalities = map(self.get_cardinality, self.all_decision_nodes)
        decision_cardinalities = dict(zip(self.all_decision_nodes, cardinalities)) #returns a dictionary matching each decision with its cardinality

        action_space_list = list(itertools.accumulate(decision_cardinalities.values(), operator.mul))  #gives number of pure strategies for each decision node (ie taking into account prev decisions)
        cols_in_each_tree_row = [1] + action_space_list

        actions_for_dec_list = []
        for card in decision_cardinalities.values():
            actions_for_dec_list.append(list(range(card)))     # appending the range of actions each decion can make as a list
        final_row_actions = list(itertools.product(*actions_for_dec_list))     # creates entry for final row of decision array (cartesian product of actions_seq_list))

        tree_initial = defaultdict(dict)   # creates a nested dictionary
        for i in range(0, self.numDecisions+1):
            for j in range(cols_in_each_tree_row[i]):     #initialises entire decision/action array with empty tuples.
                tree_initial[i][j] = ()

        for i in range(cols_in_each_tree_row[-1]):
            tree_initial[self.numDecisions][i] = final_row_actions[i]
               
        trees_queue = [tree_initial]  # list of all possible decision trees 
        return trees_queue


    def _reduce_tree_once(self, queue:List[str], bp):
        #finds node not yet evaluated and then updates tree by evaluating this node - we apply this repeatedly to fill up all nodes in the tree
        tree = queue.pop(0)
        for row in range(len(tree) -2, -1,-1):
            for col in range (0, len(tree[row])):
                node_full = bool(tree[row][col])
                if node_full:
                    continue
                else:    # if node is empty => update it by finding maximum children        
                    queue_update = self._max_childen(tree, row, col, queue, bp)
                    return queue_update

    def _max_childen(self, tree, row: int, col: int, queue, bp):
        # adds to the queue the tree(s) filled with the node updated with whichever child(ren) yield the most utilty for the agent making the decision.
        cardinalities = map(self.get_cardinality, self.all_decision_nodes)
        decision_cardinalities = dict(zip(self.all_decision_nodes, cardinalities)) #returns a dictionary matching each decision with its cardinality
        
        l = []
        dec_num_act = decision_cardinalities[self.reversed_acyclic_ordering[row]]  # number of possible actions for that decision
        for indx in range (col*dec_num_act, (col*dec_num_act)+dec_num_act):   # using col*dec_num_act and (col*dec_num_act)+dec_num_act so we iterate over all actions that agent is considering 
            l.append(self._get_ev(tree[row+1][indx], row, bp))  
        max_indexes = [i for i, j in enumerate(l) if j == max(l)]

        for i in range(len(max_indexes)):
            tree[row][col] = tree[row+1][(col*dec_num_act)+max_indexes[i]]
            new_tree = copy.deepcopy(tree)   
            queue.append(new_tree)
        return queue


    # def _get_ev(self, dec_list:List[int], row: int, bp):
    #     # returns the expected value of that decision for the agent making the decision
    #     dec = self.reversed_acyclic_ordering[row]   #gets the decision being made on this row
    #     agent = self._get_dec_agent(dec)      #gets the agent making that decision
    #     utils = self.utility_nodes[agent]       #gets the utility nodes for that agent

    #     h = bp.query(variables=self.all_utility_nodes, evidence=dict(zip(self.reversed_acyclic_ordering, dec_list)))  
    #     ev = 0
    #     print(f"h.values = {h.values}")
    #     for idx, prob in np.ndenumerate(h.values):
    #         print(f"idx = {idx}, prob = {prob}")
    #         print(f"agent = {agent}")
    #         if prob != 0:
    #             ev += prob*self.utility_domains[agent][idx[agent]]     

    #                 #ev += prob*self.utility_values[agent][idx[agent-1]]     #(need agent -1 because idx starts from 0, but agents starts from 1)
    #     return ev

    def _get_ev(self, dec_list:List[int], row: int, bp):
        #returns the expected value of that decision for the agent making the decision
        dec = self.reversed_acyclic_ordering[row]   #gets the decision being made on this row
        agent = self._get_dec_agent(dec)      #gets the agent making that decision
        utils = self.utility_nodes[agent]       #gets the utility nodes for that agent

        h = bp.query(variables=utils, evidence=dict(zip(self.reversed_acyclic_ordering, dec_list)))  
        ev = 0
        for idx, prob in np.ndenumerate(h.values):   
            for i in range(len(utils)): # account for each agent having multiple utilty nodes
                if prob != 0:
                    ev += prob*self.utility_domains[utils[i]][idx[i]]     

                        #ev += prob*self.utility_values[agent][idx[agent-1]]     #(need agent -1 because idx starts from 0, but agents starts from 1)
        return ev






    def _stopping_condition(self, queue):
        """stopping condition for recursive tree filling"""
        tree = queue[0]
        root_node_full = bool(tree[0][0])
        return root_node_full

    def _PSNE_finder(self):
        """this finds all pure strategy subgame perfect NE when the strategic relevance graph is acyclic
        - first initialises the maid with uniform random conditional probability distributions at every decision.
        - then fills up a queue with trees containing each solution
        - the queue will contain only one entry (tree) if there's only one pure strategy subgame perfect NE"""
        self.random_instantiation_dec_nodes()
        
        


        bp = BeliefPropagation(self)
        queue = self._instantiate_initial_tree()
        while not self._stopping_condition(queue):
            queue = self._reduce_tree_once(queue, bp)
        return queue

    def get_all_PSNE(self):
        """yields all pure strategy subgame perfect NE when the strategic relevance graph is acyclic
        !should still decide how the solutions are best displayed! """
        solutions = self._PSNE_finder()
        solution_array = []
        for tree in solutions:
            for row in range(len(tree)-1):           
                for col in tree[row]:
                    chosen_dec = tree[row][col][:row]
                    matching_of_chosen_dec = zip(self.reversed_acyclic_ordering, chosen_dec)
                    matching_of_solution = zip(self.reversed_acyclic_ordering, tree[row][col])
                    solution_array.append((list(matching_of_chosen_dec), list(matching_of_solution)))         
        return solution_array


            # can adapt how it displays the result (perhaps break into each agent's information sets)




 # -------------Methods for finding the MAID's strategic relevance graph and checking cyclicity ------------------------------------------------------------

    def _is_s_reachable(self, d1: str, d2: str):
        """ 
        
        - this method determines whether 'D2' is s-reachable from 'D1' (Koller and Milch 2001)
        
        A node D2 is s-reachable from a node D1 in a MACID M if there is some utility node U ∈ U_D
        such that if a new parent D2' were added to D2, there would be an active path in M from
        D2′ to U given Pa(D)∪{D}, where a path is active in a MAID if it is active in the same graph, viewed as a BN.

        """
        self.add_edge('temp_par', d2)
        agent = self._get_dec_agent(d1)
        agent_utilities = self.utility_nodes[agent]
        con_nodes = [d1] + self.get_parents(d1) 
        is_active_trail = any([self.is_active_trail('temp_par', u_node, con_nodes) for u_node in agent_utilities])
        self.remove_node('temp_par')
        return is_active_trail

    def strategic_rel_graph(self):
        """
        finds the strategic relevance graph of the MAID
        an edge D' -> D exists iff D' is s-reachable from D
        """
        G = nx.DiGraph()
        dec_pair_perms = list(itertools.permutations(self.all_decision_nodes, 2))
        for dec_pair in dec_pair_perms:
            if self._is_s_reachable(dec_pair[0], dec_pair[1]):
                G.add_edge(dec_pair[1], dec_pair[0])
        return G

    def draw_strategic_rel_graph(self):
        """
        draws a MACID's strategic relevance graph
        """
        rg = self.strategic_rel_graph()
        nx.draw_networkx(rg, node_size=400, arrowsize=20, edge_color='g', with_labels=True)
        plt.figure(2)
        plt.draw()
        

    def strategically_acyclic(self):
        """
        finds whether the MACID has an acyclic strategic relevance graph.
        """
        rg = self.strategic_rel_graph()
        if nx.is_directed_acyclic_graph(rg):
            return True
        else:
            return False

    def get_acyclic_topological_ordering(self):
        """
        first checks whether the strategic relevance graph is acyclic
        returns a topological ordering (which might not be unique) of the decision nodes
        """
        rg = self.strategic_rel_graph()
        if self.strategically_acyclic():
            return list(nx.topological_sort(rg))
        else:
            return 'The strategic relevance graph for this MACID is not acyclic and so \
                        no topological ordering can be immediately given. Use .. method instead.'




#----------------cyclic relevance graph methods:--------------------------------------

    def find_SCCs(self):
        """
        Uses Tarjan’s algorithm with Nuutila’s modifications
        - complexity is linear in the number of edges and nodes """
        rg = self.strategic_rel_graph()
        l = list(nx.strongly_connected_components(rg))

        
        numSCCs = nx.number_strongly_connected_components(rg)
        print(f"num = {numSCCs}")

        
    def _set_color_SCC(self, node, SCCs):
        colors = cm.rainbow(np.linspace(0, 1, len(SCCs)))
        for SCC in SCCs:         
            idx = SCCs.index(SCC)
            if node in SCC:
                col = colors[idx]
        return col   

    def draw_SCCs(self):
        """
        This shows the strategic relevance graph's SCCs
        """
        rg = self.strategic_rel_graph()
        SCCs = list(nx.strongly_connected_components(rg)) 
        layout = nx.kamada_kawai_layout(rg)
        colors = [self._set_color_SCC(node, SCCs) for node in rg.nodes]
        nx.draw_networkx(rg, pos=layout, node_size=400, arrowsize=20, edge_color='g', node_color=colors) 
        plt.draw()

    def component_graph(self):
        """
        draws and returns the component graph whose nodes are the maximal SCCs of the relevance graph
        the component graph will always be acyclic. Therefore, we can return a topological ordering.
        comp_graph.graph['mapping'] returns a dictionary matching the original nodes to the nodes in the new component (condensation) graph
        """
        rg = self.strategic_rel_graph()
        comp_graph = nx.condensation(rg)
        nx.draw_networkx(comp_graph, with_labels=True)
        plt.figure(4)
        plt.draw()
        return comp_graph

    def get_cyclic_topological_ordering(self):
        """first checks whether the strategic relevance graph is cyclic
        if it's cyclic 
        returns a topological ordering (which might not be unique) of the decision nodes
        """
        rg = self.strategic_rel_graph()
        if self.strategically_acyclic():
            return TypeError(f"Relevance graph is acyclic")
        else:
            comp_graph = self.component_graph()
            return list(nx.topological_sort(comp_graph))

        
        numSCCs = nx.number_strongly_connected_components(rg)
        print(f"num = {numSCCs}")

        
    def _set_color_SCC(self, node, SCCs):
        colors = cm.rainbow(np.linspace(0, 1, len(SCCs)))
        for SCC in SCCs:         
            if node in SCC:
                col = colors[SCCs.index(SCC)]
        return col   

# ----------- Methods for converting MACID to EFG for Gambit to solve ---------



    def _create_EFG_structure(self):
        """ Creates the EFG structure:
        1) Finds the MACID nodes needed for the EFG (this is the set {D ∪ Pa(D)})
        2) Creates an ordering of these nodes such that X_j precedes X_i in the ordering if and only if X_j
        is a descendant of X_i in the MACID.
        3) Labels each node X_i with a partial instantiation of the splits in the path to X_i in the EFG.
        """
        self.random_instantiation_dec_nodes()
    
        game_tree_nodes = self.all_decision_nodes + [parent for dec in self.all_decision_nodes for parent in self.get_parents(dec)]   # returns set {D ∪ Pa(D)}
        sorted_game_tree_nodes = [node for node in list(nx.topological_sort(self)) if node in game_tree_nodes] # sorting nodes in {D ∪ Pa(D)} into a topological ordering consistent with the graph
        numRows = len(sorted_game_tree_nodes) +1   # +1 means we have a row for the result after the final node has split
    
        cardinalities = map(self.get_cardinality, sorted_game_tree_nodes)
        node_cardinalities = dict(zip(sorted_game_tree_nodes, cardinalities))
    
        node_splits_list = list(itertools.accumulate(node_cardinalities.values(), operator.mul))  #gives number of unique paths to get to each row 
        nodes_in_each_tree_row = [1] + node_splits_list   # 1 root node for the first row

        efg_structure = defaultdict(dict)   # creates a nested dictionary
        shell = defaultdict(dict)
        splits_at_each_node_list = []
        for card in node_cardinalities.values():
            splits_at_each_node_list.append(list(range(card)))  

        efg_structure[0][0] = {}
        shell[0][0] = (0,0)
        
        for i in range(1, numRows):
            for j in range(nodes_in_each_tree_row[i]):     
                splits = list(itertools.product(*splits_at_each_node_list[0:i]))[j]  # fills in the instantiation of splits in the path to get the j'th element along row i of the tree
                efg_structure[i][j] = dict(zip(sorted_game_tree_nodes, splits))
                shell[i][j] = (i,j)
        return efg_structure, sorted_game_tree_nodes, shell


    def _add_utilities(self, tree, node_order):
        """
        adds the utilities as leaves in the EFG 
        """
        bp = BeliefPropagation(self)
        for idx, leaf in enumerate(tree[len(tree)-1].values()):    # iterate over leaves of EFG structure 
            tree['u'][idx] = self._get_leaf_utilities(leaf.values(), node_order, bp)
        return tree

    def _get_leaf_utilities(self, node_selection, node_order, bp):
        # finds leaf utilities by querying (doing propabalistic inference) on the BN described by the MACID
        # utilities = {0:np.arange(6), 1:-np.arange(6)}    # this should come from the examples
        evidences = dict(zip(node_order, node_selection))
        leaf_utilities = []
        for agent in range(len(self.agents)):
            utils = self.utility_nodes[agent]       #gets the utility nodes for that agent
            h = bp.query(variables=utils, evidence=evidences)
            ev = 0
            for idx, prob in np.ndenumerate(h.values):
                for i in range(len(utils)):
                    if prob != 0:
                        ev += prob*self.utility_domains[utils[i]][idx[i]]     #(need agent -1 because idx starts from 0, but agents starts from 1)
            leaf_utilities.append(ev)
        return leaf_utilities

    def _get_leaf_utilities(self, node_selection, node_order, bp):
        # finds leaf utilities by querying (doing propabalistic inference) on the BN described by the MACID
        evidences = dict(zip(node_order, node_selection))
        leaf_utilities = []

        for agent in range(len(self.agents)):
            utils = self.utility_nodes[agent]       #gets the utility nodes for that agent

            h = bp.query(variables=utils, evidence=evidences)
            ev = 0
            for idx, prob in np.ndenumerate(h.values):
                for i in range(len(utils)):  #accounts for each agent potentially having multiple utility nodes
                    if prob != 0:
                        ev += prob*self.utility_domains[utils[i]][idx[i]]     #(need agent -1 because idx starts from 0, but agents starts from 1)
            leaf_utilities.append(ev)
        return leaf_utilities

    def _preorder_traversal(self, shell, node_order):
        """
        returns the EFG node location ordering consistent with a prefix-order traversal of the EFG tree
        """   
        cardinalities = map(self.get_cardinality, node_order)
        node_cardinalities = dict(zip(node_order, cardinalities))
        
        stack = deque([])
        preorder = []
        preorder.append(shell[0][0])
        stack.append(shell[0][0]) 
        while len(stack)> 0:
            flag = 0   # checks wehther all child nodes have been visited
            if stack[len(stack)-1][0] == len(node_order):   # if top of stack is a leaf node, remove from stack
                stack.pop()  
            else:    # consider case when top of stack is a parent with children
                par_row = stack[len(stack)-1][0]
                par_col = stack[len(stack)-1][1]
                num_child = node_cardinalities[node_order[par_row]]
            for child in range(par_col*num_child, (par_col*num_child)+num_child): #iterate through children
                if shell[par_row+1][child] not in preorder: #as soon as an unvisited child is found, push it to stack and add to preorder
                    flag = 1
                    stack.append(shell[par_row+1][child])
                    preorder.append(shell[par_row+1][child])
                    break    # start again in while loop to explore this new child
            if flag == 0:    # if all children of a parent have been visited, pop this node from the stack.
                stack.pop()
        return preorder

    def _trim_efg(self, efg, node_order):
        """
        trims the EFG so at each efg node we only contain the instantiation of the splits made by the nodes' parents in the macid.
        This is necessary for determining the information sets.
        """    
        for row in range(1,len(node_order)):
            for node_splits in efg[row].values():
                for node in list(node_splits.keys()):
                    if node not in self.get_parents(node_order[row]):
                        del(node_splits[node])
        return efg


    def _info_sets(self, trimmed_efg, node_order):
        """returns info sets for each row of the efg (we can consider rows individually because every EFG derived from a macid will be symmetric and have a row for each macid node)
        nodes in the same EFG information set are:
        1) labelled with the same partial instantiation of splits (accoding to their parent nodes in the MACID)
        2) The same actions available to them. """
        info_sets = {}
        for row in range(1,len(node_order)):
            node_splits = list(trimmed_efg[row].values())
            comparable_list = [str(split) for split in node_splits]
            _, info_set_nums = np.unique(comparable_list, return_inverse=True)
            info_sets[row] = list(info_set_nums+1)  # in gambit info sets must start at 1
        return info_sets

    def _write_efg_file(self, macid_node_order, preorder, info_sets, efg):
        # writes EFG to a "macid_game.efg" file for GAMBIT. Can use GAMBIT's NE finders + GAMBIT's GUI for investigating game properties.
        cardinalities = map(self.get_cardinality, macid_node_order)
        node_cardinalities = dict(zip(macid_node_order, cardinalities))
        f = open("macid_game1.efg", "w")

        f.write("EFG 2 R \"Game\" { ")    #creates the necessary ".efg" header
        for i in range(len(self.agents)):
            f.write(f"\"player {i+1} \" ")
        f.write("}\n")
            
        terminal_inf_set = 0
        chance_inf_set = 0
        
        for node in preorder:     #creates the ".efg" body by iterating through nodes in a prefix-traversal ordering
            if node[0] < len(macid_node_order):
                macid_node = macid_node_order[node[0]]

                if self.get_node_type(macid_node)  == 'c':  #chance node
                    bp = BeliefPropagation(self)
                    probs = bp.query(variables=[macid_node], evidence=efg[node[0]][node[1]]).values
                    chance_inf_set += 1
                    f.write(f"c \"\" {chance_inf_set} \"\" ")
                    f.write("{ ")
                    for action in range(node_cardinalities[macid_node]):
                        f.write(f"\"{action}\" {probs[action]} ")
                    f.write("} 0\n")

                elif self.get_node_type(macid_node)  == 'p':  #player node
                    agent = self._get_dec_agent(macid_node)
                    f.write(f"p \"\" {agent+1} ")
                    f.write(f"{info_sets[node[0]][node[1]]+100*node[0]} \"\" ")   #messy way of handling the fact that a player could make multiple decisions => need unique info sets for each decision.
                    f.write("{ ")
                    for action in range(node_cardinalities[macid_node]):
                        f.write(f"\"{action}\" ")
                    f.write("} 0\n")

            elif node[0] == len(macid_node_order):  #terminal nodes
                terminal_inf_set +=1
                f.write(f"t \"\" {terminal_inf_set} \"\" ")
                f.write("{ ")
                for util in efg['u'][node[1]]:
                    f.write(f"{util}, ")
                f.write("}\n")    
        
        f.close()


    def MACID_to_Gambit_file(self):
        """
        Converts MACID to a ".efg" file for use with GAMBIT.
        """
        efg_structure, node_order, shell = self._create_EFG_structure()   
        efg_with_utilities = self._add_utilities(efg_structure, node_order)
        preorder = self._preorder_traversal(shell, node_order)
        trimmed_efg = self._trim_efg(efg_with_utilities, node_order)  
        info_sets = self._info_sets(trimmed_efg, node_order)
        self._write_efg_file(node_order, preorder, info_sets, efg_with_utilities)
        print("\nGambit .efg file has been created from the macid")
        return True