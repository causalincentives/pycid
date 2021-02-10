#Licensed to the Apache Software Foundation (ASF) under one or more contributor license
#agreements; and to You under the Apache License, Version 2.0.

import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.factors.discrete import TabularCPD
import logging
from typing import List, Tuple, Dict, Any, Callable, Union
#import numpy.typing as npt
import itertools
from pgmpy.inference import BeliefPropagation
import networkx as nx
from core.cpd import UniformRandomCPD, FunctionCPD, DecisionDomain
import matplotlib.pyplot as plt
import operator
from collections import defaultdict
from collections import deque
import copy
import matplotlib.cm as cm
from core.get_paths import get_motifs, get_motif
from core.macid_base import MACIDBase
import copy


class MACID(MACIDBase):
    def __init__(self, edges: List[Tuple[Union[str, int], str]],
                 node_types: Dict[Union[str, int], Dict]):
        super().__init__(edges, node_types)

    def copy_without_cpds(self):
        return MACID(self.edges(),
                      {agent: {'D': self.decision_nodes_agent[agent],
                                  'U': self.utility_nodes_agent[agent]}
                          for agent in self.agents})

    def _get_color(self, node: str) -> np.ndarray:
        """
        Assign a unique colour with each new agent's decision and utility nodes
        """
        colors = cm.rainbow(np.linspace(0, 1, len(self.agents)))
        if node in self.all_decision_nodes or node in self.all_utility_nodes:
            return colors[[self.agents.index(self.whose_node[node])]]
        else:
            return 'lightgray'  # chance node

    def get_SCCs(self) -> List[set]:
        """
        Return a list with the maximal strongly connected components of the MACID's
        full strategic relevance graph.
        Uses Tarjan’s algorithm with Nuutila’s modifications
        - complexity is linear in the number of edges and nodes """
        rg = self.strategic_rel_graph()
        return list(nx.strongly_connected_components(rg))
        
    def _set_color_SCC(self, node: str, SCCs) -> np.ndarray:
        "Assign a unique color to the set of nodes in each SCC."
        colors = cm.rainbow(np.linspace(0, 1, len(SCCs)))
        for SCC in SCCs:
            idx = SCCs.index(SCC)
            if node in SCC:
                col = colors[idx]
                return col

    def draw_SCCs(self) -> None:
        """
        Show the SCCs for the MACID's full strategic relevance graph
        """
        rg = self.strategic_rel_graph()
        SCCs = list(nx.strongly_connected_components(rg))
        layout = nx.kamada_kawai_layout(rg)
        colors = [self._set_color_SCC(node, SCCs) for node in rg.nodes]
        nx.draw_networkx(rg, pos=layout, node_size=400, arrowsize=20, edge_color='g', node_color=colors)
        plt.show()

    def condensed_relevance_graph(self) -> nx.DiGraph:
        """
        Return the condensed_relevance graph whose nodes are the maximal SCCs of the full relevance graph of the original MAID.
        - The condensed_relevance graph will always be acyclic. Therefore, we can return a topological ordering.
        """
        rg = self.strategic_rel_graph()
        con_rel = nx.condensation(rg)
        return con_rel

    def draw_condensed_relevance_graph(self) -> None:
        con_rel = self.condensed_relevance_graph()
        nx.draw_networkx(con_rel, with_labels=True)
        plt.show()

    def all_maid_subgames(self):
        """ 
        Return a list giving the set of decision nodes in each MAID subgame of the original MAID.
        """
        con_rel = self.condensed_relevance_graph()
        # con_rel.graph['mapping'] returns a dictionary matching the original relevance graph's
        # decision nodes with the SCCs they are in in the condensed relevance graph
        dec_scc_mapping = con_rel.graph['mapping']
        # invert the dec_scc_mapping dictionary which contains non-unique values:
        scc_dec_mapping = {}
        for k, v in dec_scc_mapping.items():
            scc_dec_mapping[v] = scc_dec_mapping.get(v, []) + [k]
        
        con_rel_sccs = con_rel.nodes  # the nodes of the condensed relevance graph are the maximal SCCs of the MA(C)ID
        powerset = list(itertools.chain.from_iterable(itertools.combinations(con_rel_sccs, r) 
                                                      for r in range(1, len(con_rel_sccs)+1)))
        con_rel_subgames = copy.deepcopy(powerset)
        for subset in powerset:
            for node in subset:
                if not nx.descendants(con_rel, node).issubset(subset) and subset in con_rel_subgames:
                    con_rel_subgames.remove(subset)

        dec_subgames = [[scc_dec_mapping[scc] for scc in con_rel_subgame] for con_rel_subgame in con_rel_subgames]

        return [set(itertools.chain.from_iterable(i)) for i in dec_subgames]


#### --------- TODO: All below methods are currently untested as Im going to try and align them with Tom's neater way of doing things!

    def _get_scc_topological_ordering(self):
        """first checks whether the strategic relevance graph is cyclic
        if it's cyclic
        Returns a topological ordering (which might not be unique) of the decision nodes
        """  
        con_rel_graph = self.condensed_relevance_graph()
        return list(nx.topological_sort(con_rel_graph))


    def random_instantiation_dec_nodes(self) -> None:
        """
        TODO: Change to using Tom's random instantiation
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


    def _instantiate_initial_tree(self) -> List[defaultdict]:
        #creates a tree (a nested dictionary) which we use to fill up with the subgame perfect NE of each sub-tree.
        cardinalities = map(self.get_cardinality, self.all_decision_nodes)
        decision_cardinalities = dict(zip(self.all_decision_nodes, cardinalities)) 

        #gives number of pure strategies for each decision node (ie taking into account prev decisions)
        action_space_list = list(itertools.accumulate(decision_cardinalities.values(), operator.mul))  
        cols_in_each_tree_row = [1] + action_space_list

        actions_for_dec_list = []
        for card in decision_cardinalities.values():
            actions_for_dec_list.append(list(range(card))) # appending the range of actions each decion can make as a list
        # creates entry for final row of decision array (cartesian product of actions_seq_list))
        final_row_actions = list(itertools.product(*actions_for_dec_list))     

        tree_initial = defaultdict(dict)   # creates a nested dictionary
        for i in range(0, self.numDecisions+1):
            for j in range(cols_in_each_tree_row[i]):   # initialises entire decision/action array with empty tuples.
                tree_initial[i][j] = ()

        for i in range(cols_in_each_tree_row[-1]):
            tree_initial[self.numDecisions][i] = final_row_actions[i]

        trees_queue = [tree_initial]  # list of all possible decision trees
        return trees_queue


    def _reduce_tree_once(self, queue: List[str], bp):
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
        reversed_acyclic_ordering = list(reversed(self.get_valid_acyclic_dec_node_ordering()))     
        temp_list = []
        dec_num_act = decision_cardinalities[reversed_acyclic_ordering[row]]  # number of possible actions for that decision
        for indx in range(col*dec_num_act, (col*dec_num_act)+dec_num_act):   # using col*dec_num_act and (col*dec_num_act)+dec_num_act so we iterate over all actions that agent is considering
            temp_list.append(self._get_ev(tree[row+1][indx], row, bp))
        max_indexes = [i for i, j in enumerate(temp_list) if j == max(l)]

        for i in range(len(max_indexes)):
            tree[row][col] = tree[row+1][(col*dec_num_act)+max_indexes[i]]
            new_tree = copy.deepcopy(tree)
            queue.append(new_tree)
        return queue


    def _get_ev(self, dec_list: List[int], row: int, bp) -> float:
        # TODO: use EU in MACIDBASE!
        #returns the expected value of that decision for the agent making the decision
        dec = self.reversed_acyclic_ordering[row]   #gets the decision being made on this row
        agent = self._get_dec_agent(dec)      #gets the agent making that decision
        utils = self.utility_nodes[agent]       #gets the utility nodes for that agent
        reversed_acyclic_ordering = list(reversed(self.get_valid_acyclic_dec_node_ordering()))     

        h = bp.query(variables=utils, evidence=dict(zip(reversed_acyclic_ordering, dec_list)))
        ev = 0
        for idx, prob in np.ndenumerate(h.values):
            for i in range(len(utils)): # account for each agent having multiple utilty nodes
                if prob != 0:
                    ev += prob*self.utility_domains[utils[i]][idx[i]]

                        #ev += prob*self.utility_values[agent][idx[agent-1]]     #(need agent -1 because idx starts from 0, but agents starts from 1)
        return ev

    def _stopping_condition(self, queue) -> bool:
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

    def get_all_PSNE(self) -> List[List[str]]:
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








