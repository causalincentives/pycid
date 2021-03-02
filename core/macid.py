# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.
from __future__ import annotations
# from _typeshed import NoneType
from core.cpd import FunctionCPD
import numpy as np
from typing import Any, List, Tuple, Dict, Union
# import numpy.typing as npt
import itertools
from pgmpy.inference import BeliefPropagation  # type: ignore
import networkx as nx
import matplotlib.pyplot as plt
import operator
from collections import defaultdict
import copy
import matplotlib.cm as cm
from core.macid_base import MACIDBase
from core.relevance_graph import RelevanceGraph, CondensedRelevanceGraph

class MACID(MACIDBase):

    def __init__(self, edges: List[Tuple[Union[str, int], str]],
                 node_types: Dict[Union[str, int], Dict]):
        super().__init__(edges, node_types)

    def get_sccs(self) -> List[set]:
        """
        Return a list with the maximal strongly connected components of the MACID's
        full strategic relevance graph.
        Uses Tarjan’s algorithm with Nuutila’s modifications
        - complexity is linear in the number of edges and nodes """
        rg = RelevanceGraph(self)
        return list(nx.strongly_connected_components(rg))

    def _set_color_scc(self, node: str, sccs: List[Any]) -> np.ndarray:
        "Assign a unique color to the set of nodes in each SCC."
        colors = cm.rainbow(np.linspace(0, 1, len(sccs)))
        scc_index = 0
        for idx, scc in enumerate(sccs):
            if node in scc:
                scc_index = idx
                break
        return colors[scc_index]  # type: ignore

    def draw_sccs(self) -> None:
        """
        Show the SCCs for the MACID's full strategic relevance graph
        """
        rg = RelevanceGraph(self)
        sccs = list(nx.strongly_connected_components(rg))
        layout = nx.kamada_kawai_layout(rg)
        colors = [self._set_color_scc(node, sccs) for node in rg.nodes]
        nx.draw_networkx(rg, pos=layout, node_size=400, arrowsize=20, edge_color='g', node_color=colors)
        plt.show()

    def all_maid_subgames(self) -> List[set]:
        """
        Return a list giving the set of decision nodes in each MAID subgame of the original MAID.
        """
        con_rel = CondensedRelevanceGraph(self)
        # con_rel.graph['mapping'] returns a dictionary matching the original relevance graph's
        # decision nodes with the sccs they are in in the condensed relevance graph
        dec_scc_mapping = con_rel.graph['mapping']
        # invert the dec_scc_mapping dictionary which contains non-unique values:
        scc_dec_mapping: Dict[int, List[str]] = {}
        for k, v in dec_scc_mapping.items():
            scc_dec_mapping[v] = scc_dec_mapping.get(v, []) + [k]

        con_rel_sccs = con_rel.nodes  # the nodes of the condensed relevance graph are the maximal sccs of the MA(C)ID
        powerset = list(itertools.chain.from_iterable(itertools.combinations(con_rel_sccs, r)
                                                      for r in range(1, len(con_rel_sccs) + 1)))
        con_rel_subgames = copy.deepcopy(powerset)
        for subset in powerset:
            for node in subset:
                if not nx.descendants(con_rel, node).issubset(subset) and subset in con_rel_subgames:
                    con_rel_subgames.remove(subset)

        dec_subgames = [[scc_dec_mapping[scc] for scc in con_rel_subgame] for con_rel_subgame in con_rel_subgames]

        return [set(itertools.chain.from_iterable(i)) for i in dec_subgames]

    def _get_scc_topological_ordering(self) -> List[int]:
        """
        Returns a topological ordering (which might not be unique) of the SCCs
        """
        con_rel = CondensedRelevanceGraph(self)
        return list(nx.topological_sort(con_rel))

    # def get_all_pure_spe(self) -> List[List[Tuple[Any, List[Tuple[Any, Any]], Any]]]:
    #     """Return all pure policy subgame perfect NE in the MAID when the relevance graph is acyclic"""
    #     if not RelevanceGraph(self).is_acyclic():
    #         raise Exception('The relevance graph for this (MA)CID is not acyclic and so \
    #                     this method cannot be used.')

    #     solutions = self._pure_spe_finder()
    #     spe_arrays = [self._create_spe_array(tree) for tree in solutions]
    #     return spe_arrays

    # def _pure_spe_finder(self) -> List[defaultdict]:
    #     """this finds all pure strategy subgame perfect NE when the strategic relevance graph is acyclic
    #     - first initialises the maid with uniform random conditional probability distributions at every decision.
    #     - then fills up a queue with trees containing each solution
    #     - the queue will contain only one entry (tree) if there's only one pure strategy subgame perfect NE"""
    #     for dec in self.all_decision_nodes:
    #         self.impute_random_decision(dec)  # impute random fully mixed policy to all decision nodes.

    #     bp = BeliefPropagation(self)
    #     print(type(bp))
    #     queue = self._instantiate_initial_tree()

    #     while not self._stopping_condition(queue):
    #         queue = self._reduce_tree_once(queue, bp)
    #     return queue

    # def _stopping_condition(self, queue: List[defaultdict]) -> bool:
    #     """stopping condition for recursive tree filling"""
    #     tree = queue[0]
    #     root_node_full = bool(tree[0][0])
    #     return root_node_full

    # def _create_spe_array(self, tree: defaultdict) -> List[Tuple[Any, List[Tuple[Any, Any]], Any]]:
    #     """Return the subgame perfect equilibirium in a nested list form
    #     Example output: [('D1', [], 0), ('D2', [('D1, 0)], 1), ('D2', [('D1, 1)], 0)]

    #     The first argument of each triple gives the decision node, the second argument gives the
    #     decision context being conditioned on, and the third gives the decision node's action prescribed
    #     by the pure SPE.
    #     """
    #     dec_list = self.get_valid_order()
    #     decision_cardinalities = [self.get_cardinality(dec) for dec in dec_list]

    #     spe_array = []
    #     for row in range(len(tree) - 1):
    #         cols = tree[row].keys()
    #         for i in cols:
    #             divisor = 1
    #             action_values = []
    #             for j, dec_card in reversed(list(enumerate(decision_cardinalities[:row]))):
    #                 action_values.append((i // divisor) % decision_cardinalities[j])
    #                 divisor *= dec_card
    #             decision_context_values = list(reversed(action_values))
    #             decision_context = list(zip(dec_list[:row], decision_context_values))
    #             spe_array.append((dec_list[row], decision_context, tree[row][i][row]))

    #     return spe_array

    # def _instantiate_initial_tree(self) -> List[defaultdict]:
    #     """Create a tree (a nested dictionary) used for SPE backward induction."""
    #     dec_list = self.get_valid_order()
    #     decision_cardinalities = [self.get_cardinality(dec) for dec in dec_list]
    #     # find number of pure strategies for each decision node (taking into account prev decisions)
    #     action_space_list = list(itertools.accumulate(decision_cardinalities, operator.mul))
    #     cols_in_each_tree_row = [1] + action_space_list

    #     actions_for_dec_list = []
    #     for card in decision_cardinalities:
    #         actions_for_dec_list.append(list(range(card)))     # append the range of actions each decion can make
    #     # create entry for final row of decision array
    #     final_row_actions = list(itertools.product(*actions_for_dec_list))

    #     tree_initial: defaultdict = defaultdict(dict)   # creates a nested dictionary
    #     for i in range(0, len(self.all_decision_nodes) + 1):
    #         for j in range(cols_in_each_tree_row[i]):     # initialise tree with empty tuples.
    #             tree_initial[i][j] = ()

    #     for i in range(cols_in_each_tree_row[-1]):
    #         tree_initial[len(self.all_decision_nodes)][i] = final_row_actions[i]

    #     trees_queue = [tree_initial]  # list of all possible decision trees
    #     return trees_queue

    # def _reduce_tree_once(self, queue: List[defaultdict], bp: BeliefPropagation) -> List[defaultdict]:
    #     """Find first node in tree not yet evaluated using prefix-traversal
    #     and then update the tree by evaluating this node - apply this repeatedly
    #     until tree is full"""
    #     tree = queue.pop(0)
    #     for row in range(len(tree) - 2, -1, -1):
    #         for col in range(0, len(tree[row])):
    #             node_full = bool(tree[row][col])
    #             if node_full:
    #                 continue
    #             else:    # if node is empty => update it by finding maximum children
    #                 queue_update = self._max_childen(tree, row, col, queue, bp)
    #                 return queue_update
    #     return queue  # shouldn't ve called

    # def _max_childen(self, tree: defaultdict, row: int, col: int, queue: List[defaultdict],
    #                  bp: BeliefPropagation) -> List[defaultdict]:
    #     """ Add to the queue the tree(s) filled with the node updated with whichever
    #     child(ren) yield the most utilty for the agent making the decision."""
    #     dec_list = self.get_valid_order()
    #     children_ev = []
    #     dec_num_act = self.get_cardinality(dec_list[row])  # num actions (children in the tree) of this decision

    #     # using col*dec_num_act and (col*dec_num_act)+dec_num_act so we iterate over all of the
    #     # agent's considered actions (children in the tree)
    #     for indx in range(col * dec_num_act, (col * dec_num_act) + dec_num_act):
    #         children_ev.append(self._get_ev(tree[row + 1][indx], row, bp))
    #     max_indexes = [i for i, j in enumerate(children_ev) if j == max(children_ev)]

    #     for i in range(len(max_indexes)):
    #         tree[row][col] = tree[row + 1][(col * dec_num_act) + max_indexes[i]]
    #         new_tree = copy.deepcopy(tree)
    #         queue.append(new_tree)
    #     return queue

    # def _get_ev(self, dec_instantiation: Tuple[int], row: int, bp: BeliefPropagation) -> float:
    #     """Return the expected value of a certain decision node instantiation
    #     for the agent making the decision"""
    #     dec_list = self.get_valid_order()
    #     dec = dec_list[row]

    #     agent = self.whose_node[dec]      # gets the agent making that decision
    #     utils = self.utility_nodes_agent[agent]       # gets the utility nodes for that agent
    #     factor = bp.query(variables=utils, evidence=dict(zip(dec_list, dec_instantiation)))

    #     ev = 0
    #     for idx, prob in np.ndenumerate(factor.values):
    #         for i in range(len(utils)):  # account for each agent having multiple utilty nodes
    #             if prob != 0:
    #                 ev += prob * idx[i]
    #     return ev

    def copy_without_cpds(self) -> MACID:
        """copy the MACID structure"""
        return MACID(self.edges(),
                     {agent: {'D': list(self.decision_nodes_agent[agent]),
                              'U': list(self.utility_nodes_agent[agent])}
                     for agent in self.agents})

    def _get_color(self, node: str) -> Union[str, np.ndarray]:
        """
        Assign a unique colour with each new agent's decision and utility nodes
        """
        colors = cm.rainbow(np.linspace(0, 1, len(self.agents)))
        if node in self.all_decision_nodes or node in self.all_utility_nodes:
            return colors[[self.agents.index(self.whose_node[node])]]  # type: ignore
        else:
            return 'lightgray'  # chance node

    def get_all_pure_ne(self) -> List[List[FunctionCPD]]:
        """
        Return a list of all pure Nash equilbiria in the MACID. 
        - Each NE comes as a list of FunctionCPDs for each decision node in the MACID.
        """
        return self.get_all_pure_ne_in_sg()
    
    
    # def get_all_pure_ne(self) -> List[List[FunctionCPD]]:
    # TODO: Keeping here in case Tom thinks we should keep this direct method for computing all
    # pure NE in a MACID.
    #     """
    #     Return a list of all pure Nash equilbiria - where each NE comes as a
    #     list of each decision node's corresponding FunctionCPD.
    #     """
    #     pure_ne = []

    #     def agent_pure_policies(agent: Union[str, int]) -> List[List[FunctionCPD]]:
    #         possible_dec_rules = list(map(self.possible_pure_decision_rules, self.decision_nodes_agent[agent]))
    #         return list(itertools.product(*possible_dec_rules))

    #     all_agent_pure_policies = {agent: agent_pure_policies(agent) for agent in self.agents}
    #     all_dec_decision_rules = list(map(self.possible_pure_decision_rules, self.all_decision_nodes))
    #     all_joint_policy_profiles = list(itertools.product(*all_dec_decision_rules))

    #     for jp in all_joint_policy_profiles:
    #         found_ne = True
    #         for a in self.agents:
    #             self.add_cpds(*jp)
    #             eu_jp_agent_a = self.expected_utility({}, agent=a)
    #             for agent_policy in all_agent_pure_policies[a]:
    #                 self.add_cpds(*agent_policy)
    #                 eu_deviation_agent_a = self.expected_utility({}, agent=a)
    #                 if eu_deviation_agent_a > eu_jp_agent_a:
    #                     found_ne = False
    #         if found_ne:
    #             pure_ne.append(jp)
    #     return pure_ne


    # def get_all_pure_ne2(self, decisions_in_sg: List[str] = None) -> List[List[FunctionCPD]]:
    #     """
    #     Return a list of all pure Nash equilbiria
    #     """
    #     if not decisions_in_sg:
    #         decisions_in_sg = self.all_decision_nodes

    #     agents_in_sg = list({self.whose_node[dec] for dec in decisions_in_sg})

    #     pure_ne_in_sg = []

    #     def agent_pure_policies(agent: Union[str, int]) -> List[List[FunctionCPD]]:
    #         agent_decs_in_sg = [dec for dec in self.decision_nodes_agent[agent] if dec in decisions_in_sg]
    #         possible_dec_rules = list(map(self.possible_pure_decision_rules, agent_decs_in_sg))
    #         return list(itertools.product(*possible_dec_rules))

    #     all_agent_pure_policies_in_sg = {agent: agent_pure_policies(agent) for agent in agents_in_sg}
    #     all_dec_decision_rules = list(map(self.possible_pure_decision_rules, decisions_in_sg))
    #     all_joint_policy_profiles_in_sg = list(itertools.product(*all_dec_decision_rules))

    #     decs_not_in_sg = [dec for dec in self.all_decision_nodes if dec not in decisions_in_sg]

        

    #     for jp in all_joint_policy_profiles_in_sg:
    #         found_ne = True
    #         for a in agents_in_sg:
    #             self.add_cpds(*jp)
    #             for d in decs_not_in_sg: # to create a fullly mixed joint policy profile
    #                 print(f"{d} not in subgame")
    #                 self.impute_random_decision(d)

    #             jp_with_ne_in_sg = [self.get_cpds(d) for d in self.all_decision_nodes]

    #             eu_jp_agent_a = self.expected_utility({}, agent=a)
    #             for agent_policy in all_agent_pure_policies_in_sg[a]:
    #                 self.add_cpds(*agent_policy)
    #                 eu_deviation_agent_a = self.expected_utility({}, agent=a)
    #                 if eu_deviation_agent_a > eu_jp_agent_a:
    #                     found_ne = False
    #         if found_ne:
                
    #             # pure_ne_in_sg.append(jp_with_ne_in_sg)


    #             pure_ne_in_sg.append(jp)
    #     return pure_ne_in_sg
    

    def get_all_pure_ne_in_sg(self, decisions_in_sg: List[str] = None, partial_policy_profile: List[FunctionCPD] = None) -> List[List[FunctionCPD]]:
        """
        Return a list of all pure Nash equilbiria in a MACID subgame given some partial_policy_profile over 
        some of the MACID's decision nodes. 
        - If decisions_in_sg is not specified, this method finds all pure NE in the full MACID.
        - If a partial policy is specified, the decison rules of decision nodes specified by the partial policy 
        remain unchanged.
        TODO: Check that the decisions in decisions_in_sg actually make up a subgame
        """
        if not decisions_in_sg:
            decisions_in_sg = self.all_decision_nodes

        for dec in decisions_in_sg:
            if dec not in self.all_decision_nodes:
                raise Exception(f"The node {dec} is not a decision node in the (MACID")


        agents_in_sg = list({self.whose_node[dec] for dec in decisions_in_sg})
        pure_ne_in_sg = []

        # Find all of an agent's pure policies in this subgame.
        def agent_pure_policies(agent: Union[str, int]) -> List[List[FunctionCPD]]:
            agent_decs_in_sg = [dec for dec in self.decision_nodes_agent[agent] if dec in decisions_in_sg]
            possible_dec_rules = list(map(self.possible_pure_decision_rules, agent_decs_in_sg))
            return list(itertools.product(*possible_dec_rules))

        all_agent_pure_policies_in_sg = {agent: agent_pure_policies(agent) for agent in agents_in_sg}
        all_dec_decision_rules = list(map(self.possible_pure_decision_rules, decisions_in_sg))
        all_joint_policy_profiles_in_sg = list(itertools.product(*all_dec_decision_rules))
        decs_not_in_sg = [dec for dec in self.all_decision_nodes if dec not in decisions_in_sg]

        # if a partial policy profile is input, those decision rules should not change
        if partial_policy_profile:
            pp = self.partial_policy_assignment(partial_policy_profile)     
            decs_already_optimised = [k for k, v in pp.items() if v != None]
            decs_to_be_randomised = [dec for dec in decs_not_in_sg if dec not in decs_already_optimised]
        else:
            decs_already_optimised = None
            decs_to_be_randomised = decs_not_in_sg
        
        for pp in all_joint_policy_profiles_in_sg:
            found_ne = True
            for a in agents_in_sg:
                
                # create a fullly mixed joint policy profile: 
                self.add_cpds(*pp)
                if partial_policy_profile:
                    self.add_cpds(*partial_policy_profile)
                for d in decs_to_be_randomised:                   
                    self.impute_random_decision(d)

                # agent a's expected utility according to this subgame policy profile
                eu_pp_agent_a = self.expected_utility({}, agent=a)
                for agent_policy in all_agent_pure_policies_in_sg[a]:
                    self.add_cpds(*agent_policy)

                    # agent a's expected utility if they deviate
                    eu_deviation_agent_a = self.expected_utility({}, agent=a)
                    if eu_deviation_agent_a > eu_pp_agent_a:
                        found_ne = False
            if found_ne:
                pure_ne_in_sg.append(pp)
        
        return pure_ne_in_sg

    # def joint_policy_assignment(self, joint_policy: List[FunctionCPD]) -> Dict:
    #     """Return dictionary with the joint policy assigned - ie a decision rule 
    #     to each of the MACIM's decision nodes."""
    #     new_macid = self.copy() 
    #     new_macid.add_cpds(*joint_policy)
    #     return {d: new_macid.get_cpds(d) for d in new_macid.all_decision_nodes}
        
    # def partial_policy_assignment(self, partial_policy: List[FunctionCPD]) -> Dict:
    #     """Return dictionary with the joint policy assigned - ie a decision rule 
    #     to each of the MACIM's decision nodes."""
    #     new_macid = self.copy_without_cpds() 
    #     new_macid.add_cpds(*partial_policy)
    #     return {d: new_macid.get_cpds(d) for d in new_macid.all_decision_nodes}
          
    def policy_profile_assignment(self, partial_policy: List[FunctionCPD]) -> Dict:
        """Return dictionary with the joint or partial policy profile assigned - 
        ie a decision rule for each of the MACIM's decision nodes."""
        new_macid = self.copy_without_cpds() 
        new_macid.add_cpds(*partial_policy)
        return {d: new_macid.get_cpds(d) for d in new_macid.all_decision_nodes}
    
    def get_all_pure_spe(self) -> List[List[FunctionCPD]]:
        spes: List[List[FunctionCPD]] = [[]]
        crg = CondensedRelevanceGraph(self)
        dec_scc_mapping = crg.graph['mapping']
        scc_dec_mapping = {}
        # invert the dictionary to match each scc with the decision nodes in it
        for k, v in dec_scc_mapping.items():
            scc_dec_mapping[v] = scc_dec_mapping.get(v, []) + [k]
        
        # backwards induction over the sccs in the condensed relevance graph (handling tie-breaks)
        for scc in reversed(list(nx.topological_sort(crg))):
            extended_spes = []
            dec_nodes_to_be_optimised = scc_dec_mapping[scc]
            for partial_profile in spes:
                all_ne_in_sg = self.get_all_pure_ne_in_sg(dec_nodes_to_be_optimised, partial_profile)
                for ne in all_ne_in_sg:
                    extended_spes.append(partial_profile + list(ne))
            spes = extended_spes
        return spes

