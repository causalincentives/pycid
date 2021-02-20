# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.
from __future__ import annotations
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


class MACID(MACIDBase):
    def __init__(self, edges: List[Tuple[Union[str, int], str]],
                 node_types: Dict[Union[str, int], Dict]):
        super().__init__(edges, node_types)

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

    def get_sccs(self) -> List[set]:
        """
        Return a list with the maximal strongly connected components of the MACID's
        full strategic relevance graph.
        Uses Tarjan’s algorithm with Nuutila’s modifications
        - complexity is linear in the number of edges and nodes """
        rg = self.relevance_graph()
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
        rg = self.relevance_graph()
        sccs = list(nx.strongly_connected_components(rg))
        layout = nx.kamada_kawai_layout(rg)
        colors = [self._set_color_scc(node, sccs) for node in rg.nodes]
        nx.draw_networkx(rg, pos=layout, node_size=400, arrowsize=20, edge_color='g', node_color=colors)
        plt.show()

    def condensed_relevance_graph(self) -> nx.DiGraph:
        """
        Return the condensed_relevance graph whose nodes are the maximal sccs of the full relevance graph
        of the original MAID.
        - The condensed_relevance graph will always be acyclic. Therefore, we can return a topological ordering.
        """
        rg = self.relevance_graph()
        con_rel = nx.condensation(rg)
        return con_rel

    def draw_condensed_relevance_graph(self) -> None:
        con_rel = self.condensed_relevance_graph()
        nx.draw_networkx(con_rel, with_labels=True)
        plt.show()

    def all_maid_subgames(self) -> List[set]:
        """
        Return a list giving the set of decision nodes in each MAID subgame of the original MAID.
        """
        con_rel = self.condensed_relevance_graph()
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
        con_rel_graph = self.condensed_relevance_graph()
        return list(nx.topological_sort(con_rel_graph))

    def get_all_pure_spe(self) -> List[List[Tuple[Any, List[Tuple[Any, Any]], Any]]]:
        """Return all pure policy subgame perfect NE in the MAID when the relevance graph is acyclic"""
        if not self.is_full_relevance_graph_acyclic():
            raise Exception('The relevance graph for this (MA)CID is not acyclic and so \
                        this method cannot be used.')

        solutions = self._pure_spe_finder()
        spe_arrays = [self._create_spe_array(tree) for tree in solutions]
        return spe_arrays

    def _pure_spe_finder(self) -> List[defaultdict]:
        """this finds all pure strategy subgame perfect NE when the strategic relevance graph is acyclic
        - first initialises the maid with uniform random conditional probability distributions at every decision.
        - then fills up a queue with trees containing each solution
        - the queue will contain only one entry (tree) if there's only one pure strategy subgame perfect NE"""
        for dec in self.all_decision_nodes:
            self.impute_random_decision(dec)  # impute random fully mixed policy to all decision nodes.

        bp = BeliefPropagation(self)
        print(type(bp))
        queue = self._instantiate_initial_tree()

        while not self._stopping_condition(queue):
            queue = self._reduce_tree_once(queue, bp)
        return queue

    def _stopping_condition(self, queue: List[defaultdict]) -> bool:
        """stopping condition for recursive tree filling"""
        tree = queue[0]
        root_node_full = bool(tree[0][0])
        return root_node_full

    def _create_spe_array(self, tree: defaultdict) -> List[Tuple[Any, List[Tuple[Any, Any]], Any]]:
        """Return the subgame perfect equilibirium in a nested list form
        Example output: [('D1', [], 0), ('D2', [('D1, 0)], 1), ('D2', [('D1, 1)], 0)]

        The first argument of each triple gives the decision node, the second argument gives the
        decision context being conditioned on, and the third gives the decision node's action prescribed
        by the pure SPE.
        """
        macid = self.copy_without_cpds()
        dec_list = list(nx.topological_sort(macid.relevance_graph()))
        decision_cardinalities = [self.get_cardinality(dec) for dec in dec_list]

        spe_array = []
        for row in range(len(tree) - 1):
            cols = tree[row].keys()
            for i in cols:
                divisor = 1
                action_values = []
                for j, dec_card in reversed(list(enumerate(decision_cardinalities[:row]))):
                    action_values.append((i // divisor) % decision_cardinalities[j])
                    divisor *= dec_card
                decision_context_values = list(reversed(action_values))
                decision_context = list(zip(dec_list[:row], decision_context_values))
                spe_array.append((dec_list[row], decision_context, tree[row][i][row]))

        return spe_array

    def _instantiate_initial_tree(self) -> List[defaultdict]:
        """Create a tree (a nested dictionary) used for SPE backward induction."""
        macid = self.copy_without_cpds()
        dec_list = list(nx.topological_sort(macid.relevance_graph()))
        decision_cardinalities = [self.get_cardinality(dec) for dec in dec_list]
        # find number of pure strategies for each decision node (taking into account prev decisions)
        action_space_list = list(itertools.accumulate(decision_cardinalities, operator.mul))
        cols_in_each_tree_row = [1] + action_space_list

        actions_for_dec_list = []
        for card in decision_cardinalities:
            actions_for_dec_list.append(list(range(card)))     # append the range of actions each decion can make
        # create entry for final row of decision array
        final_row_actions = list(itertools.product(*actions_for_dec_list))

        tree_initial: defaultdict = defaultdict(dict)   # creates a nested dictionary
        for i in range(0, len(self.all_decision_nodes) + 1):
            for j in range(cols_in_each_tree_row[i]):     # initialise tree with empty tuples.
                tree_initial[i][j] = ()

        for i in range(cols_in_each_tree_row[-1]):
            tree_initial[len(self.all_decision_nodes)][i] = final_row_actions[i]

        trees_queue = [tree_initial]  # list of all possible decision trees
        return trees_queue

    def _reduce_tree_once(self, queue: List[defaultdict], bp: BeliefPropagation) -> List[defaultdict]:
        """Find first node in tree not yet evaluated using prefix-traversal
        and then update the tree by evaluating this node - apply this repeatedly
        until tree is full"""
        tree = queue.pop(0)
        for row in range(len(tree) - 2, -1, -1):
            for col in range(0, len(tree[row])):
                node_full = bool(tree[row][col])
                if node_full:
                    continue
                else:    # if node is empty => update it by finding maximum children
                    queue_update = self._max_childen(tree, row, col, queue, bp)
                    return queue_update
        return queue  # shouldn't ve called

    def _max_childen(self, tree: defaultdict, row: int, col: int, queue: List[defaultdict],
                     bp: BeliefPropagation) -> List[defaultdict]:
        """ Add to the queue the tree(s) filled with the node updated with whichever
        child(ren) yield the most utilty for the agent making the decision."""
        macid = self.copy_without_cpds()
        dec_list = list(nx.topological_sort(macid.relevance_graph()))
        children_ev = []
        dec_num_act = self.get_cardinality(dec_list[row])  # num actions (children in the tree) of this decision

        # using col*dec_num_act and (col*dec_num_act)+dec_num_act so we iterate over all of the
        # agent's considered actions (children in the tree)
        for indx in range(col * dec_num_act, (col * dec_num_act) + dec_num_act):
            children_ev.append(self._get_ev(tree[row + 1][indx], row, bp))
        max_indexes = [i for i, j in enumerate(children_ev) if j == max(children_ev)]

        for i in range(len(max_indexes)):
            tree[row][col] = tree[row + 1][(col * dec_num_act) + max_indexes[i]]
            new_tree = copy.deepcopy(tree)
            queue.append(new_tree)
        return queue

    def _get_ev(self, dec_instantiation: Tuple[int], row: int, bp: BeliefPropagation) -> float:
        """Return the expected value of a certain decision node instantiation
        for the agent making the decision"""
        macid = self.copy_without_cpds()
        dec_list = list(nx.topological_sort(macid.relevance_graph()))
        dec = dec_list[row]

        agent = self.whose_node[dec]      # gets the agent making that decision
        utils = self.utility_nodes_agent[agent]       # gets the utility nodes for that agent
        factor = bp.query(variables=utils, evidence=dict(zip(dec_list, dec_instantiation)))

        ev = 0
        for idx, prob in np.ndenumerate(factor.values):
            for i in range(len(utils)):  # account for each agent having multiple utilty nodes
                if prob != 0:
                    ev += prob * idx[i]
        return ev
