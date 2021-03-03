# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.
from __future__ import annotations
from core.cpd import FunctionCPD
import numpy as np
from typing import Any, List, Tuple, Dict, Union
import itertools
import networkx as nx
import matplotlib.pyplot as plt
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
        - Each NE comes as a list of FunctionCPDs, one for each decision node in the MACID.
        """
        return self.get_all_pure_ne_in_sg()

    # def get_all_pure_ne(self) -> List[List[FunctionCPD]]:
    # TODO: Keep this here in case Tom thinks we should keep this direct method for computing all
    # pure NE in a MACID.
    #     """
    #     Return a list of all pure Nash equilbiria - where each NE comes as a
    #     list of each decision node's corresponding FunctionCPD.
    #     """
    #     all_pure_ne = []

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
    #             all_pure_ne.append(jp)
    #     return all_pure_ne

    def get_all_pure_ne_in_sg(self, decisions_in_sg: List[str] = [],
                              partial_policy_profile: List[FunctionCPD] = None) -> List[List[FunctionCPD]]:
        """
        Return a list of all pure Nash equilbiria in a MACID subgame given some partial_policy_profile over
        some of the MACID's decision nodes.
        - Each NE comes as a list of FunctionCPDs, one for each decision node in the MAID subgame.
        - If decisions_in_sg is not specified, this method finds all pure NE in the full MACID.
        - If a partial policy is specified, the decison rules of decision nodes specified by the partial policy
        remain unchanged.
        TODO: Check that the decisions in decisions_in_sg actually make up a MAID subgame
        """
        if not decisions_in_sg:
            decisions_in_sg = self.all_decision_nodes

        for dec in decisions_in_sg:
            if dec not in self.all_decision_nodes:
                raise Exception(f"The node {dec} is not a decision node in the (MACID")

        agents_in_sg = list({self.whose_node[dec] for dec in decisions_in_sg})
        all_pure_ne_in_sg: List[List[FunctionCPD]] = []

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
            partial_profile_assigned = self.policy_profile_assignment(partial_policy_profile)
            decs_already_optimised = [k for k, v in partial_profile_assigned.items() if v]
            decs_to_be_randomised = [dec for dec in decs_not_in_sg if dec not in decs_already_optimised]
        else:
            decs_to_be_randomised = decs_not_in_sg

        # NE finder
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
                all_pure_ne_in_sg.append(list(pp))

        return all_pure_ne_in_sg

    def policy_profile_assignment(self, partial_policy: List[FunctionCPD]) -> Dict:
        """Return a dictionary with the joint or partial policy profile assigned -
        ie a decision rule for each of the MACIM's decision nodes."""
        new_macid = self.copy_without_cpds()
        new_macid.add_cpds(*partial_policy)
        return {d: new_macid.get_cpds(d) for d in new_macid.all_decision_nodes}

    def get_all_pure_spe(self) -> List[List[FunctionCPD]]:
        """Return a list of all pure subgame perfect Nash equilbiria (SPE) in the MACIM
        - Each SPE comes as a list of FunctionCPDs, one for each decision node in the MACID.
        """
        spes: List[List[FunctionCPD]] = [[]]
        crg = CondensedRelevanceGraph(self)
        dec_scc_mapping = crg.graph['mapping']
        scc_dec_mapping: Dict[int, List[str]] = {}
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
