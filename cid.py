# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.

from __future__ import annotations
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
import logging
from typing import List, Tuple, Dict, Any
from pgmpy.inference.ExactInference import BeliefPropagation
import networkx as nx
from cpd import NullCPD, FunctionCPD


class CID(BayesianModel):

    def __init__(self, edges: List[Tuple[str, str]],
                 decision_nodes: List[str],
                 utility_nodes: List[str]):
        super(CID, self).__init__(ebunch=edges)
        self.decision_nodes = decision_nodes
        self.utility_nodes = utility_nodes
        assert set(self.nodes).issuperset(self.decision_nodes)
        assert set(self.nodes).issuperset(self.utility_nodes)
        self.cpds_to_add = {}

    def add_cpds(self, *cpds: TabularCPD, update_all: bool = True) -> None:
        """Add the given CPDs and initiate NullCPDs and FunctionCPDs

        The update_all option recomputes the state_names and matrices for all CPDs in the graph.
        It can be set to false to save time, if the added CPD(s) have identical state_names to
        the ones they are replacing.
        """
        if update_all:
            for cpd in self.cpds:
                self.cpds_to_add[cpd.variable] = cpd
        for cpd in cpds:
            assert cpd.variable in self.nodes
            self.cpds_to_add[cpd.variable] = cpd

        for var in nx.topological_sort(self):
            if var in self.cpds_to_add:
                cpd = self.cpds_to_add[var]
                if hasattr(cpd, "initialize_tabular_cpd"):
                    cpd.initialize_tabular_cpd(self)
                if hasattr(cpd, "values"):
                    super(CID, self).add_cpds(cpd)
                    del self.cpds_to_add[var]

    def _get_valid_order(self, nodes: List[str]):
        srt = [i for i in nx.topological_sort(self) if i in nodes]
        return srt

    def check_sufficient_recall(self) -> bool:
        decision_ordering = self._get_valid_order(self.decision_nodes)
        for i, decision1 in enumerate(decision_ordering):
            for j, decision2 in enumerate(decision_ordering[i+1:]):
                for utility in self.utility_nodes:
                    if decision2 in self._get_ancestors_of(utility):
                        cid_with_policy = self.copy()
                        cid_with_policy.add_edge('pi', decision1)
                        observed = cid_with_policy.get_parents(decision2) + [decision2]
                        connected = cid_with_policy.is_active_trail('pi', utility, observed=observed)
                        if connected:
                            logging.warning(
                                    "{} has insufficient recall of {} due to utility {}".format(
                                        decision2, decision1, utility)
                                    )
                            return False
        return True

    def impute_random_decision(self, d: str) -> None:
        """Impute a random policy to the given decision node"""
        current_cpd = self.get_cpds(d)
        if current_cpd:
            sn = current_cpd.state_names
            card = self.get_cardinality(d)
        else:
            sn = None
            card = 2
        self.add_cpds(NullCPD(d, card, state_names=sn))

    def impute_random_policy(self) -> None:
        """Impute a random policy to all decision nodes"""
        for d in self.decision_nodes:
            self.impute_random_decision(d)

    def impute_optimal_decision(self, d: str) -> None:
        """Impute an optimal policy to the given decision node"""
        self.impute_random_decision(d)
        card = self.get_cardinality(d)
        parents = self.get_parents(d)
        idx2name = self.get_cpds(d).no_to_name[d]
        state_names = self.get_cpds(d).state_names
        new = self.copy()  # this "freezes" the policy so it doesn't adapt to future interventions

        @lru_cache(maxsize=1000)
        def opt_policy(*pv: tuple) -> float:
            context = {p: pv[i] for i, p in enumerate(parents)}
            eu = []
            for d_idx in range(card):
                context[d] = d_idx
                eu.append(new.expected_utility(context))
            return idx2name[np.argmax(eu)]

        self.add_cpds(FunctionCPD(d, opt_policy, parents, state_names=state_names, label="opt"),
                      update_all=False)

    def impute_optimal_policy(self) -> None:
        """Impute a subgame perfect optimal policy to all decision nodes"""
        decisions = self._get_valid_order(self.decision_nodes)
        for d in decisions:
            self.impute_optimal_decision(d)

    def impute_conditional_expectation_decision(self, d: str, y: str) -> None:
        """Imputes a policy for d = the expectation of y conditioning on d's parents"""
        parents = self.get_parents(d)
        new = self.copy()

        @lru_cache(maxsize=1000)
        def cond_exp_policy(*pv: tuple) -> float:
            context = {p: pv[i] for i, p in enumerate(parents)}
            return new.expected_value([y], context)[0]

        self.add_cpds(FunctionCPD(d, cond_exp_policy, parents, label="cond_exp({})".format(y)))

    def solve(self) -> Dict:
        """Return dictionary with subgame perfect global policy"""
        new_cid = self.copy()
        new_cid.impute_optimal_policy()
        return {d: new_cid.get_cpds(d) for d in new_cid.decision_nodes}

    def _query(self, query: List[str], context: Dict[str, Any], intervention: dict = None):
        """Return P(query|context, do(intervention))*P(context | do(intervention)).

        Use context={} to get P(query). Or use factor.normalize to get p(query|context)"""

        # query fails if graph includes nodes not in moralized graph, so we remove them
        # cid = self.copy()
        # mm = MarkovModel(cid.moralize().edges())
        # for node in self.nodes:
        #     if node not in mm.nodes:
        #         cid.remove_node(node)
        # filtered_context = {k:v for k,v in context.items() if k in mm.nodes}
        if intervention:
            cid = self.copy()
            cid.intervene(intervention)
        else:
            cid = self

        updated_state_names = {}
        for v in query:
            cpd = cid.get_cpds(v)
            updated_state_names[v] = cpd.state_names[v]

        bp = BeliefPropagation(cid)
        # factor = bp.query(query, filtered_context)
        factor = bp.query(query, context)
        factor.state_names = updated_state_names  # factor sometimes gets state_names wrong...
        return factor

    def intervene(self, intervention: dict) -> None:
        """Given a dictionary of interventions, replace the CPDs for the relevant nodes.

        Soft interventions can be achieved by using add_cpds directly.
        """
        cpds = []
        for variable, value in intervention.items():
            cpds.append(FunctionCPD(variable, lambda *x: value,
                                    evidence=self.get_parents(variable)))

        self.add_cpds(*cpds)

    def expected_value(self, variables: List[str], context: dict, intervene: dict = None) -> List[float]:
        """Compute the expected value of a real-valued variable for a given context,
        under an optional intervention
        """
        factor = self._query(variables, context, intervention=intervene)
        factor.normalize()  # make probs add to one

        ev = np.array([0.0 for _ in factor.variables])
        for idx, prob in np.ndenumerate(factor.values):
            # idx contains the information about the value each variable takes
            # we use state_names to convert index into the actual value of the variable
            ev += prob * np.array([factor.state_names[variable][idx[var_idx]]
                                   for var_idx, variable in enumerate(factor.variables)])
            if np.isnan(ev).any():
                raise Exception("query {} | {} generated Nan from idx: {}, prob: {}, \
                                consider imputing a random decision".format(variables, context, idx, prob))
        return ev.tolist()

    def expected_utility(self, context: Dict["str", "Any"], intervene: dict = None) -> float:
        """Compute the expected utility for a given context and optional intervention

        For example:
        cid = get_minimal_cid()
        out = self.expected_utility({'D':1}) #TODO: give example that uses context"""
        return sum(self.expected_value(self.utility_nodes, context, intervene=intervene))

    def copy(self) -> CID:
        model_copy = CID(self.edges(), decision_nodes=self.decision_nodes, utility_nodes=self.utility_nodes)
        if self.cpds:
            model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        return model_copy

    def _get_color(self, node):
        if node in self.decision_nodes:
            return 'lightblue'
        elif node in self.utility_nodes:
            return 'yellow'
        else:
            return 'lightgray'

    def _get_shape(self, node):
        if node in self.decision_nodes:
            return 's'
        elif node in self.utility_nodes:
            return 'D'
        else:
            return 'o'

    def _get_label(self, node):
        cpd = self.get_cpds(node)
        if hasattr(cpd, "label"):
            return cpd.label
        else:
            return ""

    def draw(self, node_color=None, node_shape=None, node_label=None):
        color = node_color if node_color else self._get_color
        shape = node_shape if node_shape else self._get_shape
        label = node_label if node_label else self._get_label
        layout = nx.kamada_kawai_layout(self)
        labeldict = {node: label(node) for node in self.nodes}
        pos_higher = {}
        for k, v in layout.items():
            if v[1] > 0:
                pos_higher[k] = (v[0] - 0.1, v[1] - 0.2)
            else:
                pos_higher[k] = (v[0] - 0.1, v[1] + 0.2)
        nx.draw_networkx(self, pos=layout, node_size=800, arrowsize=20)
        nx.draw_networkx_labels(self, pos_higher, labeldict)
        for node in self.nodes:
            nx.draw_networkx(self.to_directed().subgraph([node]), pos=layout, node_size=800, arrowsize=20,
                             node_color=color(node),
                             node_shape=shape(node))
        plt.show()