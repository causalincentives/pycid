#Licensed to the Apache Software Foundation (ASF) under one or more contributor license
#agreements; and to You under the Apache License, Version 2.0.

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pgmpy.factors.base import BaseFactor
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.factors.continuous import ContinuousFactor
import logging
from typing import List, Tuple, Dict
from pgmpy.inference.ExactInference import BeliefPropagation
import networkx as nx
from cpd import NullCPD, FunctionCPD
import warnings


class CID(BayesianModel):

    def __init__(self, ebunch:List[Tuple[str, str]],
                 decision_nodes: List[str],
                 utility_nodes:List[str]):
        super(CID, self).__init__(ebunch=ebunch)
        self.decision_nodes = decision_nodes
        self.utility_nodes = utility_nodes
        assert set(self.nodes).issuperset(self.decision_nodes)
        assert set(self.nodes).issuperset(self.utility_nodes)
        self.cpds_to_add = {}

    def add_cpds(self, *cpds: BaseFactor, update_all: bool = True) -> None:
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
                if hasattr(cpd, "initializeTabularCPD"):
                    cpd.initializeTabularCPD(self)
                if hasattr(cpd, "values"):
                    super(CID, self).add_cpds(cpd)
                    del self.cpds_to_add[var]

    def _get_valid_order(self, nodes:List[str]):
        srt = [i for i in nx.topological_sort(self) if i in nodes]
        return srt

    def check_sufficient_recall(self) -> bool:
        decision_ordering = self._get_valid_order(self.decision_nodes)
        for i, decision1 in enumerate(decision_ordering):
            for j, decision2 in enumerate(decision_ordering[i+1:]):
                for utility in self.utility_nodes:
                    if decision2 in self._get_ancestors_of(utility):
                        cid_with_policy = self.copy()
                        cid_with_policy.add_edge('pi',decision1)
                        observed = cid_with_policy.get_parents(decision2) + [decision2]
                        connected = cid_with_policy.is_active_trail('pi', utility, observed=observed)
                        #print(decision1, decision2, connected)
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
        parents = self.get_parents(d)
        idx2name = self.get_cpds(d).no_to_name[d]
        card = self.get_cardinality(d)
        new = self.copy()
        new.impute_random_decision(d)

        @lru_cache(maxsize=1000)
        def opt_policy(*pv: tuple) -> float:
            context = {p: pv[i] for i, p in enumerate(parents)}
            eu = []
            for d_idx in range(card):
                context[d] = d_idx
                eu.append(new.expected_utility(context))
            return idx2name[np.argmax(eu)]

        self.add_cpds(FunctionCPD(d, opt_policy, parents), update_all=False)

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
            return new.expected_value(y, context)

        self.add_cpds(FunctionCPD(d, cond_exp_policy, parents), update_all=False)

    def freeze_policy(self, d: str) -> None:
        """Replace a FunctionCPD with the corresponding TabularCPD, to prevent it from updating later"""
        self.add_cpds(self.get_cpds(d).convertToTabularCPD())

    def solve(self) -> Dict:
        """Return dictionary with subgame perfect global policy"""
        new_cid = self.copy()
        new_cid.impute_optimal_policy()
        return {d: new_cid.get_cpds(d) for d in new_cid.decision_nodes}

    def _query(self, query: List["str"], context: Dict["str", "Any"]):
        #outputs P(U|context)*P(context).
        #Use context={} to get P(U). Or use factor.normalize to get p(U|context)

        #query fails if graph includes nodes not in moralized graph, so we remove them
        # cid = self.copy()
        # mm = MarkovModel(cid.moralize().edges())
        # for node in self.nodes:
        #     if node not in mm.nodes:
        #         cid.remove_node(node)
        # filtered_context = {k:v for k,v in context.items() if k in mm.nodes}

        bp = BeliefPropagation(self)
        #factor = bp.query(query, filtered_context)
        factor = bp.query(query, context)
        return factor

    def expected_utility(self, context: Dict["str", "Any"]) -> float:
        """Compute the expected utility for a given context

        For example:
        cid = get_minimal_cid()
        out = self.expected_utility({'D':1}) #TODO: give example that uses context"""
        factor = self._query(self.utility_nodes, context)
        factor.normalize() #make probs add to one

        ev = 0
        for idx, prob in np.ndenumerate(factor.values):
            utils = [factor.state_names[factor.variables[i]][j] for i,j in enumerate(idx) ]
            ev += np.sum(utils) * prob
        #ev = (factor.values * np.arange(factor.cardinality)).sum()
        return ev

    def expected_value(self, variable: str, context: dict) -> float:
        """Compute the expected value of a real-valued variable for a given context"""
        factor = self._query([variable], context)
        factor.normalize() #make probs add to one

        ev = 0.0
        for idx, prob in np.ndenumerate(factor.values):
            utils = [factor.state_names[factor.variables[i]][j] for i,j in enumerate(idx) ]
            ev += np.sum(utils) * prob
        return ev

    # def check_model(self, allow_null=True):
    #     """
    #     Check the model for various errors. This method checks for the following
    #     errors.
    #     * Checks if the sum of the probabilities for each state is equal to 1 (tol=0.01).
    #     * Checks if the CPDs associated with nodes are consistent with their parents.
    #     Returns
    #     -------
    #     check: boolean
    #         True if all the checks are passed
    #     """
    #     for node in self.nodes():
    #         cpd = self.get_cpds(node=node)
    #
    #         if cpd is None:
    #             raise ValueError("No CPD associated with {}".format(node))
    #         elif isinstance(cpd, (NullCPD, FunctionCPD)):
    #             if not allow_null:
    #                 raise ValueError(
    #                     "CPD associated with {node} is null or function cpd".format(node=node)
    #                 )
    #         elif isinstance(cpd, (TabularCPD, ContinuousFactor)):
    #             evidence = cpd.get_evidence()
    #             parents = self.get_parents(node)
    #             if set(evidence if evidence else []) != set(parents if parents else []): #TODO: do es this check appropriate cardinalities?
    #                 raise ValueError(
    #                     "CPD associated with {node} doesn't have "
    #                     "proper parents associated with it.".format(node=node)
    #                 )
    #             if not cpd.is_valid_cpd():
    #                 raise ValueError(
    #                     "Sum or integral of conditional probabilites for node {node}"
    #                     " is not equal to 1.".format(node=node)
    #                 )
    #     return True

    def copy(self) -> CID:
        model_copy = CID(self.edges(), decision_nodes=self.decision_nodes, utility_nodes=self.utility_nodes)
        if self.cpds:
            model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        return model_copy

    def __get_color(self, node):
        if node.startswith('D'):
            return 'lightblue'
        elif node.startswith('U'):
            return 'yellow'
        else:
            return 'lightgray'

    def draw(self):
        l = nx.kamada_kawai_layout(self)
        colors = [self.__get_color(node) for node in self.nodes]
        nx.draw_networkx(self, pos=l, node_color=colors)
