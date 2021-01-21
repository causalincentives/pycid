# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.

from __future__ import annotations
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
import logging
from typing import List, Tuple, Dict, Any, Callable
from pgmpy.inference.ExactInference import BeliefPropagation
import networkx as nx
from core.cpd import UniformRandomCPD, FunctionCPD, DecisionDomain
from core.macid_base import MACIDBase


class CID(MACIDBase):

    def __init__(self, edges: List[Tuple[str, str]],
                 decision_nodes: List[str],
                 utility_nodes: List[str]):
        super().__init__(edges, {0: {'D': decision_nodes, 'U': utility_nodes}})

    def check_sufficient_recall(self) -> bool:
        # TODO update to use MACID relevance graph
        decision_ordering = self._get_valid_order(self.all_decision_nodes)
        for i, decision1 in enumerate(decision_ordering):
            for j, decision2 in enumerate(decision_ordering[i+1:]):
                for utility in self.all_utility_nodes:
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

    def impute_random_policy(self) -> None:
        """Impute a random policy to all decision nodes"""
        for d in self.all_decision_nodes:
            self.impute_random_decision(d)

    def impute_optimal_policy(self) -> None:
        """Impute a subgame perfect optimal policy to all decision nodes"""
        if not self.check_sufficient_recall():
            raise Exception("CID lacks sufficient recall, so cannot be solved by backwards induction")
        decisions = reversed(self._get_valid_order(self.all_decision_nodes))
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
        """Return dictionary with subgame perfect global policy

        to impute back the result, use add_cpds(*list(cid.solve().values())),
        or the impute_optimal_policy method
        """
        new_cid = self.copy()
        new_cid.impute_optimal_policy()
        return {d: new_cid.get_cpds(d) for d in new_cid.all_decision_nodes}

    def copy(self) -> CID:
        model_copy = CID(self.edges(), decision_nodes=self.all_decision_nodes, utility_nodes=self.all_utility_nodes)
        if self.cpds:
            model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        return model_copy

    def _get_color(self, node: str) -> str:
        if node in self.all_decision_nodes:
            return 'lightblue'
        elif node in self.all_utility_nodes:
            return 'yellow'
        else:
            return 'lightgray'
