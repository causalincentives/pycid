# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.

from __future__ import annotations
from typing import List, Tuple, Dict
from core.macid_base import MACIDBase


class CID(MACIDBase):

    def __init__(self, edges: List[Tuple[str, str]],
                 decision_nodes: List[str],
                 utility_nodes: List[str]):
        super().__init__(edges, {0: {'D': decision_nodes, 'U': utility_nodes}})

    def copy_without_cpds(self) -> CID:
        return CID(self.edges(), self.all_decision_nodes, self.all_utility_nodes)

    def impute_random_policy(self) -> None:
        """Impute a random policy to all decision nodes"""
        for d in self.all_decision_nodes:
            self.impute_random_decision(d)

    def impute_optimal_policy(self) -> None:
        """Impute a subgame perfect optimal policy to all decision nodes"""
        if not self.sufficient_recall():
            raise Exception("CID lacks sufficient recall, so cannot be solved by backwards induction")
        decisions = reversed(self._get_valid_order(self.all_decision_nodes))
        for d in decisions:
            self.impute_optimal_decision(d)

    def solve(self) -> Dict:
        """Return dictionary with subgame perfect global policy

        to impute back the result, use add_cpds(*list(cid.solve().values())),
        or the impute_optimal_policy method
        """
        new_cid = self.copy()
        new_cid.impute_optimal_policy()
        return {d: new_cid.get_cpds(d) for d in new_cid.all_decision_nodes}

    def copy(self) -> CID:
        model_copy = CID(self.edges(),
                         decision_nodes=self.all_decision_nodes,
                         utility_nodes=self.all_utility_nodes)
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
