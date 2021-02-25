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
        self.decision_nodes = self.decision_nodes_agent[0]
        self.utility_nodes = self.utility_nodes_agent[0]

    def impute_optimal_policy(self) -> None:
        """Impute a subgame perfect optimal policy to all decision nodes"""
        if not self.sufficient_recall():
            raise Exception("CID lacks sufficient recall, so cannot be solved by backwards induction")
        decisions = reversed(self.get_valid_order(self.decision_nodes))
        for d in decisions:
            self.impute_optimal_decision(d)

    def impute_random_policy(self) -> None:
        """Impute a random policy to all decision nodes in the CID"""
        for d in self.decision_nodes:
            self.impute_random_decision(d)

    def solve(self) -> Dict:
        """Return dictionary with subgame perfect global policy

        to impute back the result, use add_cpds(*list(cid.solve().values())),
        or the impute_optimal_policy method
        """
        new_cid = self.copy()
        new_cid.impute_optimal_policy()
        return {d: new_cid.get_cpds(d) for d in new_cid.decision_nodes}

    def copy_without_cpds(self) -> CID:
        return CID(self.edges(), list(self.decision_nodes), list(self.utility_nodes))

    def _get_color(self, node: str) -> str:
        if node in self.decision_nodes:
            return 'lightblue'
        elif node in self.utility_nodes:
            return 'yellow'
        else:
            return 'lightgray'
