#Licensed to the Apache Software Foundation (ASF) under one or more contributor license
#agreements; and to You under the Apache License, Version 2.0.

from typing import List, Dict, Callable, Optional

import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.inference import BeliefPropagation
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD

import networkx as nx
import pylab as plt


class CID(BayesianModel):

    def __init__(self, *args, decision_nodes : List[str] = None, utility_nodes : List[str] = None, **kwargs) -> None:
        super(CID, self).__init__(*args, **kwargs)
        if not decision_nodes:
            raise NameError("At least one decision node must be specified")
        if not set(decision_nodes).issubset(self.nodes):
            raise NameError("Decision node must be subset of graph nodes ")
        self._decision_nodes: List[str] = decision_nodes
        if not utility_nodes:
            raise NameError("At least one utility node must be specified")
        if not set(utility_nodes).issubset(self.nodes):
            raise NameError("utility node must be subset of graph nodes ")
        self._utility_nodes: List[str] = utility_nodes

    def assign_cpd(self, node: str,
                   card : Optional[int] = None,
                   policy : Optional[Callable[[Dict[int, str]], List[str]]] = None) -> None:
        """Assigns a (new) CPD to the specified node according to given policy.
           Assigns uniform policy if only cardinality given.
        """
        if node not in self.nodes:
            raise NameError("Given node not present in graph")
        if card and policy:
            raise NameError("Conflicting arguments: Do not provide both cardinality and policy")
        if not card:
            try:
                card = self.get_cardinality(node)
            except AttributeError:
                raise NameError("Cardinality neither given nor inferred")
        if not policy:
            policy = lambda x: list(range(card))

        dist = [[] for _ in range(self.get_cardinality(node))]
        for ec in self.possible_contexts(node):
            chosen = policy(ec)
            if len(chosen) == 0:
                raise NameError("policy outputs zero values on input "+str(ec))
            for i, l in enumerate(dist):
                if i in chosen:
                    l.append(1/len(chosen))
                else:
                    l.append(0)
        evidence_card = [self.get_cardinality(v) for v in self.get_parents(node)]

        new_cpd = TabularCPD(
            variable=node,
            variable_card=card,
            values=dist,
            evidence=self.get_parents(node),
            evidence_card=evidence_card
        )
        self.add_cpds(new_cpd)

    def expected_utility(self, query: Optional[Dict[str, int]] = None):
        assert self.check_model()
        # TODO: check if query has positive probability. How?
        belief_propagation = BeliefPropagation(self)
        utility_post = belief_propagation.query(variables=self._utility_nodes, evidence=query)
        if np.isnan(utility_post.values).any():
            raise NameError("Distribution contains NaNs, did the query have positive prob" + str(query))

        utility_post.normalize()
        return sum([sum(i) * x for i, x in np.ndenumerate(utility_post.values)])

    def optimal_decisions(self, decision_node: str, context_dict : Dict[str, int]) -> List[int]:
        if decision_node not in self.nodes:
            raise NameError("Decision node not in graph")
        context_nodes = self.get_parents(decision_node)
        if set(context_dict.keys()).symmetric_difference(context_nodes):
            raise NameError("Given context doesn't match parents of decision node")

        # First assign a full-support CPD to the decision node, to avoid NaNs
        if not self.get_cpds(decision_node) or (self.get_cpds(decision_node).values == 0).any():
            self.assign_cpd(decision_node)

        dec_card = self.get_cardinality(decision_node)

        query = dict(context_dict)
        eu = []
        for d in range(dec_card):
            query[decision_node] = d
            eu.append(self.expected_utility(query))

        opt_dec = []
        for d in range(dec_card):
            if eu[d] == max(eu):
                opt_dec.append(d)
        assert len(opt_dec) > 0
        return opt_dec

    def possible_contexts(self, node: str) -> List[Dict[str, int]]:
        context_nodes = self.get_parents(node)
        res_list = [{c: 0 for c in context_nodes}]
        for c in context_nodes:
            dicts_to_add = []
            for d in res_list:
                for v in range(1, self.get_cardinality(c)):
                    d_copy = dict(d)
                    d_copy[c] = v
                    dicts_to_add.append(d_copy)
            res_list.extend(dicts_to_add)
        return res_list

    def local_optimal_policy(self, decision_node: str) -> Callable[[Dict[str, int]], List[int]]:
        return lambda pdc: self.optimal_decisions(decision_node, pdc)

    def check_sufficient_recall(self) -> bool:
        mapping_extension = self.copy()
        policy_nodes = []
        for d in self._decision_nodes:
            policy_node = "pi"+d
            policy_nodes.append(policy_node)
            mapping_extension.add_node(policy_node)
            mapping_extension.add_edge(policy_node, d)
        for i, d in enumerate(self._decision_nodes):
            fa = self.get_parents(d) + [d]
            for j in range(0, i):
                for u in self._utility_nodes:
                    if mapping_extension.is_active_trail(policy_nodes[j], u, fa):
                        return False
        return True

    def solve(self) -> float:
        """An optimal policy will be represented in the CPDs of the decision nodes"""
        for d in reversed(self._decision_nodes):
            policy = self.local_optimal_policy(d)
            self.assign_cpd(d, policy=policy)
            assert self.check_model()
        return self.expected_utility()

    def plot(self):
        nx.draw(self, with_labels=True)
        # TODO: Add node shapes for decision, utility nodes to the plot
        plt.show()
