#Licensed to the Apache Software Foundation (ASF) under one or more contributor license
#agreements; and to You under the Apache License, Version 2.0.

import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.factors.continuous import ContinuousFactor
import logging
from typing import List, Tuple
import itertools
from pgmpy.inference.ExactInference import BeliefPropagation
import networkx as nx
from cpd import NullCPD, FunctionCPD
import warnings
from pgmpy.models import MarkovModel


class CID(BayesianModel):
    def __init__(self, ebunch:List[Tuple[str, str]],
                 decision_nodes: List[str],
                 utility_nodes:List[str]):
        super(CID, self).__init__(ebunch=ebunch)
        self.decision_nodes = decision_nodes
        self.utility_nodes = utility_nodes

    def nr_observations(self, decision):
        #get nonrequisite observations
        nonrequisite = []
        parents = self.get_parents(decision)
        for obs in parents:
            observed = list(set(parents+ [decision]) - set([obs]))
            connected = set(self.active_trail_nodes([obs], observed=observed)[obs])
            downstream_utilities = set([i for i in self.utility_nodes if decision in self._get_ancestors_of(i)])
            #if len([u for u in downstream_utilities if u in connected])==0:
            #import ipdb; ipdb.set_trace()
            if not connected.intersection(downstream_utilities):
                nonrequisite.append(obs)
        return nonrequisite

    def trimmed(self):
        #return the trimmed version of the graph
        #based on algorithm from Sect 4.5 of Lauritzen and Nilsson 2011, but simplified
        #using the assumption that the graph is soluble
        cid = self.copy()
        decisions = cid.decision_nodes
        while True:
            removed = 0
            for decision in decisions:
                nonrequisite = cid.nr_observations(decision)
                for nr in nonrequisite:
                    removed += 1
                    cid.remove_edge(nr, decision)
            if removed==0:
                break
        return cid

    def _get_valid_order(self, nodes:List[str]):
        srt = [i for i in nx.topological_sort(self) if i in nodes]
        return srt

    def check_sufficient_recall(self):
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

    def add_cpds(self, *cpds: DiscreteFactor) -> None:
        """Add the given CPDs and initiate NullCPDs and FunctionCPDs"""
        for cpd in cpds:
            if not isinstance(cpd, (TabularCPD, ContinuousFactor, NullCPD, FunctionCPD)):
                raise ValueError("Only TabularCPD, ContinuousFactor, FunctionCPD, or NullCPD can be added.")

            if set(cpd.scope()) - set(cpd.scope()).intersection(set(self.nodes())):
                raise ValueError("CPD defined on variable not in the model", cpd)

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    logging.warning(
                        "Replacing existing CPD for {var}".format(var=cpd.variable)
                    )
                    self.cpds[prev_cpd_index] = cpd
                    break
            else:
                self.cpds.append(cpd)

        # Once all CPDs are added, initialize the TablularCPD matrices
        for cpd in cpds:
            if isinstance(cpd, FunctionCPD) or isinstance(cpd, NullCPD):
                cpd.initializeTabularCPD(self)

    def impute_optimal_policy(self):
        """Impute an optimal policy to all decision nodes"""
        self.add_cpds(*self.solve().values())

    def _indices_to_prob_table(self, indices, n_actions):
        return np.eye(n_actions)[indices].T

    def solve(self):
        """Return dictionary with subgame perfect global policy"""
        new_cid = self.copy()
        # get ordering
        decisions = self._get_valid_order(self.decision_nodes)
        # solve in reverse ordering
        sp_policies = {}
        for decision in reversed(decisions):
            sp_policy = new_cid._get_sp_policy(decision)  # TODO make more efficient by just instantiating one BP object
            new_cid.add_cpds(sp_policy)
            sp_policies[decision] = sp_policy
        # input each policy once it's solve
        return sp_policies

    def _possible_contexts(self, decision):
        parents = self.get_parents(decision)
        if parents:
            contexts = []
            parent_cards = [self.get_cardinality(p) for p in parents]
            context_tuples = itertools.product(*[range(card) for card in parent_cards])
            for context_tuple in context_tuples:
                contexts.append({p:c for p,c in zip(parents, context_tuple)})
            return contexts
        else:
            return None

    def _get_sp_policy(self, decision):
        actions = []
        contexts = self._possible_contexts(decision)
        if contexts:
            for context in contexts:
                act = self._optimal_decisions(decision, context)[0]
                actions.append(act)
        else:
            act = self._optimal_decisions(decision, {})[0]
            actions.append(act)

        prob_table = self._indices_to_prob_table(actions, self.get_cardinality(decision))

        variable_card = self.get_cardinality(decision)
        evidence = self.get_parents(decision)
        evidence_card = [self.get_cardinality(e) for e in evidence]
        cpd = TabularCPD(
                decision,
                variable_card,
                prob_table,
                evidence,
                evidence_card
                )
        return cpd

    def _optimal_decisions(self, decision, context):
        utilities = []
        #net = cid._impute_random_policy()
        acts = np.arange(self.get_cpds(decision).variable_card)
        for act in acts:
            context = context.copy()
            context[decision] = act
            ev = self.expected_utility(context)
            utilities.append(ev)
        indices = np.where(np.array(utilities)==np.max(utilities))
        if len(acts[indices])==0:
            warnings.warn('zero prob on {} so all actions deemed optimal'.format(context))
            return np.array(acts)
        return acts[indices]

    def _query(self, query, context):
        #outputs P(U|context)*P(context).
        #Use context={} to get P(U). Or use factor.normalize to get p(U|context)

        #query fails if graph includes nodes not in moralized graph, so we remove them
        cid = self.copy()
        mm = MarkovModel(cid.moralize().edges())
        for node in self.nodes:
            if node not in mm.nodes:
                cid.remove_node(node)
        filtered_context = {k:v for k,v in context.items() if k in mm.nodes}

        bp = BeliefPropagation(self)
        factor = bp.query(query, filtered_context)
        return factor

    def expected_utility(self, context:dict):
        # for example:
        # cid = get_minimal_cid()
        # out = self.expected_utility({'D':1}) #TODO: give example that uses context
        factor = self._query(self.utility_nodes, context)
        factor.normalize() #make probs add to one

        ev = 0
        for idx, prob in np.ndenumerate(factor.values):
            utils = [factor.state_names[factor.variables[i]][j] for i,j in enumerate(idx) ]
            ev += np.sum(utils) * prob
        #ev = (factor.values * np.arange(factor.cardinality)).sum()
        return ev

    def copy(self):
        model_copy = CID(self.edges(), decision_nodes=self.decision_nodes, utility_nodes=self.utility_nodes, )
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
