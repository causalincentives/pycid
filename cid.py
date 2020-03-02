#Licensed to the Apache Software Foundation (ASF) under one or more contributor license 
#agreements; and to You under the Apache License, Version 2.0.

import numpy as np
from pgmpy.factors.base import BaseFactor
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import (
    TabularCPD,
    JointProbabilityDistribution,
    DiscreteFactor,
)
from pgmpy.factors.continuous import ContinuousFactor
import logging
import typing
from typing import List, Tuple
import itertools
from pgmpy.inference.ExactInference import BeliefPropagation
import functools


class NullCPD(BaseFactor):
    def __init__(self, variable, variable_card):
        self.variable = variable
        self.variable_card = variable_card #is this correct?
        self.cardinality = [variable_card] #possible problem because this usually includes cardinality of parents
        self.variables = [self.variable]

    def scope(self):
        return [self.variable]

    def copy(self):
        return NullCPD(self.variable, self.variable_card)

    #def to_factor(self):
    #    return self


class CID(BayesianModel):
    def __init__(self, ebunch:List[Tuple[str, str]]=None, utilities:List[str]=None):
        super(CID, self).__init__(ebunch=ebunch)
        self.utilities = utilities

    def _get_decisions(self):
        decisions = [node.variable for node in self.cpds if isinstance(node, NullCPD)]
        if not decisions: #TODO: can be deleted
            raise ValueError('the cid has no NullCPDs')
        return decisions

    def _get_valid_order(self, nodes:List[str]):
        def compare(node1, node2): 
            return node1 != node2 and node1 in self._get_ancestors_of(node2)
        ordering = sorted(nodes, key=functools.cmp_to_key(compare))
        return ordering


    def check_sufficient_recall(self):
        decision_ordering = self._get_valid_order(self._get_decisions())
        for i, decision1 in enumerate(decision_ordering):
            for j, decision2 in enumerate(decision_ordering[i+1:]):
                for utility in self.utilities:
                    if decision2 in self._get_ancestors_of(utility):
                        cid_with_policy = self.copy()
                        cid_with_policy.add_edge('pi',decision1)
                        observed = cid_with_policy.get_parents(decision2)
                        connected = cid_with_policy.is_active_trail('pi', utility, observed=observed)
                        #print(decision1, decision2, connected)
                        if connected:
                            logging.warning(
                                    "{} has insufficient recall of {} due to utility {}".format(
                                        decision2, decision1, utility)
                                    )
                            return False
        return True

    def add_cpds(self, *cpds):
        for cpd in cpds:
            if not isinstance(cpd, (TabularCPD, ContinuousFactor, NullCPD)):
                raise ValueError("Only TabularCPD, ContinuousFactor, or NullCPD can be added.")

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

    def check_model(self, allow_null=True):
        """
        Check the model for various errors. This method checks for the following
        errors.

        * Checks if the sum of the probabilities for each state is equal to 1 (tol=0.01).
        * Checks if the CPDs associated with nodes are consistent with their parents.

        Returns
        -------
        check: boolean
            True if all the checks are passed
        """
        for node in self.nodes():
            cpd = self.get_cpds(node=node)

            if cpd is None:
                raise ValueError("No CPD associated with {}".format(node))
            elif isinstance(cpd, (TabularCPD, ContinuousFactor)):
                evidence = cpd.get_evidence()
                parents = self.get_parents(node)
                if set(evidence if evidence else []) != set(parents if parents else []):
                    raise ValueError(
                        "CPD associated with {node} doesn't have "
                        "proper parents associated with it.".format(node=node)
                    )
                if not cpd.is_valid_cpd():
                    raise ValueError(
                        "Sum or integral of conditional probabilites for node {node}"
                        " is not equal to 1.".format(node=node)
                    )
            elif isinstance(cpd, (NullCPD)):
                if not allow_null:
                    raise ValueError(
                        "CPD associated with {node} is nullcpd".format(node=node)
                    )
        return True

    #def _get_local_policies(self, decision):
    #    #returns a list of all possible CPDs
    #    pass

    def _impute_random_policy(self):
        #imputes random uniform policy to all NullCPDs
        new = self.copy()
        for cpd in new.get_cpds():
            if isinstance(cpd, NullCPD):
                n_actions = cpd.variable_card
                parents = new.get_parents(cpd.variable)
                parents_card = [self.get_cardinality(p) for p in parents]
                transition_probs = np.ones((n_actions, np.product(parents_card).astype(int)))/n_actions
                uniform_policy = TabularCPD(
                        cpd.variable, 
                        cpd.variable_card, 
                        transition_probs,
                        evidence=parents,
                        evidence_card = parents_card
                        )
                new.add_cpds(uniform_policy)
        return new

    def _indices_to_prob_table(self, indices, n_actions):
        return np.eye(n_actions)[indices].T

    def solve(self):
        #returns dictionary with subgame perfect global policy
        new_cid = self.copy()
        # get ordering
        decisions = self._get_valid_order(self._get_decisions())
        # solve in reverse ordering
        sp_policies = {}
        for decision in reversed(decisions):
            sp_policy = new_cid._get_sp_policy(decision)
            new_cid.add_cpds(sp_policy)
            sp_policies[decision] = sp_policy
        # input each policy once it's solve
        return sp_policies

    def _get_sp_policy(self, decision):
        actions = []
        parents = self.get_parents(decision)
        if parents:
            parent_cards = [self.get_cardinality(p) for p in parents]
            context_tuples = itertools.product(*[range(card) for card in parent_cards])
            for context_tuple in context_tuples:
                context = {p:c for p,c in zip(parents, context_tuple)}
                act = self._get_best_act(decision, context)
                actions.append(act)
        else:
            act = self._get_best_act(decision, {})
            actions.append(act)

        #actions = []
        #parents = self.get_parents(decision)
        #parent_cards = [self.get_cardinality(p) for p in parents]
        #contexts = itertools.product((range(card) for card in parent_cards))
        #for context in contexts:
        #    act = self._get_best_act(self, decision, context)
        #    actions.append(act)

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

    def _get_best_act(self, decision, context):
        utilities = []
        #net = cid._impute_random_policy()
        acts = np.arange(self.get_cpds(decision).variable_card)
        for act in acts:
            ev = self._act_utility(decision, context, act)
            utilities.append(ev)
        return acts[np.argmax(utilities)]

    def _act_utility(self, decision:str, context:dict, act:int):
        # for example: 
        # cid = get_minimal_cid()
        # out = self._act_utility('A', {}, 1) #TODO: give example that uses context
        bp = BeliefPropagation(self._impute_random_policy())
        context[decision] = act #add act to decision context
        factor = bp.query(self.utilities, context)
        factor.normalize() #make probs add to one

        ev = 0
        for idx, prob in np.ndenumerate(factor.values):
            utils = idx #TODO: change utils to be arbitrary-valued and use idx as indices to retrieve them
            ev += np.sum(utils) * prob
        #ev = (factor.values * np.arange(factor.cardinality)).sum()
        return ev

    def copy(self):
        model_copy = CID(utilities=self.utilities)
        model_copy.add_nodes_from(self.nodes())
        model_copy.add_edges_from(self.edges())
        if self.cpds:
            model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        return model_copy




def get_minimal_cid():
    from pgmpy.factors.discrete.CPD import TabularCPD
    cid = CID([('A', 'B')], ['B'])
    cpd = TabularCPD('B',2,[[1., 0.], [0., 1.]], evidence=['A'], evidence_card = [2])
    nullcpd = NullCPD('A', 2)
    cid.add_cpds(nullcpd, cpd)
    return cid

def get_3node_cid():
    from pgmpy.factors.discrete.CPD import TabularCPD
    cid = CID([('A', 'B'), ('B', 'C')], ['C'])
    nullcpd = NullCPD('A', 2)
    #cpd = TabularCPD('B',2, np.eye(2), evidence=['A'], evidence_card = [2])
    nullcpd2 = NullCPD('B', 2)
    cpd2 = TabularCPD('C',2, np.eye(2), evidence=['B'], evidence_card = [2])
    cid.add_cpds(nullcpd, nullcpd2, cpd2)
    #cid.add_cpds(nullcpd, cpd, cpd2)
    return cid

def get_insufficient_recall_cid():
    cid = CID([('A','U'),('B','U')], ['U'])
    tabcpd = TabularCPD('U', 2, np.random.randn(2,4), evidence=['A','B'], evidence_card=[2,2])
    cid.add_cpds(NullCPD('A', 2), NullCPD('B', 2), tabcpd)
    return cid
        
