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
import networkx as nx


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

    def marginalize(self, variables, inplace=True): #TODO (maybe): decrease cardinality by len(variables)
        if not inplace:
            return self
    #def to_factor(self):
    #    return self


class CID(BayesianModel):
    def __init__(self, ebunch:List[Tuple[str, str]]=None, utilities:List[str]=None):
        super(CID, self).__init__(ebunch=ebunch)
        self.utilities = utilities

    def nr_observations(self, decision):
        #get nonrequisite observations
        nonrequisite = []
        parents = self.get_parents(decision)
        for obs in parents:
            observed = list(set(parents+ [decision]) - set([obs]))
            connected = set(self.active_trail_nodes([obs], observed=observed)[obs])
            downstream_utilities = set([i for i in self.utilities if decision in self._get_ancestors_of(i)])
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
        decisions = cid._get_decisions()
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

    def _get_decisions(self):
        decisions = [node.variable for node in self.cpds if isinstance(node, NullCPD)]
        if not decisions: #TODO: can be deleted
            raise ValueError('the cid has no NullCPDs')
        return decisions

    def _get_valid_order(self, nodes:List[str]):
        def compare(node1, node2): 
            if node1==node2:
                return 0
            elif node1 in self._get_ancestors_of(node2):
                return -1
            elif node2 in self._get_ancestors_of(node1):
                return 1
            else:
                return 0
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
        return acts[indices]

    def expected_utility(self, context:dict):
        # for example: 
        # cid = get_minimal_cid()
        # out = self.expected_utility({'D':1}) #TODO: give example that uses context
        bp = BeliefPropagation(self._impute_random_policy())
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




def get_minimal_cid():
    from pgmpy.factors.discrete.CPD import TabularCPD
    cid = CID([('A', 'B')], ['B'])
    cpd = TabularCPD('B',2,[[1., 0.], [0., 1.]], evidence=['A'], evidence_card = [2])
    nullcpd = NullCPD('A', 2)
    cid.add_cpds(nullcpd, cpd)
    return cid

def get_3node_cid():
    from pgmpy.factors.discrete.CPD import TabularCPD
    cid = CID([('S', 'D'), ('S', 'U'), ('D', 'U')], ['U'])
    scpd = TabularCPD('S',2,np.array([[.5],[.5]]))#, evidence=[], evidence_card = [])
    mat = np.array([[0,1, 1,0], [1,0,0,1]])
    ucpd = TabularCPD('U', 2, mat, evidence=['S', 'D'], evidence_card=[2,2])
    nullcpd = NullCPD('D', 2)
    cid.add_cpds(nullcpd, scpd, ucpd)
    return cid

def get_5node_cid():
    from pgmpy.factors.discrete.CPD import TabularCPD
    cid = CID([
        ('S1', 'D'), 
        ('S1', 'U1'), 
        ('S2', 'D'), 
        ('S2', 'U2'), 
        ('D', 'U1'),
        ('D', 'U2')
        ], 
        ['U1', 'U2'])
    s1cpd = TabularCPD('S1',2,np.array([[.5],[.5]]))
    s2cpd = TabularCPD('S2',2,np.array([[.5],[.5]]))
    mat = np.array([[0,1, 1,0], [1,0,0,1]])
    u1cpd = TabularCPD('U1', 2, mat, evidence=['S1', 'D'], evidence_card=[2,2])
    u2cpd = TabularCPD('U2', 2, mat, evidence=['S2', 'D'], evidence_card=[2,2])
    nullcpd = NullCPD('D', 2)
    cid.add_cpds(nullcpd, s1cpd, s2cpd, u1cpd, u2cpd)
    return cid

def get_2dec_cid():
    from pgmpy.factors.discrete.CPD import TabularCPD
    cid = CID([
        ('S1', 'S2'), 
        ('S1','D1'), 
        ('D1','S2'),
        ('S2', 'U'), 
        ('S2', 'D2'), 
        ('D2', 'U')
        ], 
        ['U'])
    s1cpd = TabularCPD('S1',2,np.array([[.5],[.5]]))#, evidence=[], evidence_card = [])
    d1cpd = NullCPD('D1', 2)
    d2cpd = NullCPD('D2', 2)
    mat1 = np.array([[0,1, 1,0], [1,0,0,1]])
    s2cpd = TabularCPD('S2', 2, mat1, evidence=['S1', 'D1'], evidence_card=[2,2])
    mat2 = np.array([[0,1, 1,0], [1,0,0,1]])
    ucpd = TabularCPD('U', 2, mat2, evidence=['S2', 'D2'], evidence_card=[2,2])
    cid.add_cpds(s1cpd, d1cpd, s2cpd, d2cpd, ucpd)
    return cid



def get_insufficient_recall_cid():
    cid = CID([('A','U'),('B','U')], ['U'])
    tabcpd = TabularCPD('U', 2, np.random.randn(2,4), evidence=['A','B'], evidence_card=[2,2])
    cid.add_cpds(NullCPD('A', 2), NullCPD('B', 2), tabcpd)
    return cid
        
