#Licensed to the Apache Software Foundation (ASF) under one or more contributor license
#agreements; and to You under the Apache License, Version 2.0.
import itertools
from typing import List
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
import numpy as np


def get_random_cpd(name, card, evidence, evidence_card):
    transition_probs = np.ones((card, np.product(evidence_card).astype(int))) / card
    return TabularCPD(
        name,
        card,
        transition_probs,
        evidence=evidence,
        evidence_card=evidence_card
    )


class NullCPD(TabularCPD):
    """NullCPD class is a class used for decision nodes, to avoid having to specify a probability matrix

    It only becomes a fully initialized TabularCPD once the method initializeTabularCPD
    is run.
    """

    def __init__(self, variable, variable_card, state_names=None):
        self.variable = variable
        self.variable_card = variable_card #is this correct?
        self.cardinality = [variable_card] #TODO: possible problem because this usually includes cardinality of parents
        self.variables = [self.variable]
        if state_names:
            self.state_names = state_names[variable]
        else:
            self.state_names = {variable : list(range(variable_card))}

    def scope(self):
        return [self.variable]

    def copy(self):
        return NullCPD(self.variable, self.variable_card, state_names=self.state_names)

    def __repr__(self):
        return "<NullCPD {}:{}>".format(self.variable, self.variable_card)
    #def to_factor(self):
    #    return self

    def initializeTabularCPD(self, cid):
        """initialize the TabularCPD with a matrix representing a uniform random distribution"""
        parents = cid.get_parents(self.variable)
        parents_card = [cid.get_cardinality(p) for p in parents]
        transition_probs = np.ones((self.variable_card, np.product(parents_card).astype(int))) / self.variable_card
        super(NullCPD, self).__init__(self.variable, self.variable_card, transition_probs,
                                      parents, parents_card)


class FunctionCPD(TabularCPD):
    """FunctionCPD class used to specify relationship between variables with a function rather than
    a probability matrix

    Once inserted into a BayesianModel, the function initializeTabularCPD converts the function
    into a probability table for the TabularCPD. It is necessary to wait with this until a surrounding
    model is known, since the state names depends on the values of the parents.
    """

    def __init__(self, variable: str, f, evidence):
        self.variable = variable
        self.f = f
        self.evidence = evidence

    def scope(self):
        return [self.variable]

    def copy(self):
        return FunctionCPD(self.variable, self.f, self.evidence)

    def __repr__(self):
        return "<FunctionCPD {}:{}>".format(self.variable, self.f)

    def parent_values(self, cid: BayesianModel) -> List[List]:
        """Return a list of lists for the values each parent can take (based on the parent state names)"""
        parent_values = []
        for p in self.evidence:
            p_cpd = cid.get_cpds(p)
            if isinstance(p, FunctionCPD):
                parent_values.append(p.possible_values())
            elif p_cpd.state_names:
                parent_values.append(list(p_cpd.state_names[p]))
            else:
                raise Exception("unknown parent values for {}".format(p))
        return parent_values

    def possible_values(self, cid: BayesianModel) -> List:
        """The possible values this variable can take, given the values the parents can take"""
        parent_values = self.parent_values(cid)
        return sorted(set([self.f(*x) for x in itertools.product(*parent_values)]))

    def initializeTabularCPD(self, cid: BayesianModel) -> None:
        """Initialize the probability table for the inherited TabularCPD"""
        state_names = {self.variable: self.possible_values(cid)}
        card = len(state_names[self.variable])
        evidence = cid.get_parents(self.variable)
        evidence_card = [cid.get_cpds(p).cardinality[0] for p in evidence]

        matrix = np.array([[int(self.f(*i) == t)
                            for i in itertools.product(*self.parent_values(cid))]
                           for t in state_names[self.variable]])

        super(FunctionCPD, self).__init__(self.variable, card,
                                          matrix, evidence, evidence_card,
                                          state_names=state_names)
