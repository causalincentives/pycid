#Licensed to the Apache Software Foundation (ASF) under one or more contributor license
#agreements; and to You under the Apache License, Version 2.0.

from __future__ import annotations
import itertools
from inspect import getsourcelines
from logging import warning
from typing import List, Callable, Dict
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
import numpy as np


# class RandomCPD(TabularCPD):
#
#     def __init__(self, variable, card=2, evidence=[], evidence_card=[]):
#         matrix = np.ones((card, np.product(evidence_card).astype(int))) / card
#         super(RandomCPD, self).__init__(variable, card, matrix, evidence=evidence, evidence_card=evidence_card)
#
#     def __repr__(self):
#         return "<RandomCPD {}:{}>".format(self.variable, self.variable_card)
#
#     def copy(self):
#         return RandomCPD(self.variable, self.variable_card, self.evidence, self.evidence_card)


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
            assert isinstance(state_names, dict)
            assert isinstance(state_names[variable], list)
            self.state_names = state_names
        else:
            self.state_names = {variable : list(range(variable_card))}

    def scope(self) -> List[str]:
        return [self.variable]

    def copy(self) -> NullCPD:
        return NullCPD(self.variable, self.variable_card, state_names=self.state_names)

    def __repr__(self) -> str:
        return "<NullCPD {}:{}>".format(self.variable, self.variable_card)

    def __str__(self) -> str:
        return "<NullCPD {}:{}>".format(self.variable, self.variable_card)
    #def to_factor(self):
    #    return self

    def initialize_tabular_cpd(self, cid: BayesianModel) -> bool:
        """initialize the TabularCPD with a matrix representing a uniform random distribution"""
        parents = cid.get_parents(self.variable)
        parents_card = [cid.get_cardinality(p) for p in parents]
        transition_probs = np.ones((self.variable_card, np.product(parents_card).astype(int))) / self.variable_card
        super(NullCPD, self).__init__(self.variable, self.variable_card, transition_probs,
                                      parents, parents_card, state_names=self.state_names)
        return True

    def get_label(self):
        return "Rand({})".format(self.variable_card)

class FunctionCPD(TabularCPD):
    """FunctionCPD class used to specify relationship between variables with a function rather than
    a probability matrix

    Once inserted into a BayesianModel, initialize_tabular_cpd converts the function
    into a probability matrix for the TabularCPD. It is necessary to wait with this until the values
    of the parents have been computed, since the state names depends on the values of the parents.
    """

    def __init__(self, variable: str, f: Callable, evidence: List[str], state_names: Dict = None) -> None:
        """Initialize FunctionCPD with a variable name and a function

        state_names can optionally be provided, to force the domain of the distribution.
        These state_names must include the values the variable can take as a result of its
        function.
        """
        self.variable = variable
        self.variables = [self.variable]
        self.cardinality = [2]  # Placeholder values
        self.variable_card = 2
        self.f = f
        self.evidence = evidence
        if state_names:
            assert isinstance(state_names, dict)
            assert isinstance(state_names[variable], list)
            self.force_state_names = state_names[variable]
        else:
            self.force_state_names = None

    def scope(self) -> List[str]:
        return [self.variable]

    def copy(self) -> FunctionCPD:
        return FunctionCPD(self.variable, self.f, self.evidence, state_names=self.force_state_names)

    def __repr__(self) -> str:
        return "<FunctionCPD {}:{}>".format(self.variable, self.f)

    def __str__(self) -> str:
        return "<FunctionCPD {}:{}>".format(self.variable, self.f)

    def parent_values(self, cid: BayesianModel) -> List[List]:
        """Return a list of lists for the values each parent can take (based on the parent state names)"""
        parent_values = []
        for p in self.evidence:
            p_cpd = cid.get_cpds(p)
            if p_cpd and hasattr(p_cpd, 'state_names'):
                parent_values.append(p_cpd.state_names[p])
            else:
                return None
                #raise Exception("unknown values for parent {}".format(p))
        return parent_values

    def possible_values(self, cid: BayesianModel) -> List:
        """The possible values this variable can take, given the values the parents can take"""
        parent_values = self.parent_values(cid)
        if parent_values is None:
            return None
        else:
            return sorted(set([self.f(*x) for x in itertools.product(*parent_values)]))

    def initialize_tabular_cpd(self, cid: BayesianModel) -> bool:
        """Initialize the probability table for the inherited TabularCPD

        Returns True if successful, False otherwise
        """
        poss_values = self.possible_values(cid)
        if not poss_values:
            warning("won't initialize {} at this point".format(self.variable))
            return False
        if self.force_state_names:
            state_names_list = self.force_state_names
            if not set(poss_values).issubset(state_names_list):
                raise Exception("variable {} can take value outside given state_names".format(self.variable))
        else:
            state_names_list = poss_values

        card = len(state_names_list)
        evidence = cid.get_parents(self.variable)
        evidence_card = [cid.get_cardinality(p) for p in evidence]
        matrix = np.array([[int(self.f(*i) == t)
                            for i in itertools.product(*self.parent_values(cid))]
                           for t in state_names_list])
        state_names = {self.variable: state_names_list}

        super(FunctionCPD, self).__init__(self.variable, card,
                                          matrix, evidence, evidence_card,
                                          state_names=state_names)

    def get_label(self):
        sl = getsourcelines(self.f)[0][0]
        start = sl.find('lambda') + 7
        middle = sl.find(':', start, len(sl))
        end = sl.find(',', middle, len(sl))
        return sl[middle+2:end] if start!=6 else ""