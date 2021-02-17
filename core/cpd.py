# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.

from __future__ import annotations
import itertools
from inspect import getsourcelines
from logging import warning
import random
from typing import List, Callable, Dict, Union
from pgmpy.factors.discrete import TabularCPD  # type: ignore
from pgmpy.models import BayesianModel  # type: ignore
import numpy as np


class UniformRandomCPD(TabularCPD):
    """UniformRandomPD class creates a uniform random CPD given parents in graph

    It only becomes a fully initialized TabularCPD once the method initializeTabularCPD
    is run.
    """

    def __init__(self, variable: str, state_names: List, label: str = None):
        self.variable = variable
        self.variable_card = len(state_names)
        self.state_names = {variable: state_names}
        self.label = label if label else f"DiscUni({state_names})"
        # we call super().__init__() in initialize_tabular_cpd instead

    def copy(self) -> UniformRandomCPD:
        return UniformRandomCPD(self.variable, self.state_names[self.variable])

    def __repr__(self) -> str:
        return f"<UniformRandomCPD {self.variable}:{self.variable_card}>"

    def __str__(self) -> str:
        return f"<UniformRandomCPD {self.variable}:{self.variable_card}>"

    def initialize_tabular_cpd(self, cid: BayesianModel) -> bool:
        """initialize the TabularCPD with a matrix representing a uniform random distribution"""
        parents = cid.get_parents(self.variable)
        # check that parents are initialized
        for parent in parents:
            if not cid.get_cpds(parent):
                return False
        parents_card = [cid.get_cardinality(p) for p in parents]
        transition_matrix = np.ones((self.variable_card, np.product(parents_card).astype(int))) / self.variable_card
        super().__init__(self.variable, self.variable_card, transition_matrix,
                         parents, parents_card, state_names=self.state_names)
        return True


class FunctionCPD(TabularCPD):
    """FunctionCPD class used to specify relationship between variables with a function rather than
    a probability matrix

    Once inserted into a BayesianModel, initialize_tabular_cpd converts the function
    into a probability matrix for the TabularCPD. It is necessary to wait with this until the values
    of the parents have been computed, since the state names depends on the values of the parents.
    """

    def __init__(self, variable: str, f: Callable, evidence: List[str],
                 state_names: Dict = None, label: str = None) -> None:
        """Initialize FunctionCPD with a variable name and a function

        state_names can optionally be provided, to force the domain of the distribution.
        These state_names must include the values the variable can take as a result of its
        function.
        """
        self.variable = variable
        self.f = f
        self.evidence = evidence
        if state_names:
            assert isinstance(state_names, dict)
            assert isinstance(state_names[variable], list)
            self.force_state_names = state_names[variable]
        else:
            self.force_state_names = None
        if label:
            self.label = label
        else:
            sl = getsourcelines(self.f)[0][0]
            lambda_pos = sl.find('lambda')
            if lambda_pos > -1:  # can't infer label if not defined by lambda expression
                colon = sl.find(':', lambda_pos, len(sl))
                end = sl.find(',', colon, len(sl))  # TODO this only works for simple expressions with no commas
                self.label = sl[colon + 2: end]
            elif hasattr(self.f, "__name__"):
                self.label = self.f.__name__
            else:
                self.label = ""
        # we call super().__init__() in initialize_tabular_cpd instead

    def scope(self) -> List[str]:
        return [self.variable]

    def copy(self) -> FunctionCPD:
        state_names = {self.variable: self.force_state_names} if self.force_state_names else None
        return FunctionCPD(self.variable, self.f, self.evidence, state_names=state_names)

    def __repr__(self) -> str:
        return "<FunctionCPD {}:{}>".format(self.variable, self.f)

    def __str__(self) -> str:
        return "<FunctionCPD {}:{}>".format(self.variable, self.f)

    def parent_values(self, cid: BayesianModel) -> Union[List[List], None]:
        """Return a list of lists for the values each parent can take (based on the parent state names)"""
        parent_values = []
        for p in self.evidence:
            p_cpd = cid.get_cpds(p)
            if p_cpd and hasattr(p_cpd, 'state_names'):
                parent_values.append(p_cpd.state_names[p])
            else:
                return None
        return parent_values

    def possible_values(self, cid: BayesianModel) -> Union[List[List], None]:
        """The possible values this variable can take, given the values the parents can take"""
        parent_values = self.parent_values(cid)
        if parent_values is None:
            return None
        else:
            return sorted(set([self.f(*x) for x in itertools.product(*parent_values)]))

    def initialize_tabular_cpd(self, cid: BayesianModel) -> Union[bool, None]:
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

        super().__init__(self.variable, card,
                         matrix, evidence, evidence_card,
                         state_names=state_names)
        return None


class RandomlySampledFunctionCPD(FunctionCPD):
    """
    Instantiates a randomly chosen FunctionCPD for the variable
    """

    def __init__(self, variable: str, evidence: List[str]) -> None:
        possible_functions = [
            lambda *pv: np.prod(pv),
            lambda *pv: np.sum(pv),
            lambda *pv: 1 - np.prod(pv),
            lambda *pv: 1 - np.sum(pv),
        ]
        super().__init__(variable, random.choice(possible_functions), evidence)


class DecisionDomain(UniformRandomCPD):
    """DecisionDomain is used to specify the domain for a decision

    Under the hood it becomes a UniformRandomCPD
    """

    def __init__(self, variable: str, state_names: List):
        super().__init__(variable, state_names, label=f"Dec({state_names})")

    def copy(self) -> DecisionDomain:
        return DecisionDomain(self.variable, state_names=self.state_names[self.variable])

    def __repr__(self) -> str:
        return f"<DecisionDomain {self.variable}:{self.state_names[self.variable]}>"

    def __str__(self) -> str:
        return f"<DecisionDomain {self.variable}:{self.state_names[self.variable]}>"
