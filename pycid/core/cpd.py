from __future__ import annotations

import itertools
import random
from inspect import getsourcelines
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from pgmpy.factors.discrete import TabularCPD  # type: ignore
from pgmpy.models import BayesianModel  # type: ignore

State = Any


class UniformRandomCPD(TabularCPD):
    """UniformRandomPD class creates a uniform random CPD given parents in graph

    It only becomes a fully initialized TabularCPD once the method initializeTabularCPD
    is run.
    """

    def __init__(self, variable: str, state_names: Sequence[State], label: Optional[str] = None):
        """Create a UniformRandomCPD

        Call `initialize_tabular_cpd` to complete the initialization.

        Parameters
        ----------
        variable: The variable name.

        state_names: The possible outcomes of the variable.

        label: An optional label used to describe this distribution.
        """
        self.variable = variable
        self.variable_card = len(state_names)
        self.state_names = {variable: state_names}
        self.label = label if label is not None else f"DiscUni({state_names})"
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
        super().__init__(
            self.variable, self.variable_card, transition_matrix, parents, parents_card, state_names=self.state_names
        )
        return True


class FunctionCPD(TabularCPD):
    """FunctionCPD class used to specify relationship between variables with a function rather than
    a probability matrix

    Once inserted into a BayesianModel, initialize_tabular_cpd converts the function
    into a probability matrix for the TabularCPD. It is necessary to wait with this until the values
    of the parents have been computed, since the state names depends on the values of the parents.
    """

    def __init__(
        self,
        variable: str,
        function: Callable[..., State],
        state_names: Optional[Dict[str, Sequence[State]]] = None,
        label: str = None,
    ) -> None:
        """Initialize FunctionCPD with a variable name and a function


        Parameters
        ----------
        variable: The variable name.

        function: A function mapping evidence observations to an outcome for this variable.
            Observations are passed by position according to the order of `evidence`.

        state_names: An optional specification of the variable's domain.
            Must include all values this variable can take as a result of its function.

        label: An optional label used to describe this distribution.
        """
        self.variable = variable
        self.function = function
        self.cid: Optional[BayesianModel] = None

        if state_names is not None:
            assert isinstance(state_names, dict)
            assert isinstance(state_names[variable], list)
            self.force_state_names: Optional[Sequence[State]] = state_names[variable]
        else:
            self.force_state_names = None

        if label is not None:
            self.label = label
        else:
            sl = ""
            try:
                sl = getsourcelines(self.function)[0][0]
            except OSError:
                lambda_pos = -1  # Could not find source
            else:
                lambda_pos = sl.find("lambda")
            if lambda_pos > -1:  # can't infer label if not defined by lambda expression
                colon = sl.find(":", lambda_pos, len(sl))
                end = sl.find(",", colon, len(sl))  # TODO this only works for simple expressions with no commas
                self.label = sl[colon + 2 : end]
            elif hasattr(self.function, "__name__"):
                self.label = self.function.__name__
            else:
                self.label = ""
        # we call super().__init__() in initialize_tabular_cpd instead

    def parents_instantiated(self, cid: BayesianModel) -> bool:
        """Checks that all parents have been instantiated, which is a pre-condition for instantiating self"""
        for p in cid.get_parents(self.variable):
            p_cpd = cid.get_cpds(p)
            if not (p_cpd and hasattr(p_cpd, "state_names")):
                return False
        return True

    def parent_values(self, cid: BayesianModel) -> List[Dict[str, Any]]:
        """Return a list of lists for the values each parent can take (based on the parent state names)"""
        assert self.parents_instantiated(cid)
        parent_values = []
        for p in cid.get_parents(self.variable):
            p_cpd = cid.get_cpds(p)
            if p_cpd and hasattr(p_cpd, "state_names"):
                parent_values.append(p_cpd.state_names[p])
        pv_list = list(itertools.product(*parent_values))
        return [{p.lower(): pv[i] for i, p in enumerate(cid.get_parents(self.variable))} for pv in pv_list]

    def possible_values(self, cid: BayesianModel) -> List[List]:
        """The possible values this variable can take, given the values the parents can take"""
        assert self.parents_instantiated(cid)
        return sorted(set([self.function(**x) for x in self.parent_values(cid)]))

    def initialize_tabular_cpd(self, cid: BayesianModel) -> None:
        """Initialize the probability table for the inherited TabularCPD.

        Requires that all parents in the CID have already been instantiated.
        """
        if not self.parents_instantiated(cid):
            raise ValueError(f"Parents of {self.variable} are not yet instantiated.")
        self.cid = cid
        if self.force_state_names:
            state_names_list = self.force_state_names
            if not set(self.possible_values(cid)).issubset(state_names_list):
                raise ValueError("variable {} can take value outside given state_names".format(self.variable))
        else:
            state_names_list = self.possible_values(cid)

        card = len(state_names_list)
        evidence = cid.get_parents(self.variable)
        evidence_card = [cid.get_cardinality(p) for p in evidence]
        matrix = np.array([[int(self.function(**i) == t) for i in self.parent_values(cid)] for t in state_names_list])
        state_names = {self.variable: state_names_list}

        super().__init__(self.variable, card, matrix, evidence, evidence_card, state_names=state_names)

    def scope(self) -> List[str]:
        return [self.variable]

    def copy(self) -> FunctionCPD:
        state_names = {self.variable: self.force_state_names} if self.force_state_names else None
        return FunctionCPD(self.variable, self.function, state_names=state_names)

    def dictionary(self) -> Dict[str, str]:
        return {str(pv): str(self.function(**pv)) for pv in self.parent_values(self.cid)}

    def __repr__(self) -> str:
        mapping = ""
        if self.cid and self.parents_instantiated(self.cid):
            dictionary = self.dictionary()
            data = [["Input:         ", "Output:        "]]
            data += [[key, dictionary[key]] for key in sorted(list(dictionary.keys()))]
            for row in data:
                mapping += "\n" + "{: >15} {: >15}".format(*row)
        return f"<FunctionCPD {self.variable}:{self.function}> {mapping}"

    def __str__(self) -> str:
        return self.__repr__()


class RandomlySampledFunctionCPD(FunctionCPD):
    """
    Instantiates a randomly chosen FunctionCPD for the variable
    """

    def __init__(self, variable: str) -> None:
        possible_functions = [
            lambda **pv: np.prod(list(pv.values())),
            lambda **pv: np.sum(list(pv.values())),
            lambda **pv: 1 - np.prod(list(pv.values())),
            lambda **pv: 1 - np.sum(list(pv.values())),
        ]
        super().__init__(variable, random.choice(possible_functions))


class DecisionDomain(UniformRandomCPD):
    """DecisionDomain is used to specify the domain for a decision

    Under the hood it becomes a UniformRandomCPD
    """

    def __init__(self, variable: str, state_names: Sequence[State]):
        super().__init__(variable, state_names, label=f"Dec({state_names})")

    def copy(self) -> DecisionDomain:
        return DecisionDomain(self.variable, state_names=self.state_names[self.variable])

    def __repr__(self) -> str:
        return f"<DecisionDomain {self.variable}:{self.state_names[self.variable]}>"

    def __str__(self) -> str:
        return f"<DecisionDomain {self.variable}:{self.state_names[self.variable]}>"
