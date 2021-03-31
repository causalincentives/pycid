from __future__ import annotations

import inspect
import itertools
import types
from inspect import getsourcelines
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
from pgmpy.factors.discrete import TabularCPD  # type: ignore

Outcome = Any

if TYPE_CHECKING:
    from pycid import MACIDBase


class ParentsNotReadyException(ValueError):
    pass


def function_copy(f: Callable) -> Callable:
    """
    return a function with same code, globals, defaults, closure, and
    name (or provide a new name)
    """
    fn = types.FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)  # type: ignore
    # in case f was given attrs (note this dict is a shallow copy):
    fn.__dict__.update(f.__dict__)
    return fn


class StochasticFunctionCPD(TabularCPD):
    """
    StochasticFunctionCPD class used to specify relationship between variables with a stochastic
    function, rather than with a probability matrix

    Stochastic functions are represented with dictionaries {outcome: probability}.
    For example, {0: 0.1, 1: 0.9} describes a Bernoulli(0.9) distribution.

    It's also possible to specify the distribution as a function of the parent outcomes.
    For example, if Y is a child of binary-valued variable X, then we can say the Y
    copies the value of X with 90% probability with the function:

    lambda x: {x: 0.9, 1-x: 0.1}.

    In fact, lambda x: {x: 0.9} suffices, as available probability mass is spread evenly
    on unspecified outcomes.

    The possible outcomes can be specified with the domain= keyword to __init__.

    Once inserted into a CID, initialize_tabular_cpd converts the function
    into a probability matrix for the TabularCPD. It is necessary to wait with this until
    the domains of the parents have been specified/computed.
    """

    def __init__(
        self,
        variable: str,
        stochastic_function: Callable[..., Dict[Outcome, Union[int, float]]],
        domain: Optional[Sequence[Outcome]] = None,
        label: str = None,
    ) -> None:
        """Initialize StochasticFunctionCPD with a variable name and a stochastic function.


        Parameters
        ----------
        variable: The variable name.

        stochastic_function: A stochastic function that maps parent outcomes to a distribution
        over outcomes for this variable (see doc-string for class).
        The different parents are identified by name: the arguments to the function must
        be lowercase versions of the names of the parent variables. For example, if X has
        parents Y, S1, and Obs, the arguments to function must be y, s1, and obs.

        domain: An optional specification of the variable's domain.
            Must include all values this variable can take as a result of its function.

        label: An optional label used to describe this distribution.
        """
        self.variable = variable
        self.stochastic_function = stochastic_function
        self.cid: Optional[MACIDBase] = None

        assert isinstance(domain, (list, type(None)))
        self.force_domain: Optional[Sequence[Outcome]] = domain
        self.domain: Optional[Sequence[Outcome]] = domain

        assert isinstance(label, (str, type(None)))
        self.label = label if label is not None else self.compute_label(stochastic_function)
        # we call super().__init__() in initialize_tabular_cpd instead

    @staticmethod
    def compute_label(function: Callable) -> str:
        sl = ""
        try:
            sl = getsourcelines(function)[0][0]
        except OSError:
            lambda_pos = -1  # Could not find source
        else:
            lambda_pos = sl.find("lambda")
        if lambda_pos > -1:  # can't infer label if not defined by lambda expression
            colon = sl.find(":", lambda_pos, len(sl))
            seen_parenthesis = 0
            for i in range(colon, len(sl)):
                if sl[i] in {"(", "[", "{"}:
                    seen_parenthesis += 1
                elif sl[i] in {")", "]", "}"}:
                    seen_parenthesis -= 1
                if seen_parenthesis == 0 and sl[i] == "," or seen_parenthesis == -1:
                    return sl[colon + 2 : i]
            return sl[colon + 2 : len(sl)]
        elif hasattr(function, "__name__"):
            return function.__name__
        return ""

    def check_function_arguments_match_parent_names(self, cid: MACIDBase) -> None:
        """Raises a ValueError if the parents in the CID don't match the argument to the specified function"""
        sig = inspect.signature(self.stochastic_function).parameters
        arg_kinds = [arg_kind.kind.name for arg_kind in sig.values()]
        args = set(sig)
        lower_case_parents = {p.lower() for p in cid.get_parents(self.variable)}
        if "VAR_KEYWORD" not in arg_kinds and args != lower_case_parents:
            raise ValueError(
                f"function for {self.variable} mismatch parents on"
                f" {args.symmetric_difference(lower_case_parents)}, "
            )

    def parents_instantiated(self, cid: MACIDBase) -> bool:
        """Checks that all parents have been instantiated, which is a pre-condition for instantiating self"""
        for p in cid.get_parents(self.variable):
            p_cpd = cid.get_cpds(p)
            if not (p_cpd and hasattr(p_cpd, "state_names")):
                return False
        return True

    def parent_values(self, cid: MACIDBase) -> Iterator[Dict[str, Outcome]]:
        """Return a list of lists for the values each parent can take (based on the parent state names)"""
        assert self.parents_instantiated(cid)
        parent_values_list = []
        for p in cid.get_parents(self.variable):
            p_cpd = cid.get_cpds(p)
            if p_cpd and hasattr(p_cpd, "state_names"):
                parent_values_list.append(p_cpd.state_names[p])

        for parent_values in itertools.product(*parent_values_list):
            yield {p.lower(): parent_values[i] for i, p in enumerate(cid.get_parents(self.variable))}

    def possible_values(self, cid: MACIDBase) -> List[Outcome]:
        """The possible values this variable can take, given the values the parents can take"""
        assert self.parents_instantiated(cid)
        return sorted(
            set().union(*[self.stochastic_function(**x).keys() for x in self.parent_values(cid)])  # type: ignore
        )

    def initialize_tabular_cpd(self, cid: MACIDBase) -> None:
        """Initialize the probability table for the inherited TabularCPD.

        Requires that all parents in the CID have already been instantiated.
        """
        if not self.parents_instantiated(cid):
            raise ParentsNotReadyException(f"Parents of {self.variable} are not yet instantiated.")
        self.cid = cid
        if self.force_domain:
            if not set(self.possible_values(cid)).issubset(self.force_domain):
                raise ValueError("variable {} can take value outside given state_names".format(self.variable))

        domain: Sequence[Outcome] = self.force_domain if self.force_domain else self.possible_values(cid)

        def complete_prob_dictionary(
            prob_dictionary: Dict[Outcome, Union[int, float]]
        ) -> Dict[Outcome, Union[int, float]]:
            """Complete a probability dictionary with probabilities for missing outcomes"""
            missing_outcomes = set(domain) - set(prob_dictionary.keys())
            missing_prob_mass = 1 - sum(prob_dictionary.values())  # type: ignore
            for outcome in missing_outcomes:
                prob_dictionary[outcome] = missing_prob_mass / len(missing_outcomes)
            return prob_dictionary

        card = len(domain)
        evidence = cid.get_parents(self.variable)
        evidence_card = [cid.get_cardinality(p) for p in evidence]
        probability_list = []
        for pv in self.parent_values(cid):
            probabilities = complete_prob_dictionary(self.stochastic_function(**pv))
            probability_list.append([probabilities[t] for t in domain])
        probability_matrix = np.array(probability_list).T
        if not np.allclose(probability_matrix.sum(axis=0), 1, atol=0.01):
            raise ValueError(f"The values for {self.variable} do not sum to 1 \n{probability_matrix}")
        if (probability_matrix < 0).any() or (probability_matrix > 1).any():
            raise ValueError(f"The probabilities for {self.variable} are not within range 0-1\n{probability_matrix}")
        self.domain = domain

        super().__init__(
            self.variable, card, probability_matrix, evidence, evidence_card, state_names={self.variable: self.domain}
        )

    def copy(self) -> StochasticFunctionCPD:
        return StochasticFunctionCPD(
            str(self.variable),
            function_copy(self.stochastic_function),
            domain=list(self.force_domain) if self.force_domain else None,
        )

    def __repr__(self) -> str:
        if self.cid and self.parents_instantiated(self.cid):
            dictionary: Dict[str, Union[Dict, Outcome]] = {}
            for pv in self.parent_values(self.cid):
                probabilities = self.stochastic_function(**pv)
                for outcome in probabilities:
                    if probabilities[outcome] == 1:
                        dictionary[str(pv)] = outcome
                        break
                else:
                    dictionary[str(pv)] = probabilities
            mapping = "\n".join([str(key) + "  ->  " + str(dictionary[key]) for key in sorted(list(dictionary.keys()))])
        else:
            mapping = ""
        return f"{type(self).__name__}<{self.variable}:{self.stochastic_function}> \n{mapping}"

    def __str__(self) -> str:
        return self.__repr__()


class FunctionCPD(StochasticFunctionCPD):
    """FunctionCPD class used to specify relationship between variables with a function rather than
    a probability matrix

    Once inserted into a CID, initialize_tabular_cpd converts the function
    into a probability matrix for the TabularCPD. It is necessary to wait with this until the values
    of the parents have been computed, since the state names depends on the values of the parents.
    """

    def __init__(
        self,
        variable: str,
        function: Callable[..., Outcome],
        domain: Optional[Sequence[Outcome]] = None,
        label: str = None,
    ) -> None:
        """Initialize FunctionCPD with a variable name and a function


        Parameters
        ----------
        variable: The variable name.

        function: A function mapping parent outcomes to an outcome for this variable.
        The different parents are identified by name: the arguments to the function must
        be lowercase versions of the names of the parent variables. For example, if X has
        parents Y, S1, and Obs, the arguments to function must be y, s1, and obs.

        domain: An optional specification of the variable's domain.
            Must include all values this variable can take as a result of its function.

        label: An optional label used to describe this distribution.
        """
        self.function = function
        super().__init__(
            variable,
            lambda **x: {function(**x): 1},
            domain=domain,
            label=label if label is not None else StochasticFunctionCPD.compute_label(function),
        )


class UniformRandomCPD(StochasticFunctionCPD):
    """UniformRandomPD class creates a uniform random CPD given parents in graph

    It only becomes a fully initialized TabularCPD once the method initializeTabularCPD
    is run.
    """

    def __init__(self, variable: str, domain: Sequence[Outcome], label: Optional[str] = None):
        """Create a UniformRandomCPD

        Call `initialize_tabular_cpd` to complete the initialization.

        Parameters
        ----------
        variable: The variable name.

        domain: The possible outcomes of the variable.

        label: An optional label used to describe this distribution.
        """
        super().__init__(
            variable, lambda **pv: {}, domain=domain, label=label if label is not None else f"DiscUni({domain})"
        )

    def copy(self) -> UniformRandomCPD:
        return UniformRandomCPD(str(self.variable), list(self.domain), label=str(self.label))  # type: ignore

    def __repr__(self) -> str:
        return f"<UniformRandomCPD {self.variable}:{self.variable_card}>"


class DecisionDomain(UniformRandomCPD):
    """DecisionDomain is used to specify the domain for a decision

    Under the hood it becomes a UniformRandomCPD, to satisfy BayesianModel.check_model()
    """

    def __init__(self, variable: str, domain: Sequence[Outcome]):
        """Create a DecisionDomain

        Call `initialize_tabular_cpd` to complete the initialization.

        Parameters
        ----------
        variable: The variable name.

        domain: The allowed decisions.
        """
        super().__init__(variable, domain, label=f"Dec({domain})")

    def copy(self) -> DecisionDomain:
        return DecisionDomain(str(self.variable), domain=list(self.domain) if self.domain else None)  # type: ignore

    def __repr__(self) -> str:
        return f"<DecisionDomain {self.variable}:{self.domain}>"
