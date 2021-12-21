from __future__ import annotations

import inspect
import itertools
from inspect import getsourcelines
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Union

import numpy as np
from pgmpy.factors.discrete import TabularCPD  # type: ignore

Outcome = Any

if TYPE_CHECKING:
    from pycid import CausalBayesianNetwork


class ParentsNotReadyException(ValueError):
    pass


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
        stochastic_function: Callable[..., Union[Outcome, Mapping[Outcome, Union[int, float]]]],
        cbn: CausalBayesianNetwork,
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
        match the names of the parent variables. For example, if X has
        parents Y, S1, and Obs, the arguments to function must be Y, S1, and Obs.

        domain: An optional specification of the variable's domain.
            Must include all values this variable can take as a result of its function.

        label: An optional label used to describe this distribution.
        """
        self.variable = variable
        self.func = stochastic_function
        self.cbn = cbn

        assert isinstance(domain, (list, type(None)))
        self.force_domain: Optional[Sequence[Outcome]] = domain

        assert isinstance(label, (str, type(None)))
        self.label = label if label is not None else self.compute_label(stochastic_function)

        self.check_function_arguments_match_parent_names()
        if self.force_domain:
            if not set(self.possible_values()).issubset(self.force_domain):
                raise ValueError("variable {} can take value outside given state_names".format(self.variable))

        self.domain = self.force_domain if self.force_domain else self.possible_values()

        def complete_prob_dictionary(
            prob_dictionary: Mapping[Outcome, Union[int, float]]
        ) -> Mapping[Outcome, Union[int, float]]:
            """Complete a probability dictionary with probabilities for missing outcomes"""
            prob_dictionary = {key: value for key, value in prob_dictionary.items() if value is not None}
            missing_outcomes = set(self.domain) - set(prob_dictionary.keys())
            missing_prob_mass = 1 - sum(prob_dictionary.values())  # type: ignore
            for outcome in missing_outcomes:
                prob_dictionary[outcome] = missing_prob_mass / len(missing_outcomes)
            return prob_dictionary

        card = len(self.domain)
        evidence = cbn.get_parents(self.variable)
        evidence_card = [cbn.get_cardinality(p) for p in evidence]
        probability_list = []
        for pv in self.parent_values():
            probabilities = complete_prob_dictionary(self.stochastic_function(**pv))
            probability_list.append([probabilities[t] for t in self.domain])
        probability_matrix = np.array(probability_list).T
        if not np.allclose(probability_matrix.sum(axis=0), 1, rtol=0, atol=0.01):  # type: ignore
            raise ValueError(f"The values for {self.variable} do not sum to 1 \n{probability_matrix}")
        if (probability_matrix < 0).any() or (probability_matrix > 1).any():  # type: ignore
            raise ValueError(f"The probabilities for {self.variable} are not within range 0-1\n{probability_matrix}")

        super().__init__(
            self.variable, card, probability_matrix, evidence, evidence_card, state_names={self.variable: self.domain}
        )

    def stochastic_function(self, **pv: Outcome) -> Mapping[Outcome, float]:
        ret = self.func(**pv)
        if isinstance(ret, Mapping):
            return ret
        else:
            return {ret: 1}

    @staticmethod
    def compute_label(function: Callable) -> str:
        if hasattr(function, "__name__"):
            return function.__name__
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
        return ""

    def check_function_arguments_match_parent_names(self) -> None:
        """Raises a ValueError if the parents in the CID don't match the argument to the specified function"""
        sig = inspect.signature(self.stochastic_function).parameters
        arg_kinds = [arg_kind.kind.name for arg_kind in sig.values()]
        args = set(sig)
        if "VAR_KEYWORD" not in arg_kinds and args != set(self.cbn.get_parents(self.variable)):
            raise ValueError(
                f"function for {self.variable} mismatch parents on"
                f" {args.symmetric_difference(set(self.cbn.get_parents(self.variable)))}, "
            )

    def parent_values(self) -> Iterator[Dict[str, Outcome]]:
        """Return a list of lists for the values each parent can take (based on the parent state names)"""
        parent_values_list = []
        try:
            for p in self.cbn.get_parents(self.variable):
                parent_values_list.append(self.cbn.model.domain[p])
        except KeyError:
            raise ParentsNotReadyException(f"Parent {p} of {self.variable} not yet instantiated")
        for parent_values in itertools.product(*parent_values_list):
            yield {p: parent_values[i] for i, p in enumerate(self.cbn.get_parents(self.variable))}

    def possible_values(self) -> List[Outcome]:
        """The possible values this variable can take, given the values the parents can take"""
        return sorted(
            set().union(*[self.stochastic_function(**x).keys() for x in self.parent_values()])  # type: ignore
        )

    def copy(self) -> StochasticFunctionCPD:
        return StochasticFunctionCPD(
            str(self.variable),
            self.func,
            self.cbn,
            domain=list(self.force_domain) if self.force_domain else None,
        )

    def __repr__(self) -> str:
        dictionary: Dict[str, Union[Dict, Outcome]] = {}
        try:
            for pv in self.parent_values():  # type: ignore
                probabilities = self.stochastic_function(**pv)
                for outcome in probabilities:
                    if probabilities[outcome] == 1:
                        dictionary[str(pv)] = outcome
                        break
                else:
                    dictionary[str(pv)] = probabilities
        except ParentsNotReadyException:
            pass
        mapping = "\n".join([str(key) + "  ->  " + str(dictionary[key]) for key in sorted(list(dictionary.keys()))])
        return f"{type(self).__name__}<{self.variable}:{self.func}> \n{mapping}"


class ConstantCPD(StochasticFunctionCPD):
    def __init__(
        self,
        variable: str,
        dictionary: Mapping,
        cbn: CausalBayesianNetwork,
        domain: Sequence[Outcome] = None,
        label: Optional[str] = None,
    ):
        super().__init__(variable, lambda **pv: dictionary, cbn, domain=domain, label=label or str(dictionary))


class DecisionDomain(ConstantCPD):
    """DecisionDomain is used to specify the domain for a decision

    Under the hood it becomes a UniformRandomCPD, to satisfy BayesianModel.check_model()
    """

    def __init__(self, variable: str, cbn: CausalBayesianNetwork, domain: Sequence[Outcome]):
        """Create a DecisionDomain

        Call `initialize_tabular_cpd` to complete the initialization.

        Parameters
        ----------
        variable: The variable name.

        domain: The allowed decisions.
        """
        super().__init__(variable, {}, cbn, domain, label=f"Dec({domain})")

    def copy(self) -> DecisionDomain:
        return DecisionDomain(str(self.variable), self.cbn, domain=list(self.domain))

    def __repr__(self) -> str:
        return f"<DecisionDomain {self.variable}:{self.domain}>"


def bernoulli(p: float) -> Dict[Outcome, float]:
    """Create a CPD for a variable that follows a Bernoulli(p) distribution."""
    return {0: 1 - p, 1: p}


def discrete_uniform(domain: List[Outcome]) -> Dict[Outcome, float]:
    "Assign a variable a CPD which is a discrete uniform random distribution over the given domain."
    return {outcome: 1 / len(domain) for outcome in domain}


def noisy_copy(
    value: Outcome, probability: float = 0.9, domain: List[Outcome] = None
) -> Dict[Outcome, Optional[float]]:
    """specify a variable's CPD as copying the value of some other variable with a certain probability."""
    dist = dict.fromkeys(domain) if domain else {}
    dist[value] = probability
    return dist
