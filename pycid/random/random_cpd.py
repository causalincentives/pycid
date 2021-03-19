from typing import List

import numpy as np

from pycid import StochasticFunctionCPD


def random_cpd(variable: str, domain: List = [0, 1], smoothness: float = 1.0) -> StochasticFunctionCPD:
    """
    Sample a random CPD, with outcomes in the given domain

    Parameters
    ----------
    variable: Name of variable

    domain: List of possible outcomes, defaults to [0, 1]

    smoothness: How different the probabilities for different probabilities are.
    When small (e.g. 0.001), most probability mass falls on a single outcome, and
    when large (e.g. 1000), the distribution approaches a uniform distribution.
    """

    def prob_vector() -> List:
        return np.random.dirichlet(np.ones(len(domain)) * smoothness, size=1).flat  # type: ignore

    return StochasticFunctionCPD(
        variable,
        lambda **pv: {domain[i]: prob for i, prob in enumerate(prob_vector())},
        domain=domain if domain else [0, 1],
    )
