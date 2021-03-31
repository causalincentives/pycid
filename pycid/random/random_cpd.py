from __future__ import annotations

import contextlib
from typing import Dict, Iterator, Sequence

import numpy as np

from pycid import StochasticFunctionCPD
from pycid.core.cpd import Outcome


@contextlib.contextmanager
def temp_seed(seed: int) -> Iterator[None]:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class RandomCPD(StochasticFunctionCPD):
    """
    Sample a random CPD, with outcomes in the given domain
    """

    def __init__(
        self, variable: str, domain: Sequence[Outcome] = [0, 1], smoothness: float = 1.0, seed: int = None
    ) -> None:
        """
        Parameters
        ----------
        variable: Name of variable

        domain: List of possible outcomes, defaults to [0, 1]

        smoothness: How different the probabilities for different probabilities are.
        When small (e.g. 0.001), most probability mass falls on a single outcome, and
        when large (e.g. 1000), the distribution approaches a uniform distribution.

        seed: Set the random seed
        """
        self.seed = seed or np.random.randint(0, 10000)
        self.smoothness = smoothness

        def random_stochastic_function(**pv: Outcome) -> Dict[Outcome, float]:
            with temp_seed(self.seed + hash(frozenset(pv.items())) % 2 ** 31 - 1):
                prob_vec = np.random.dirichlet(np.ones(len(self.domain)) * self.smoothness, size=1).flat  # type: ignore
            return {self.domain[i]: prob for i, prob in enumerate(prob_vec)}  # type: ignore

        super().__init__(
            variable,
            random_stochastic_function,
            domain=domain if domain else [0, 1],
            label=f"RandomCPD({self.smoothness}, {self.seed})",
        )

    def copy(self) -> RandomCPD:
        return RandomCPD(self.variable, self.domain, self.smoothness, self.seed)  # type: ignore
