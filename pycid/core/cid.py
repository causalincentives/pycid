from __future__ import annotations

import random
from typing import Dict, Iterable, List, Tuple

from pycid.core.cpd import StochasticFunctionCPD
from pycid.core.macid_base import MACIDBase


class CID(MACIDBase):
    """A Causal Influence Diagram"""

    def __init__(
        self,
        edges: Iterable[Tuple[str, str]],
        decisions: Iterable[str],
        utilities: Iterable[str],
    ):
        """Initialize a Causal Influence Diagram

        Parameters
        ----------
        edges: A set of directed edges. Each is a pair of node labels (tail, head).

        decisions: The decision nodes of the agent.

        utilities: The utility nodes of the agent.
        """
        # Initialize a MACID with a single agent labelled `0`
        super().__init__(edges=edges, agent_decisions={0: decisions}, agent_utilities={0: utilities})

    def impute_optimal_policy(self) -> None:
        """Impute an optimal policy to all decision nodes"""
        self.impute_random_policy()
        if self.sufficient_recall():
            decisions = self.get_valid_order(self.decisions)
            for d in reversed(decisions):
                self.impute_optimal_decision(d)
        else:
            self.add_cpds(*random.choice(self.optimal_policies()))

    def optimal_policies(self) -> List[Tuple[StochasticFunctionCPD, ...]]:
        """
        Return a list of all deterministic optimal policies.
        # TODO: Subgame perfectness option
        """
        return self.optimal_pure_policies(self.decisions)

    def impute_random_policy(self) -> None:
        """Impute a random policy to all decision nodes in the CID"""
        for d in self.decisions:
            self.impute_random_decision(d)

    def solve(self) -> Dict:
        """Return dictionary with subgame perfect global policy

        to impute back the result, use add_cpds(*list(cid.solve().values())),
        or the impute_optimal_policy method
        """
        new_cid = self.copy()
        new_cid.impute_optimal_policy()
        return {d: new_cid.get_cpds(d) for d in new_cid.decisions}

    def copy_without_cpds(self) -> CID:
        """
        Return a copy of the CID without the CPDs.
        """
        return CID(self.edges(), self.decisions, self.utilities)

    def _get_color(self, node: str) -> str:
        if node in self.decisions:
            return "lightblue"
        elif node in self.utilities:
            return "yellow"
        else:
            return "lightgray"
