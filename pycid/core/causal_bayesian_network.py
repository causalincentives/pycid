from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List, Tuple, Union, Set

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference.ExactInference import BeliefPropagation
from pgmpy.models import BayesianModel

from pycid.core.cpd import FunctionCPD, Outcome, ParentsNotReadyException, StochasticFunctionCPD, UniformRandomCPD


class CausalBayesianNetwork(BayesianModel):
    """Causal Bayesian Network

    A Causal Bayesian Network is a Bayesian Network where the directed edges represent every causal relationship
    between the Bayesian Network's variables.
    """

    def __init__(self, edges: Iterable[Tuple[str, str]]):
        """Initialize a Causal Bayesian Network

        Parameters
        ----------
        edges: A set of directed edges. Each is a pair of node labels (tail, head).
        """
        super().__init__(ebunch=edges)

        self._cpds_to_add: Dict[str, TabularCPD] = {}

    def remove_edge(self, u: str, v: str) -> None:
        super().remove_edge(u, v)
        # remove_edge can be called before cpds have been defined
        if hasattr(self, "cpds"):
            cpd = self.get_cpds(v)
            if isinstance(cpd, UniformRandomCPD):
                self.add_cpds(cpd)

    def add_edge(self, u: str, v: str) -> None:
        super().add_edge(u, v)
        # add_edge can be called before cpds have been defined
        if hasattr(self, "cpds"):
            cpd = self.get_cpds(v)
            if isinstance(cpd, UniformRandomCPD):
                self.add_cpds(cpd)

    def add_cpds(self, *cpds: TabularCPD) -> None:
        """
        Add the given CPDs and initialize FunctionCPDs, UniformRandomCPDs etc
        """

        # Add each cpd to self._cpds_to_add after doing some checks
        for cpd in cpds:
            # assert cpd.variable in self.nodes
            assert isinstance(cpd, TabularCPD)
            if isinstance(cpd, StochasticFunctionCPD):
                cpd.check_function_arguments_match_parent_names(self)
            self._cpds_to_add[cpd.variable] = cpd

        # Initialize CPDs in topological order. Call super().add_cpds if initialized
        # successfully. Otherwise leave in self._cpds_to_add.
        for var in nx.topological_sort(self):
            if var in self._cpds_to_add:
                cpd_to_add = self._cpds_to_add[var]
                if hasattr(cpd_to_add, "initialize_tabular_cpd"):
                    try:
                        cpd_to_add.initialize_tabular_cpd(self)
                    except ParentsNotReadyException:
                        pass
                if hasattr(cpd_to_add, "values"):  # cpd_to_add has been initialized
                    # if the domains have changed, remember to update all descendants:
                    previous_cpd = self.get_cpds(var)
                    if (
                        previous_cpd
                        and hasattr(previous_cpd, "domain")
                        and previous_cpd.state_names[var] != cpd_to_add.state_names[var]
                    ):
                        for descendant in nx.descendants(self, var):
                            if descendant not in self._cpds_to_add and self.get_cpds(descendant):
                                self._cpds_to_add[descendant] = self.get_cpds(descendant)

                    # add cpd to BayesianModel, and remove it from _cpds_to_add
                    #
                    # pgmpy produces warnings when overwriting an existing CPD
                    # It writes warnings directly to the 'root' context so
                    # to suppress those we disable warnings for all loggers
                    logging.disable(logging.WARN)
                    super().add_cpds(cpd_to_add)
                    logging.disable(logging.NOTSET)  # Unset
                    del self._cpds_to_add[var]

        # Sync state_names, trusting that each CPD has up-to-date knowledge about itself
        state_names = {}
        for cpd in self.get_cpds():
            state_names[cpd.variable] = cpd.state_names[cpd.variable]
        for cpd in self.get_cpds():
            cpd.store_state_names(None, None, state_names)

    def query(
        self, query: Iterable[str], context: Dict[str, Outcome], intervention: Dict[str, Outcome] = None
    ) -> BeliefPropagation:
        """Return P(query|context, do(intervention))*P(context | do(intervention)).

        Use factor.normalize to get p(query|context, do(intervention)).
        Use context={} to get P(query).

        Parameters
        ----------
        query: A set of nodes to query.

        context: Node values to condition upon. A dictionary mapping of node => outcome.

        intervention: Interventions to apply. A dictionary mapping node => outcome.
        """

        for variable, outcome in context.items():
            if outcome not in self.get_cpds(variable).state_names[variable]:
                raise ValueError(f"The outcome {outcome} is not in the domain of {variable}")

        if intervention is None:
            intervention = {}

        # First, apply the intervention (if any)
        if intervention:
            cbn = self.copy()
            cbn.intervene(intervention)
        else:
            cbn = self

        # query fails if graph includes nodes not in a connected component, so we remove them
        undirected = cbn.to_undirected()
        connected_nodes: Set[str] = set().union(  # type: ignore
            *[nx.node_connected_component(undirected, var) for var in query]
        )
        context = {k: v for k, v in context.items() if k in connected_nodes}
        for node in list(cbn.nodes):
            if node not in connected_nodes:
                cbn.remove_node(node)
        if not nx.is_connected(cbn.to_undirected()):
            raise ValueError(f"query {query} contains nodes in disconnected components")

        bp = BeliefPropagation(cbn)

        with np.errstate(invalid="ignore"):  # Suppress numpy warnings for 0/0
            factor = bp.query(query, context, show_progress=False)
        return factor

    def intervene(self, intervention: Dict[str, Outcome]) -> None:
        """Given a dictionary of interventions, replace the CPDs for the relevant nodes.

        Soft interventions can be achieved by using self.add_cpds() directly.

        Parameters
        ----------
        intervention: Interventions to apply. A dictionary mapping node => value.
        """
        for variable in intervention:
            for p in self.get_parents(variable):  # remove ingoing edges
                self.remove_edge(p, variable)
            self.add_cpds(
                FunctionCPD(
                    variable, lambda: intervention[variable], domain=self.get_cpds(variable).state_names[variable]
                )
            )

    def expected_value(
        self,
        variables: Iterable[str],
        context: Dict[str, Outcome],
        intervention: Dict[str, Outcome] = None,
    ) -> List[float]:
        """Compute the expected value of a real-valued variable for a given context,
        under an optional intervention

        Parameters
        ----------
        variables: A set of variables to evaluate.

        context: Node values to condition upon. A dictionary mapping of node => value.

        intervention: Interventions to apply. A dictionary mapping node => value.
        """
        factor = self.query(variables, context, intervention=intervention)
        factor.normalize()  # make probs add to one

        ev = np.array([0.0 for _ in factor.variables])
        for idx, prob in np.ndenumerate(factor.values):
            # idx contains the information about the value each variable takes
            # we use state_names to convert index into the actual value of the variable
            ev += prob * np.array(
                [factor.state_names[variable][idx[var_idx]] for var_idx, variable in enumerate(factor.variables)]
            )
            if np.isnan(ev).any():
                raise RuntimeError(
                    "query {} | {} generated Nan from idx: {}, prob: {}, \
                                consider imputing a random decision".format(
                        variables, context, idx, prob
                    )
                )
        return ev.tolist()  # type: ignore

    def copy_without_cpds(self) -> CausalBayesianNetwork:
        """copy the CausalBayesianNetwork object"""
        return CausalBayesianNetwork(edges=self.edges)

    def copy(self) -> CausalBayesianNetwork:
        """copy the MACIDBase object"""
        model_copy = self.copy_without_cpds()
        if self.cpds:
            model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        return model_copy

    def _get_color(self, node: str) -> Union[np.ndarray, str]:
        # TODO: the return type is like this because otherwise it violates the "Liskov substitution principle".
        # all nodes in a CBN are chance nodes
        return "lightgray"

    def _get_shape(self, node: str) -> str:
        # all nodes in a CBN are chance nodes
        return "o"

    def _get_label(self, node: str) -> str:
        cpd = self.get_cpds(node)
        if hasattr(cpd, "label"):
            return cpd.label  # type: ignore
        elif hasattr(cpd, "__name__"):
            return cpd.__name__  # type: ignore
        else:
            return ""

    def draw(
        self,
        node_color: Callable[[str], Union[str, np.ndarray]] = None,
        node_shape: Callable[[str], str] = None,
        node_label: Callable[[str], str] = None,
    ) -> None:
        """
        Draw the MACID or CID.
        """
        color = node_color if node_color else self._get_color
        shape = node_shape if node_shape else self._get_shape
        label = node_label if node_label else self._get_label
        layout = nx.kamada_kawai_layout(self)
        label_dict = {node: label(node) for node in self.nodes}
        pos_higher = {}
        for k, v in layout.items():
            if v[1] > 0:
                pos_higher[k] = (v[0] - 0.1, v[1] - 0.2)
            else:
                pos_higher[k] = (v[0] - 0.1, v[1] + 0.2)
        nx.draw_networkx(self, pos=layout, node_size=800, arrowsize=20)
        nx.draw_networkx_labels(self, pos_higher, label_dict)
        for node in self.nodes:
            nx.draw_networkx(
                self.to_directed().subgraph([node]),
                pos=layout,
                node_size=800,
                arrowsize=20,
                node_color=color(node),
                node_shape=shape(node),
            )
        plt.show()

    def draw_property(self, node_property: Callable[[str], bool], color: str = "red") -> None:
        """Draw a CBN, CID, or MACID with the nodes satisfying node_property highlighted"""

        def node_color(node: str) -> Union[np.ndarray, str]:
            if node_property(node):
                return color
            else:
                return self._get_color(node)

        self.draw(node_color=node_color)
