from __future__ import annotations

import itertools
import logging
import random
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    KeysView,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference.ExactInference import BeliefPropagation
from pgmpy.models import BayesianModel

from pycid.core.cpd import DecisionDomain, FunctionCPD, ParentsNotReadyException, State, UniformRandomCPD
from pycid.core.relevance_graph import RelevanceGraph

AgentLabel = Hashable  # Could be a TypeVar instead but that might be overkill


class MACIDBase(BayesianModel):
    """Base structure of a Multi-Agent Causal Influence Diagram.

    Attributes
    ----------
    agent_decisions: The decision nodes of each agent.
        A dictionary mapping agent label => nodes labels.

    agent_utilities: The utility nodes of each agent.
        A dictionary mapping agent label => node labels.

    decision_agent: The agent owner of each decision node.
        A dictionary mapping decision node label => agent label.

    utility_agent: The agent owner of each utility node.
        A dictionary mapping utility node label => agent label.
    """

    def __init__(
        self,
        edges: Iterable[Tuple[str, str]],
        agent_decisions: Mapping[AgentLabel, Iterable[str]],
        agent_utilities: Mapping[AgentLabel, Iterable[str]],
    ):
        """Initialize a new MACIDBase instance.

        Parameters
        ----------
        edges: A set of directed edges. Each is a pair of node labels (tail, head).

        agent_decisions: The decision nodes of each agent.
            A mapping of agent label => nodes labels.

        agent_utilities: The utility nodes of each agent.
            A mapping of agent label => node labels.
        """
        super().__init__(ebunch=edges)

        self.agent_decisions = {agent: list(nodes) for agent, nodes in agent_decisions.items()}
        self.agent_utilities = {agent: list(nodes) for agent, nodes in agent_utilities.items()}

        self.decision_agent = {node: agent for agent, nodes in self.agent_decisions.items() for node in nodes}
        self.utility_agent = {node: agent for agent, nodes in self.agent_utilities.items() for node in nodes}

        self._cpds_to_add: Dict[str, TabularCPD] = {}
        self.state_names: Dict[str, State] = {}

    @property
    def decisions(self) -> KeysView[str]:
        """The set of all decision nodes"""
        return self.decision_agent.keys()

    @property
    def utilities(self) -> KeysView[str]:
        """The set of all utility nodes"""
        return self.utility_agent.keys()

    @property
    def agents(self) -> KeysView[AgentLabel]:
        """The set of all agents"""
        return self.agent_utilities.keys()

    def remove_edge(self, u: str, v: str) -> None:
        super().remove_edge(u, v)
        # remove_edge can be called during __init__ when cpds is not yet defined
        if not hasattr(self, "cpds"):
            return
        cpd = self.get_cpds(v)
        if isinstance(cpd, UniformRandomCPD):
            self.add_cpds(cpd)

    def add_edge(self, u: str, v: str) -> None:
        super().add_edge(u, v)
        # add_edge can be called during __init__ when cpds is not yet defined
        if not hasattr(self, "cpds"):
            return
        cpd = self.get_cpds(v)
        if isinstance(cpd, UniformRandomCPD):
            self.add_cpds(cpd)

    def make_decision(self, node: str, agent: AgentLabel = 0) -> None:
        """"Turn a chance or utility node into a decision node."""
        cpd = self.get_cpds(node)
        if cpd is None:
            raise ValueError(f"node {node} has not yet been assigned a domain.")
        elif isinstance(cpd, DecisionDomain):
            # Already a decision
            pass
        else:
            cpd_new = DecisionDomain(node, cpd.state_names[node])
            self.agent_decisions[agent].append(node)
            self.decision_agent[node] = agent
            self.add_cpds(cpd_new)

    def make_chance(self, node: str) -> None:
        """Turn a decision node into a chance node."""
        agent = self.decision_agent.pop(node)
        self.agent_decisions[agent].remove(node)

    def add_cpds(self, *cpds: TabularCPD) -> None:
        """
        Add the given CPDs and initialize FunctionCPDs, UniformRandomCPDs etc
        """

        # Add each cpd to self._cpds_to_add after doing some checks
        for cpd in cpds:
            assert cpd.variable in self.nodes
            assert isinstance(cpd, TabularCPD)
            if isinstance(cpd, DecisionDomain) and cpd.variable not in self.decisions:
                raise ValueError(f"trying to add DecisionDomain to non-decision node {cpd.variable}")
            if hasattr(cpd, "check_parents"):
                cpd.check_parents(self)
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

    def query(
        self, query: Iterable[str], context: Dict[str, Any], intervention: Dict[str, Any] = None
    ) -> BeliefPropagation:
        """Return P(query|context, do(intervention))*P(context | do(intervention)).

        Use factor.normalize to get p(query|context, do(intervention)).
        Use context={} to get P(query).

        Parameters
        ----------
        query: A set of nodes to query.

        context: Node values to condition upon. A dictionary mapping of node => value.

        intervention: Interventions to apply. A dictionary mapping node => value.
        """

        for variable, value in context.items():
            if value not in self.state_names[variable]:
                raise ValueError(f"The value {value} is not in the domain of {variable}")

        if intervention is None:
            intervention = {}

        # First apply the intervention, if any
        if intervention:
            cid = self.copy()
            cid.intervene(intervention)
        else:
            cid = self

        # Check that strategically relevant decisions have a policy specified
        mech_graph = MechanismGraph(cid)
        for decision in cid.decisions:
            for query_node in query:
                if mech_graph.is_active_trail(
                    decision + "mec", query_node, observed=list(context.keys()) + list(intervention.keys())
                ):
                    cpd = cid.get_cpds(decision)
                    if not cpd:
                        raise ValueError(f"no DecisionDomain specified for {decision}")
                    elif isinstance(cpd, DecisionDomain):
                        raise ValueError(
                            f"P({query}|{context}, do({intervention})) depends on {decision}, but no policy imputed"
                        )

        # query fails if graph includes nodes not in moralized graph, so we remove them
        # cid = self.copy()
        # mm = MarkovModel(cid.moralize().edges())
        # for node in self.nodes:
        #     if node not in mm.nodes:
        #         cid.remove_node(node)
        # filtered_context = {k:v for k,v in context.items() if k in mm.nodes}

        updated_state_names = {}
        for v in query:
            cpd = cid.get_cpds(v)
            updated_state_names[v] = cpd.state_names[v]

        # Make a copy of self and revise the context without state_names (to handle a pgmpy bug),
        copy = cid.copy_without_cpds()
        for cpd in cid.cpds:
            evidence = cpd.variables[1:] if len(cpd.variables) > 1 else None
            evidence_card = cpd.cardinality[1:] if len(cpd.variables) > 1 else None
            copy.add_cpds(TabularCPD(cpd.variable, cpd.variable_card, cpd.get_values(), evidence, evidence_card))
        revised_context = {  # state_names are switched to their state number
            variable: self.get_cpds(variable).name_to_no[variable][value] for variable, value in context.items()
        }

        bp = BeliefPropagation(copy)
        # TODO: check for probability 0 queries

        with np.errstate(invalid="ignore"):  # Suppress numpy warnings for 0/0
            factor = bp.query(query, revised_context, show_progress=False)
        factor.state_names = updated_state_names  # reintroduce the state_names
        return factor

    def intervene(self, intervention: Dict[str, Any]) -> None:
        """Given a dictionary of interventions, replace the CPDs for the relevant nodes.

        Soft interventions can be achieved by using self.add_cpds() directly.

        Parameters
        ----------
        intervention: Interventions to apply. A dictionary mapping node => value.
        """
        for variable in intervention:
            for p in self.get_parents(variable):  # remove ingoing edges
                self.remove_edge(p, variable)
            self.add_cpds(FunctionCPD(variable, lambda: intervention[variable], domain=self.state_names[variable]))

    def expected_value(
        self,
        variables: Iterable[str],
        context: Dict[str, Any],
        intervention: Dict[str, Any] = None,
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

    def expected_utility(
        self, context: Dict[str, Any], intervention: Dict[str, Any] = None, agent: AgentLabel = 0
    ) -> float:
        """Compute the expected utility of an agent for a given context and optional intervention

        For example:
        cid = get_minimal_cid()
        out = self.expected_utility({'D':1}) #TODO: give example that uses context

        Parameters
        ----------
        context: Node values to condition upon. A dictionary mapping of node => value.

        intervention: Interventions to apply. A dictionary mapping node => value.

        agent: Evaluate the utility of this agent.
        """
        return sum(self.expected_value(self.agent_utilities[agent], context, intervention=intervention))

    def get_valid_order(self, nodes: Optional[Iterable[str]] = None) -> List[str]:
        """Get a topological order of the specified set of nodes (this may not be unique).

        By default, a topological ordering of the decision nodes is given"""
        if not nx.is_directed_acyclic_graph(self):
            raise ValueError("A topological ordering of nodes can only be returned if the (MA)CID is acyclic")

        if nodes is None:
            nodes = self.decisions
        else:
            nodes = set(nodes)
            for node in nodes:
                if node not in self.nodes:
                    raise KeyError(f"{node} is not in the (MA)CID.")

        srt = [node for node in nx.topological_sort(self) if node in nodes]
        return srt

    def is_s_reachable(self, d1: Union[str, Iterable[str]], d2: Union[str, Iterable[str]]) -> bool:
        """
        Determine whether 'D2' is s-reachable from 'D1' (Koller and Milch, 2001)

        A node D2 is s-reachable from a node D1 in a MACID M if there is some utility node U ∈ U_D1 ∩ Desc(D1)
        such that if a new parent D2' were added to D2, there would be an active path in M from
        D2′ to U given Pa(D)∪{D}, where a path is active in a MAID if it is active in the same graph, viewed as a BN.

        """
        assert d2 in self.decisions
        return self.is_r_reachable(d1, d2)

    def is_r_reachable(self, decisions: Union[str, Iterable[str]], nodes: Union[str, Iterable[str]]) -> bool:
        """
        Determine whether (a set of) node(s) is r-reachable from decision in the (MA)CID.

        - A node 𝑉 is r-reachable from a decision 𝐷 ∈ 𝑫^𝑖 in a MAID, M = (𝑵, 𝑽, 𝑬),
        if a newly added parent 𝑉ˆ of 𝑉 satisfies 𝑉ˆ ̸⊥ 𝑼^𝑖 ∩ Desc_𝐷 | Fa_𝐷 .
        - If a node V is r-reachable from a decision D that means D strategically or probabilisticaly relies on V.
        """
        if isinstance(decisions, str):
            decisions = [decisions]
        if isinstance(nodes, str):
            nodes = [nodes]
        mg = MechanismGraph(self)
        for decision in decisions:
            for node in nodes:
                con_nodes = [decision] + self.get_parents(decision)
                agent_utilities = self.agent_utilities[self.decision_agent[decision]]
                for utility in set(agent_utilities).intersection(nx.descendants(self, decision)):
                    if mg.is_active_trail(node + "mec", utility, con_nodes):
                        return True
        return False

    def sufficient_recall(self, agent: Optional[AgentLabel] = None) -> bool:
        """
        Finds whether a (MA)CID has sufficient recall.

        Agent i in the MAID has sufficient recall if the relevance graph
        restricted to contain only i's decision nodes is acyclic.

        If an agent is specified, sufficient recall is checked only for that agent.
        Otherwise, the check is done for all agents.
        """
        if agent is None:
            agents: Collection = self.agents
        elif agent not in self.agents:
            raise ValueError(f"There is no agent {agent}, in this (MA)CID")
        else:
            agents = [agent]

        for a in agents:
            rg = RelevanceGraph(self, self.agent_decisions[a])
            if not rg.is_acyclic():
                return False
        return True

    def pure_decision_rules(self, decision: str) -> List[FunctionCPD]:
        """Return a list of the decision rules available at the given decision"""

        cpd: TabularCPD = self.get_cpds(decision)
        evidence_card = cpd.cardinality[1:]
        parents = cpd.variables[1:]
        domain = cpd.state_names[decision]

        parent_values = []
        for p in parents:
            p_cpd = self.get_cpds(p)
            parent_values.append(p_cpd.state_names[p])
        pv_list = list(itertools.product(*parent_values))
        possible_arguments = [{p.lower(): pv[i] for i, p in enumerate(parents)} for pv in pv_list]

        # We begin by representing each possible decision as a list values, with length
        # equal the number of decision contexts
        functions_as_lists = list(itertools.product(domain, repeat=np.product(len(possible_arguments))))

        def arg2idx(pv: Dict[str, Any]) -> int:
            """Convert a decision context into an index for the function list"""
            idx = 0
            for i, parent in enumerate(parents):
                name_to_no: Dict[Any, int] = self.get_cpds(parent).name_to_no[parent]
                idx += name_to_no[pv[parent.lower()]] * np.product(evidence_card[:i])
            assert 0 <= idx <= len(functions_as_lists)
            return idx

        function_cpds: List[FunctionCPD] = []
        for func_list in functions_as_lists:

            def produce_function(early_eval_func_list: tuple = func_list) -> Callable:
                return lambda **parent_values: early_eval_func_list[arg2idx(parent_values)]

            function_cpds.append(FunctionCPD(decision, produce_function(), domain=cpd.state_names[cpd.variable]))
        return function_cpds

    def pure_strategies(self, decision_nodes: Iterable[str]) -> Iterator[Tuple[FunctionCPD, ...]]:
        """
        Iterate over all of an agent's pure policies in this subgame.
        """
        possible_dec_rules = list(map(self.pure_decision_rules, decision_nodes))
        return itertools.product(*possible_dec_rules)

    def optimal_pure_strategies(self, decisions: Iterable[str], eps: float = 1e-8) -> List[Tuple[FunctionCPD, ...]]:
        """Find all optimal strategies for a given set of decisions.

        - All decisions must belong to the same agent.
        - eps is the margin of error when comparing utilities for equality.
        """
        if not decisions:
            return []
        decisions = set(decisions)
        try:
            (agent,) = {self.decision_agent[d] for d in decisions}
        except ValueError:
            raise ValueError("Decisions not all from the same agent")

        macid = self.copy()
        for d in macid.decisions:
            if (
                isinstance(macid.get_cpds(d), DecisionDomain)
                and not macid.is_s_reachable(decisions, d)
                and d not in decisions
            ):
                macid.impute_random_decision(d)

        optimal_strategies = []
        max_utility = float("-inf")
        for strategy in macid.pure_strategies(decisions):
            macid.add_cpds(*strategy)
            expected_utility = macid.expected_utility({}, agent=agent)
            if abs(expected_utility - max_utility) < eps:
                optimal_strategies.append(strategy)
            elif expected_utility > max_utility:
                optimal_strategies = [strategy]
                max_utility = expected_utility
        return optimal_strategies

    def optimal_pure_decision_rules(self, decision: str) -> List[FunctionCPD]:
        """
        Return a list of all optimal decision rules for a given decision
        """
        return [strategy[0] for strategy in self.optimal_pure_strategies([decision])]

    def impute_random_decision(self, d: str) -> None:
        """Impute a random policy to the given decision node"""
        current_cpd = self.get_cpds(d)
        if current_cpd:
            sn = current_cpd.state_names[d]
        else:
            raise ValueError(f"can't figure out domain for {d}, did you forget to specify DecisionDomain?")
        self.add_cpds(UniformRandomCPD(d, sn))

    def impute_fully_mixed_policy_profile(self) -> None:
        """Impute a fully mixed policy profile - ie a random decision rule to all decision nodes"""
        for d in self.decisions:
            self.impute_random_decision(d)

    def impute_optimal_decision(self, d: str) -> None:
        """Impute an optimal policy to the given decision node"""
        self.add_cpds(random.choice(self.optimal_pure_decision_rules(d)))
        # self.impute_random_decision(d)
        # cpd = self.get_cpds(d)
        # parents = cpd.variables[1:]
        # idx2name = cpd.no_to_name[d]
        # utility_nodes = self.agent_utilities[self.decision_agent[d]]
        # descendant_utility_nodes = list(set(utility_nodes).intersection(nx.descendants(self, d)))
        # new = self.copy()  # using a copy "freezes" the policy so it doesn't adapt to future interventions
        #
        # @lru_cache(maxsize=1000)
        # def opt_policy(*parent_values: tuple) -> Any:
        #     nonlocal descendant_utility_nodes
        #     context: Dict[str, Any] = {parents[i]: parent_values[i] for i in range(len(parents))}
        #     eu = []
        #     for d_idx in range(new.get_cardinality(d)):
        #         context[d] = idx2name[d_idx]
        #         eu.append(sum(new.expected_value(descendant_utility_nodes, context)))
        #     return idx2name[np.argmax(eu)]
        #
        # self.add_cpds(FunctionCPD(d, opt_policy, parents, state_names=cpd.state_names, label="opt"))

    def impute_conditional_expectation_decision(self, d: str, y: str) -> None:
        """Imputes a policy for d = the expectation of y conditioning on d's parents"""
        # TODO: Move to analyze, as this is not really a core feature?
        parents = self.get_parents(d)
        new = self.copy()

        @lru_cache(maxsize=1000)
        def cond_exp_policy(**pv: tuple) -> float:
            context = {p: pv[p.lower()] for p in parents}
            return new.expected_value([y], context)[0]

        self.add_cpds(FunctionCPD(d, cond_exp_policy, label="cond_exp({})".format(y)))

    # Wrapper around DAG.active_trail_nodes to accept arbitrary iterables for observed.
    # Really, DAG.active_trail_nodes should accept Sets, especially since it does
    # inefficient membership checks on observed as a list.
    def active_trail_nodes(
        self, variables: Union[str, List[str], Tuple[str, ...]], observed: Optional[Iterable[str]] = None
    ) -> Dict[str, Set[str]]:
        return super().active_trail_nodes(variables, list(observed))  # type: ignore

    def copy_without_cpds(self) -> MACIDBase:
        """copy the MACIDBase object without its CPDs"""
        return MACIDBase(
            edges=self.edges(),
            agent_decisions=self.agent_decisions,
            agent_utilities=self.agent_utilities,
        )

    def copy(self) -> MACIDBase:
        """copy the MACIDBase object"""
        model_copy = self.copy_without_cpds()
        if self.cpds:
            model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        return model_copy

    def _get_color(self, node: str) -> Union[np.ndarray, str]:
        """
        Assign a unique colour to each new agent's decision and utility nodes
        """
        agents = list(self.agents)
        colors = cm.rainbow(np.linspace(0, 1, len(agents)))
        try:
            agent = self.decision_agent[node]
        except KeyError:
            try:
                agent = self.utility_agent[node]
            except KeyError:
                agent = None
        if agent is not None:
            color: np.ndarray = colors[agents.index(agent)]
            return color
        else:
            return "lightgray"  # chance node

    def _get_shape(self, node: str) -> str:
        if node in self.decisions:
            return "s"
        elif node in self.utilities:
            return "D"
        else:
            return "o"

    def _get_label(self, node: str) -> Any:
        cpd = self.get_cpds(node)
        if hasattr(cpd, "label"):
            return cpd.label
        elif hasattr(cpd, "__name__"):
            return cpd.__name__
        else:
            return ""

    def draw(
        self,
        node_color: Callable[[str], str] = None,
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
        """Draw a CID with the nodes satisfying node_property highlighted"""

        def node_color(node: str) -> Any:
            if node_property(node):
                return color
            else:
                return self._get_color(node)

        self.draw(node_color=node_color)


class MechanismGraph(MACIDBase):
    """A mechanism graph has an extra parent node+"mec" for each node"""

    def __init__(self, cid: MACIDBase):
        super().__init__(
            edges=cid.edges(),
            agent_decisions=cid.agent_decisions,
            agent_utilities=cid.agent_utilities,
        )

        for node in cid.nodes:
            if node[:-3] == "mec":
                raise ValueError("can't create a mechanism graph when node {node} already ends with mec")
            self.add_node(node + "mec")
            self.add_edge(node + "mec", node)
        # TODO: adapt the parameterization from cid as well
