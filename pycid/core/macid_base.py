from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import (
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
    Sequence,
    Set,
    Tuple,
    Union,
)

import matplotlib.cm as cm
import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference.ExactInference import BeliefPropagation

from pycid.core.causal_bayesian_network import CausalBayesianNetwork, Relationship
from pycid.core.cpd import DecisionDomain, Outcome, StochasticFunctionCPD
from pycid.core.relevance_graph import RelevanceGraph

AgentLabel = Hashable  # Could be a TypeVar instead but that might be overkill


class MACIDBase(CausalBayesianNetwork):
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

    class Model(CausalBayesianNetwork.Model):
        def __setitem__(self, variable: str, relationship: Union[Relationship, Sequence]) -> None:
            if isinstance(relationship, (DecisionDomain, Sequence)) and variable not in self.cbn.decisions:
                raise ValueError(f"trying to add DecisionDomain to non-decision node {variable}")
            super().__setitem__(variable, relationship)

        def to_tabular_cpd(self, variable: str, relationship: Union[Relationship, Sequence[Outcome]]) -> TabularCPD:
            if isinstance(relationship, Sequence):
                return DecisionDomain(variable, self.cbn, relationship)
            else:
                return super().to_tabular_cpd(variable, relationship)

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
        super().__init__(edges=edges)

        self.agent_decisions = {agent: list(nodes) for agent, nodes in agent_decisions.items()}
        self.agent_utilities = {agent: list(nodes) for agent, nodes in agent_utilities.items()}

        self.decision_agent = {node: agent for agent, nodes in self.agent_decisions.items() for node in nodes}
        self.utility_agent = {node: agent for agent, nodes in self.agent_utilities.items() for node in nodes}

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

    def make_decision(self, node: str, agent: AgentLabel = 0) -> None:
        """ "Turn a chance or utility node into a decision node.
        - agent specifies which agent the decision node should belong to in a MACID.
        """
        self.make_chance(node)
        self.agent_decisions[agent].append(node)
        self.decision_agent[node] = agent
        cpd = self.get_cpds(node)
        if cpd and not isinstance(cpd, DecisionDomain):
            self.add_cpds(DecisionDomain(node, self, cpd.state_names[node]))

    def make_utility(self, node: str, agent: AgentLabel = 0) -> None:
        """ "Turn a chance or utility node into a decision node."""
        self.make_chance(node)
        self.agent_utilities[agent].append(node)
        self.utility_agent[node] = agent

    def make_chance(self, node: str) -> None:
        """Turn a decision node into a chance node."""
        if node not in self.nodes():
            raise KeyError(f"The node {node} is not in the (MA)CID")
        elif node in set(self.decisions):
            agent = self.decision_agent.pop(node)
            self.agent_decisions[agent].remove(node)
        elif node in set(self.utilities):
            agent = self.utility_agent.pop(node)
            self.agent_utilities[agent].remove(node)

    def add_cpds(self, *cpds: TabularCPD, **relationships: Union[Relationship, List[Outcome]]) -> None:
        super().add_cpds(*cpds, **relationships)

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
            if outcome not in self.get_cpds(variable).domain:
                raise ValueError(f"The outcome {outcome} is not in the domain of {variable}")

        intervention = intervention or {}

        # Check that strategically relevant decisions have a policy specified
        mech_graph = MechanismGraph(self)
        for intervention_var in intervention:
            for parent in self.get_parents(intervention_var):
                mech_graph.remove_edge(parent, intervention_var)
        for decision in self.decisions:
            for query_node in query:
                if mech_graph.is_active_trail(
                    decision + "mec", query_node, observed=list(context.keys()) + list(intervention.keys())
                ):
                    cpd = self.get_cpds(decision)
                    if not cpd:
                        raise ValueError(f"no DecisionDomain specified for {decision}")
                    elif isinstance(cpd, DecisionDomain):
                        raise ValueError(
                            f"P({query}|{context}, do({intervention})) depends on {decision}, but no policy imputed"
                        )

        return super().query(query, context, intervention)

    def expected_utility(
        self, context: Dict[str, Outcome], intervention: Dict[str, Outcome] = None, agent: AgentLabel = 0
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
        Returns true if the agent has sufficient recall in the (MA)CID.

        Agent i in the (MA)CID has sufficient recall if the relevance graph
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

    def pure_decision_rules(self, decision: str) -> Iterator[StochasticFunctionCPD]:
        """Return a list of the decision rules available at the given decision"""

        domain = self.model.domain[decision]
        parents = self.get_parents(decision)
        parent_cardinalities = [self.get_cardinality(parent) for parent in parents]

        # We begin by representing each possible decision rule as a tuple of outcomes, with
        # one element for each possible decision context
        number_of_decision_contexts = int(np.product(parent_cardinalities))
        functions_as_tuples = itertools.product(domain, repeat=number_of_decision_contexts)

        def arg2idx(pv: Dict[str, Outcome]) -> int:
            """Convert a decision context into an index for the function list"""
            idx = 0
            for i, parent in enumerate(parents):
                name_to_no: Dict[Outcome, int] = self.get_cpds(parent).name_to_no[parent]
                idx += name_to_no[pv[parent]] * int(np.product(parent_cardinalities[:i]))
            assert 0 <= idx <= number_of_decision_contexts
            return idx

        for func_list in functions_as_tuples:

            def produce_function(early_eval_func_list: tuple = func_list) -> Callable:
                # using a default argument is a trick to get func_list to evaluate early
                return lambda **parent_values: early_eval_func_list[arg2idx(parent_values)]

            yield StochasticFunctionCPD(decision, produce_function(), self, domain=domain)

    def pure_policies(self, decision_nodes: Iterable[str]) -> Iterator[Tuple[StochasticFunctionCPD, ...]]:
        """
        Iterate over all of an agent's pure policies in this subgame.
        """
        possible_dec_rules = list(map(self.pure_decision_rules, decision_nodes))
        return itertools.product(*possible_dec_rules)

    def optimal_pure_policies(
        self, decisions: Iterable[str], rel_tol: float = 1e-9
    ) -> List[Tuple[StochasticFunctionCPD, ...]]:
        """Find all optimal policies for a given set of decisions.

        - All decisions must belong to the same agent.
        - rel_tol: is the relative tolerance. It is the amount of error allowed, relative to the larger
        absolute value of the two values it is comparing (the two utilities.)
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

        optimal_policies = []
        max_utility = float("-inf")
        for policy in macid.pure_policies(decisions):
            macid.add_cpds(*policy)
            expected_utility = macid.expected_utility({}, agent=agent)
            if math.isclose(expected_utility, max_utility, rel_tol=rel_tol):
                optimal_policies.append(policy)
            elif expected_utility > max_utility:
                optimal_policies = [policy]
                max_utility = expected_utility
        return optimal_policies

    def optimal_pure_decision_rules(self, decision: str) -> List[StochasticFunctionCPD]:
        """
        Return a list of all optimal pure decision rules for a given decision
        """
        return [policy[0] for policy in self.optimal_pure_policies([decision])]

    def impute_random_decision(self, d: str) -> None:
        """Impute a random policy to the given decision node"""
        try:
            domain = self.model.domain[d]
        except KeyError:
            raise ValueError(f"can't figure out domain for {d}, did you forget to specify DecisionDomain?")
        else:
            self.model[d] = StochasticFunctionCPD(
                d, lambda **pv: {outcome: 1 / len(domain) for outcome in domain}, self, domain, label="random_decision"
            )

    def impute_fully_mixed_policy_profile(self) -> None:
        """Impute a fully mixed policy profile - ie a random decision rule to all decision nodes"""
        for d in self.decisions:
            self.impute_random_decision(d)

    def remove_all_decision_rules(self) -> None:
        """Remove the decision rules from all decisions in the (MA)CID"""
        for d in self.decisions:
            self.model[d] = self.model.domain[d]

    def impute_optimal_decision(self, decision: str) -> None:
        """Impute an optimal policy to the given decision node"""
        # self.add_cpds(random.choice(self.optimal_pure_decision_rules(d)))
        self.impute_random_decision(decision)
        domain = self.model.domain[decision]
        utility_nodes = self.agent_utilities[self.decision_agent[decision]]
        descendant_utility_nodes = list(set(utility_nodes).intersection(nx.descendants(self, decision)))
        copy = self.copy()  # using a copy "freezes" the policy so it doesn't adapt to future interventions

        @lru_cache(maxsize=1000)
        def opt_policy(**parent_values: Outcome) -> Outcome:
            eu = {}
            for d in domain:
                parent_values[decision] = d
                eu[d] = sum(copy.expected_value(descendant_utility_nodes, parent_values))
            return max(eu, key=eu.get)  # type: ignore

        self.add_cpds(StochasticFunctionCPD(decision, opt_policy, self, domain=domain, label="opt"))

    def impute_conditional_expectation_decision(self, decision: str, y: str) -> None:
        """Imputes a policy for decision = the expectation of y conditioning on d's parents"""
        # TODO: Move to analyze, as this is not really a core feature?
        copy = self.copy()

        @lru_cache(maxsize=1000)
        def cond_exp_policy(**pv: Outcome) -> float:
            if y in pv:
                return pv[y]  # type: ignore
            else:
                return copy.expected_value([y], pv)[0]

        self.add_cpds(**{decision: cond_exp_policy})

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
            color: np.ndarray = colors[[agents.index(agent)]]
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
