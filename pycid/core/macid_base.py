from __future__ import annotations

import itertools
from functools import lru_cache
from typing import (
    Callable,
    Collection,
    Dict,
    Hashable,
    Iterable,
    KeysView,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    Iterator,
)

import matplotlib.cm as cm
import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference.ExactInference import BeliefPropagation

from pycid.core.causal_bayesian_network import CausalBayesianNetwork
from pycid.core.cpd import DecisionDomain, FunctionCPD, Outcome, UniformRandomCPD
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
        """"Turn a chance or utility node into a decision node."""
        self.make_chance(node)
        self.agent_decisions[agent].append(node)
        self.decision_agent[node] = agent
        cpd = self.get_cpds(node)
        if cpd and not isinstance(cpd, DecisionDomain):
            self.add_cpds(DecisionDomain(node, cpd.state_names[node]))

    def make_utility(self, node: str, agent: AgentLabel = 0) -> None:
        """"Turn a chance or utility node into a decision node."""
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

    def add_cpds(self, *cpds: TabularCPD) -> None:
        """
        Add the given CPDs and initialize them.

        A CPD can be a TabularCPD, FunctionCPD, UniformRandomCPD, DecisionDomain, or a StochasticFunctionCPD.
        """

        # For MACIDs and CIDs we need to do an extra check
        for cpd in cpds:
            if isinstance(cpd, DecisionDomain) and cpd.variable not in self.decisions:
                raise ValueError(f"trying to add DecisionDomain to non-decision node {cpd.variable}")
        super().add_cpds(*cpds)

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

    def pure_decision_rules(self, decision: str) -> Iterator[FunctionCPD]:
        """Return a list of the decision rules available at the given decision"""

        domain = self.get_cpds(decision).state_names[decision]
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
                idx += name_to_no[pv[parent.lower()]] * int(np.product(parent_cardinalities[:i]))
            assert 0 <= idx <= number_of_decision_contexts
            return idx

        for func_list in functions_as_tuples:

            def produce_function(early_eval_func_list: tuple = func_list) -> Callable:
                # using a default argument is a trick to get func_list to evaluate early
                return lambda **parent_values: early_eval_func_list[arg2idx(parent_values)]

            yield FunctionCPD(decision, produce_function(), domain=domain)

    def pure_policies(self, decision_nodes: Iterable[str]) -> Iterator[Tuple[FunctionCPD, ...]]:
        """
        Iterate over all of an agent's pure policies in this subgame.
        """
        possible_dec_rules = list(map(self.pure_decision_rules, decision_nodes))
        return itertools.product(*possible_dec_rules)

    def optimal_pure_policies(self, decisions: Iterable[str], eps: float = 1e-8) -> List[Tuple[FunctionCPD, ...]]:
        """Find all optimal policies for a given set of decisions.

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

        optimal_policies = []
        max_utility = float("-inf")
        for policy in macid.pure_policies(decisions):
            macid.add_cpds(*policy)
            expected_utility = macid.expected_utility({}, agent=agent)
            if abs(expected_utility - max_utility) < eps:
                optimal_policies.append(policy)
            elif expected_utility > max_utility:
                optimal_policies = [policy]
                max_utility = expected_utility
        return optimal_policies

    def optimal_pure_decision_rules(self, decision: str) -> List[FunctionCPD]:
        """
        Return a list of all optimal decision rules for a given decision
        """
        return [policy[0] for policy in self.optimal_pure_policies([decision])]

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

    def remove_all_decision_rules(self) -> None:
        """ Remove the decision rules from all decisions in the (MA)CID"""
        for d in self.decisions:
            cpd = self.get_cpds(d)
            if cpd is None:
                raise ValueError(f"decision {d} has not yet been assigned a domain.")
            elif isinstance(cpd, DecisionDomain):
                continue
            else:
                self.add_cpds(DecisionDomain(d, cpd.domain))

    def impute_optimal_decision(self, d: str) -> None:
        """Impute an optimal policy to the given decision node"""
        # self.add_cpds(random.choice(self.optimal_pure_decision_rules(d)))
        self.impute_random_decision(d)
        cpd = self.get_cpds(d)
        parents = cpd.variables[1:]
        idx2name = cpd.no_to_name[d]
        utility_nodes = self.agent_utilities[self.decision_agent[d]]
        descendant_utility_nodes = list(set(utility_nodes).intersection(nx.descendants(self, d)))
        new = self.copy()  # using a copy "freezes" the policy so it doesn't adapt to future interventions

        @lru_cache(maxsize=1000)
        def opt_policy(**parent_values: Outcome) -> Outcome:
            nonlocal descendant_utility_nodes
            context: Dict[str, Outcome] = {p: parent_values[p.lower()] for p in parents}
            eu = []
            for d_idx in range(new.get_cardinality(d)):
                context[d] = idx2name[d_idx]
                eu.append(sum(new.expected_value(descendant_utility_nodes, context)))
            return idx2name[np.argmax(eu)]

        self.add_cpds(FunctionCPD(d, opt_policy, domain=cpd.state_names[d], label="opt"))

    def impute_conditional_expectation_decision(self, d: str, y: str) -> None:
        """Imputes a policy for d = the expectation of y conditioning on d's parents"""
        # TODO: Move to analyze, as this is not really a core feature?
        parents: List[str] = self.get_parents(d)
        new = self.copy()

        @lru_cache(maxsize=1000)
        def cond_exp_policy(**pv: Outcome) -> float:
            if y.lower() in pv:
                return pv[y.lower()]  # type: ignore
            else:
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
