# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.

from __future__ import annotations
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
from typing import List, Tuple, Dict, Any, Callable, Union
from pgmpy.inference.ExactInference import BeliefPropagation
import networkx as nx
from core.cpd import UniformRandomCPD, FunctionCPD, DecisionDomain
import itertools
import matplotlib.cm as cm


class MACIDBase(BayesianModel):

    def __init__(self,
                 edges: List[Tuple[str, str]],
                 node_types: Dict[Union[str, int], Dict]):
        super().__init__(ebunch=edges)
        self.node_types = node_types
        # dictionary matching each agent with their decision and utility nodes eg {'A': ['U1', 'U2'], 'B': ['U3', 'U4']}
        self.utility_nodes_agent = {i: node_types[i]['U'] for i in node_types}
        self.decision_nodes_agent = {i: node_types[i]['D'] for i in node_types}
        self.all_decision_nodes: List[str] = list(set().union(*list(self.decision_nodes_agent.values())))
        self.all_utility_nodes: List[str] = list(set().union(*list(self.utility_nodes_agent.values())))
        self.agents = list(node_types.keys())   # gives a list of the MAID's agents
        self.whose_node = {}
        for agent in self.agents:
            for node in self.decision_nodes_agent[agent]:
                self.whose_node[node] = agent
            for node in self.utility_nodes_agent[agent]:
                self.whose_node[node] = agent
        assert set(self.nodes).issuperset(self.all_decision_nodes)
        assert set(self.nodes).issuperset(self.all_utility_nodes)
        self.cpds_to_add: Dict[str, TabularCPD] = {}

    def add_cpds(self, *cpds: TabularCPD) -> None:
        """
        Add the given CPDs and initiate FunctionCPDs, UniformRandomCPDs etc
        """
        for cpd in cpds:
            assert cpd.variable in self.nodes
            assert isinstance(cpd, TabularCPD)
            if isinstance(cpd, DecisionDomain) and cpd.variable not in self.all_decision_nodes:
                raise Exception(f"trying to add DecisionDomain to non-decision node {cpd.variable}")
            if isinstance(cpd, FunctionCPD) and set(cpd.evidence) != set(self.get_parents(cpd.variable)):
                raise Exception(f"parents {cpd.evidence} of {cpd} " + f"don't match graph parents \
                                {self.get_parents(cpd.variable)}")
            self.cpds_to_add[cpd.variable] = cpd

        for var in nx.topological_sort(self):
            if var in self.cpds_to_add:
                cpd_to_add = self.cpds_to_add[var]
                if hasattr(cpd_to_add, "initialize_tabular_cpd"):
                    cpd_to_add.initialize_tabular_cpd(self)
                if hasattr(cpd_to_add, "values"):  # cpd_to_add has been initialized
                    # if the state_names have changed, remember to update all descendants:
                    previous_cpd = self.get_cpds(var)
                    if previous_cpd and previous_cpd.state_names[var] != cpd_to_add.state_names[var]:
                        for descendant in nx.descendants(self, var):
                            if descendant not in self.cpds_to_add and self.get_cpds(descendant):
                                self.cpds_to_add[descendant] = self.get_cpds(descendant)
                    # add cpd to BayesianModel, and remove it from cpds_to_add
                    super().add_cpds(cpd_to_add)
                    del self.cpds_to_add[var]

    def _get_valid_order(self, nodes: List[str]) -> List[str]:
        srt = [i for i in nx.topological_sort(self) if i in nodes]
        return srt

    def impute_random_decision(self, d: str) -> None:
        """Impute a random policy to the given decision node"""
        current_cpd = self.get_cpds(d)
        if current_cpd:
            sn = current_cpd.state_names[d]
        else:
            raise Exception(f"can't figure out domain for {d}, did you forget to specify DecisionDomain?")
        self.add_cpds(UniformRandomCPD(d, sn))

    def impute_optimal_decision(self, d: str) -> None:
        """Impute an optimal policy to the given decision node"""
        self.impute_random_decision(d)
        card = self.get_cardinality(d)
        parents = self.get_parents(d)
        idx2name = self.get_cpds(d).no_to_name[d]
        state_names = self.get_cpds(d).state_names
        utility_nodes = self.utility_nodes_agent[self.whose_node[d]]
        descendant_utility_nodes = list(set(utility_nodes).intersection(nx.descendants(self, d)))
        new = self.copy()  # this "freezes" the policy so it doesn't adapt to future interventions

        @lru_cache(maxsize=1000)
        def opt_policy(*pv: tuple) -> Any:
            nonlocal descendant_utility_nodes
            context : Dict[str, Any] = {p: pv[i] for i, p in enumerate(parents)}
            eu = []
            for d_idx in range(card):
                context[d] = d_idx  # TODO should this be id2name[d_idx]?
                eu.append(new.expected_value(descendant_utility_nodes, context))
            return idx2name[np.argmax(eu)]

        self.add_cpds(FunctionCPD(d, opt_policy, parents, state_names=state_names, label="opt"))

    def impute_conditional_expectation_decision(self, d: str, y: str) -> None:
        """Imputes a policy for d = the expectation of y conditioning on d's parents"""
        parents = self.get_parents(d)
        new = self.copy()

        @lru_cache(maxsize=1000)
        def cond_exp_policy(*pv: tuple) -> float:
            context = {p: pv[i] for i, p in enumerate(parents)}
            return new.expected_value([y], context)[0]

        self.add_cpds(FunctionCPD(d, cond_exp_policy, parents, label="cond_exp({})".format(y)))

    def mechanism_graph(self) -> MACIDBase:
        """Returns a mechanism graph with an extra parent node+"mec" for each node"""
        mg = self.copy_without_cpds()
        for node in self.nodes:
            mg.add_node(node + "mec")
            mg.add_edge(node + "mec", node)
        return mg

    def _query(self, query: List[str], context: Dict[str, Any], intervention: dict = None) -> BeliefPropagation:
        """Return P(query|context, do(intervention))*P(context | do(intervention)).

        Use factor.normalize to get p(query|context, do(intervention)).
        Use context={} to get P(query). """

        # check that graph is sufficiently instantiated to determine query,
        # in particular that strategically relevant decisions have a policy specified
        mech_graph = self.mechanism_graph()
        for decision in self.all_decision_nodes:
            for query_node in query:
                if mech_graph.is_active_trail(decision + "mec", query_node, observed=list(context.keys())):
                    cpd = self.get_cpds(decision)
                    if not cpd:
                        raise Exception(f"no DecisionDomain specified for {decision}")
                    elif isinstance(cpd, DecisionDomain):
                        raise Exception(f"query {query}|{context} depends on {decision}, but no policy imputed for it")

        # query fails if graph includes nodes not in moralized graph, so we remove them
        # cid = self.copy()
        # mm = MarkovModel(cid.moralize().edges())
        # for node in self.nodes:
        #     if node not in mm.nodes:
        #         cid.remove_node(node)
        # filtered_context = {k:v for k,v in context.items() if k in mm.nodes}
        if intervention:
            cid = self.copy()
            cid.intervene(intervention)
        else:
            cid = self

        updated_state_names = {}
        for v in query:
            cpd = cid.get_cpds(v)
            updated_state_names[v] = cpd.state_names[v]

        bp = BeliefPropagation(cid)
        # factor = bp.query(query, filtered_context)
        factor = bp.query(query, context)
        factor.state_names = updated_state_names  # factor sometimes gets state_names wrong...
        return factor

    def intervene(self, intervention: dict) -> None:
        """Given a dictionary of interventions, replace the CPDs for the relevant nodes.

        Soft interventions can be achieved by using add_cpds directly.
        """
        cpds = []
        for variable, value in intervention.items():
            cpds.append(FunctionCPD(variable, lambda *x: value,
                                    evidence=self.get_parents(variable)))

        self.add_cpds(*cpds)

    def expected_value(self, variables: List[str], context: dict, intervene: dict = None) -> List[float]:
        """Compute the expected value of a real-valued variable for a given context,
        under an optional intervention
        """
        factor = self._query(variables, context, intervention=intervene)
        factor.normalize()  # make probs add to one

        ev = np.array([0.0 for _ in factor.variables])
        for idx, prob in np.ndenumerate(factor.values):
            # idx contains the information about the value each variable takes
            # we use state_names to convert index into the actual value of the variable
            ev += prob * np.array([factor.state_names[variable][idx[var_idx]]
                                   for var_idx, variable in enumerate(factor.variables)])
            if np.isnan(ev).any():
                raise Exception("query {} | {} generated Nan from idx: {}, prob: {}, \
                                consider imputing a random decision".format(variables, context, idx, prob))
        return ev.tolist()  # type: ignore

    def expected_utility(self, context: Dict["str", "Any"],
                         intervene: dict = None, agent: Union[str, int] = 0) -> float:
        """Compute the expected utility for a given context and optional intervention

        For example:
        cid = get_minimal_cid()
        out = self.expected_utility({'D':1}) #TODO: give example that uses context
        """
        return sum(self.expected_value(self.utility_nodes_agent[agent],
                                       context, intervene=intervene))

    def copy_without_cpds(self) -> MACIDBase:
        """copy the MACIDBase object without its CPDs"""
        return MACIDBase(self.edges(),
                         {agent: {'D': self.decision_nodes_agent[agent],
                                  'U': self.utility_nodes_agent[agent]}
                          for agent in self.agents})

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
        colors = cm.rainbow(np.linspace(0, 1, len(self.agents)))
        if node in self.all_decision_nodes or node in self.all_utility_nodes:
            return colors[[self.agents.index(self.whose_node[node])]]  # type: ignore
        else:
            return 'lightgray'  # chance node

    def _get_shape(self, node: str) -> str:
        if node in self.all_decision_nodes:
            return 's'
        elif node in self.all_utility_nodes:
            return 'D'
        else:
            return 'o'

    def _get_label(self, node: str) -> Any:
        cpd = self.get_cpds(node)
        if hasattr(cpd, "label"):
            return cpd.label
        elif hasattr(cpd, "__name__"):
            return cpd.__name__
        else:
            return ""

    def draw(self,
             node_color: Callable[[str], str] = None,
             node_shape: Callable[[str], str] = None,
             node_label: Callable[[str], str] = None) -> None:
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
            nx.draw_networkx(self.to_directed().subgraph([node]), pos=layout, node_size=800, arrowsize=20,
                             node_color=color(node),
                             node_shape=shape(node))
        plt.show()

    def draw_property(self, node_property: Callable[[str], bool], color: str = 'red') -> None:
        """Draw a CID with nodes satisfying property highlighted"""

        def node_color(node: str) -> Any:
            if node_property(node):
                return color
            else:
                return self._get_color(node)

        self.draw(node_color=node_color)

    def is_s_reachable(self, d1: str, d2: str) -> bool:
        """
        Determine whether 'D2' is s-reachable from 'D1' (Koller and Milch, 2001)

        A node D2 is s-reachable from a node D1 in a MACID M if there is some utility node U ∈ U_D1 ∩ Desc(D1)
        such that if a new parent D2' were added to D2, there would be an active path in M from
        D2′ to U given Pa(D)∪{D}, where a path is active in a MAID if it is active in the same graph, viewed as a BN.

        """
        mg = self.mechanism_graph()
        agent = mg.whose_node[d1]
        agent_utilities = mg.utility_nodes_agent[agent]
        descended_agent_utilities = [util for util in agent_utilities if util in nx.descendants(mg, d1)]
        con_nodes = [d1] + mg.get_parents(d1)
        s_reachable = any([mg.is_active_trail(d2 + "mec", u_node, con_nodes) for u_node in descended_agent_utilities])
        return s_reachable

    def is_r_reachable(self, decision: str, node: str) -> bool:
        """
        Determine whether node is r-reachable from decision in the (MA)CID.
        - A node 𝑉 is r-reachable from a decision 𝐷 ∈ 𝑫^𝑖 in a MAID, M = (𝑵, 𝑽, 𝑬),
        if a newly added parent 𝑉ˆ of 𝑉 satisfies 𝑉ˆ ̸⊥ 𝑼^𝑖 ∩ Desc_𝐷 | Fa_𝐷 .
        - If a node V is r-reachable from a decision D that means D strategically or probabilisticaly relies on V.
        """
        mg = self.mechanism_graph()
        agent = mg.whose_node[decision]
        agent_utilities = mg.utility_nodes_agent[agent]
        rel_agent_utilities = [util for util in agent_utilities if util in nx.descendants(mg, decision)]
        con_nodes = [decision] + mg.get_parents(decision)
        r_reachable = any([mg.is_active_trail(node + "mec", u_node, con_nodes) for u_node in rel_agent_utilities])
        return r_reachable

    def relevance_graph(self, decisions: List[str] = None) -> nx.DiGraph:
        """
        Find the relevance graph for a set of decision nodes in the MACID
        see: Hammond, L., Fox, J., Everitt, T., Abate, A., & Wooldridge, M. (2021).
        Equilibrium Refinements for Multi-Agent Influence Diagrams: Theory and Practice.
        Default: the set of decision nodes is all decision nodes in the MAID.
        - an edge D -> D' exists iff D' is r-reachable from D (ie D strategically or probabilistically relies on D')
        """
        if decisions is None:
            decisions = self.all_decision_nodes
        rel_graph = nx.DiGraph()
        rel_graph.add_nodes_from(decisions)
        dec_pair_perms = list(itertools.permutations(decisions, 2))
        for dec_pair in dec_pair_perms:
            if self.is_s_reachable(dec_pair[0], dec_pair[1]):
                rel_graph.add_edge(dec_pair[0], dec_pair[1])
        return rel_graph

    def draw_relevance_graph(self, decisions: List[str] = None) -> None:
        """
        Draw the MACID's relevance graph for the given set of decision nodes.
        Default: draw the relevance graph for all decision nodes in the MACID.
        """
        if decisions is None:
            decisions = self.all_decision_nodes
        rg = self.relevance_graph(decisions)
        nx.draw_networkx(rg, node_size=400, arrowsize=20, node_color='k', font_color='w',
                         edge_color='k', with_labels=True)
        plt.show()

    def is_full_relevance_graph_acyclic(self) -> bool:
        """
        Finds whether the relevance graph for all of the decision nodes in the MACID is acyclic.
        """
        rg = self.relevance_graph()
        return nx.is_directed_acyclic_graph(rg)  # type: ignore

    def sufficient_recall(self, agent: Union[str, int] = 0) -> bool:
        """
        Finds whether an agent has sufficient recall in a (MA)CID.
        Agent i in the MAID has sufficient recall if the relevance graph
        restricted to contain only i's decision nodes is acyclic.
        """
        if agent not in self.agents:
            raise Exception(f"There is no agent {agent}, in this (MA)CID")

        rg = self.relevance_graph(self.decision_nodes_agent[agent])
        return nx.is_directed_acyclic_graph(rg)  # type: ignore

    def get_valid_acyclic_dec_node_ordering(self) -> List[str]:
        """
        Return a topological ordering (which might not be unique) of the decision nodes.
        if the relevance graph is acyclic
        """
        rg = self.relevance_graph()
        if not self.is_full_relevance_graph_acyclic():
            raise Exception('The relevance graph for this MACID is not acyclic and so \
                        no topological ordering can be immediately given.')
        else:
            return list(nx.topological_sort(rg))
