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



class MACIDBase(BayesianModel):

    def __init__(self,
                 edges: List[Tuple[str, str]],
                 node_types: Dict[Union[str, int], Dict]):
        super().__init__(ebunch=edges)
        self.node_types = node_types
        self.utility_nodes_agent = {i: node_types[i]['U'] for i in node_types}     # this gives a dictionary matching each agent with their decision and utility nodes
        self.decision_nodes_agent = {i: node_types[i]['D'] for i in node_types}     # eg {'A': ['U1', 'U2'], 'B': ['U3', 'U4']}
        self.all_decision_nodes = list(set().union(*list(self.decision_nodes_agent.values())))
        self.all_utility_nodes = list(set().union(*list(self.utility_nodes_agent.values())))
        self.agents = list(node_types.keys())   # gives a list of the MAID's agents
        self.whose_node = {}
        for agent in self.agents:
            for node in self.decision_nodes_agent[agent]:
                self.whose_node[node] = agent
            for node in self.utility_nodes_agent[agent]:
                self.whose_node[node] = agent
        assert set(self.nodes).issuperset(self.all_decision_nodes)
        assert set(self.nodes).issuperset(self.all_utility_nodes)
        self.cpds_to_add = {}

    def add_cpds(self, *cpds: TabularCPD, update_all: bool = True) -> None:
        """Add the given CPDs and initiate NullCPDs and FunctionCPDs

        The update_all option recomputes the state_names and matrices for all CPDs in the graph.
        It can be set to false to save time, if the added CPD(s) have identical state_names to
        the ones they are replacing.
        """
        # TODO: get rid of update_all, by instead adding any children of a node whose state_names
        #       have changed to cpds_to_add
        if update_all:
            for cpd in self.cpds:
                self.cpds_to_add[cpd.variable] = cpd
        for cpd in cpds:
            assert cpd.variable in self.nodes
            self.cpds_to_add[cpd.variable] = cpd

        for var in nx.topological_sort(self):
            if var in self.cpds_to_add:
                cpd = self.cpds_to_add[var]
                if hasattr(cpd, "initialize_tabular_cpd"):
                    cpd.initialize_tabular_cpd(self)
                if hasattr(cpd, "values"):
                    super().add_cpds(cpd)
                    del self.cpds_to_add[var]

    def _get_valid_order(self, nodes: List[str]):
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

    def impute_optimal_decision(self, d: str, agent: int = 0) -> None:
        """Impute an optimal policy to the given decision node"""
        self.impute_random_decision(d)
        card = self.get_cardinality(d)
        parents = self.get_parents(d)
        idx2name = self.get_cpds(d).no_to_name[d]
        state_names = self.get_cpds(d).state_names
        new = self.copy()  # this "freezes" the policy so it doesn't adapt to future interventions

        @lru_cache(maxsize=1000)
        def opt_policy(*pv: tuple):
            context = {p: pv[i] for i, p in enumerate(parents)}
            eu = []
            for d_idx in range(card):
                context[d] = d_idx
                eu.append(new.expected_utility(context, agent=agent))
            return idx2name[np.argmax(eu)]

        self.add_cpds(FunctionCPD(d, opt_policy, parents, state_names=state_names, label="opt"),
                      update_all=False)

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
            mg.add_node(node+"mec")
            mg.add_edge(node+"mec", node)
        return mg

    def _query(self, query: List[str], context: Dict[str, Any], intervention: dict = None):
        """Return P(query|context, do(intervention))*P(context | do(intervention)).

        Use factor.normalize to get p(query|context, do(intervention)).
        Use context={} to get P(query). """

        # check that graph is sufficiently instantiated to determine query,
        # in particular that strategically relevant decisions have a policy specified
        mech_graph = self.mechanism_graph()
        for decision in self.all_decision_nodes:
            for query_node in query:
                if mech_graph.is_active_trail(decision+"mec", query_node, observed=list(context.keys())):
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
        return ev.tolist()

    def expected_utility(self, context: Dict["str", "Any"],
                         intervene: dict = None, agent: Union[str, int] = 0) -> float:
        """Compute the expected utility for a given context and optional intervention

        For example:
        cid = get_minimal_cid()
        out = self.expected_utility({'D':1}) #TODO: give example that uses context"""
        return sum(self.expected_value(self.utility_nodes_agent[agent],
                                       context, intervene=intervene))

    def copy_without_cpds(self):
        return MACIDBase(self.edges(),
                         {agent: {'D': self.decision_nodes_agent[agent],
                                  'U': self.utility_nodes_agent[agent]}
                          for agent in self.agents})

    def copy(self) -> MACIDBase:
        model_copy = self.copy_without_cpds()
        if self.cpds:
            model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        return model_copy

    def _get_color(self, node: str) -> str:
        if node in self.all_decision_nodes:
            return 'lightblue'
        elif node in self.all_utility_nodes:
            return 'yellow'
        else:
            return 'lightgray'

    def _get_shape(self, node: str) -> str:
        if node in self.all_decision_nodes:
            return 's'
        elif node in self.all_utility_nodes:
            return 'D'
        else:
            return 'o'

    def _get_label(self, node: str) -> str:
        cpd = self.get_cpds(node)
        if hasattr(cpd, "label"):
            return cpd.label
        else:
            return ""

    def draw(self,
             node_color: Callable[[str], str] = None,
             node_shape: Callable[[str], str] = None,
             node_label: Callable[[str], str] = None):
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

    def draw_property(self, node_property: Callable[[str], bool], color='red'):
        """Draw a CID with nodes satisfying property highlighted"""

        def node_color(node: str) -> str:
            if node_property(node):
                return color
            else:
                return self._get_color(node)

        self.draw(node_color=node_color)

    def is_s_reachable(self, d1: str, d2: str) -> bool:
        """ 
        Determine whether 'D2' is s-reachable from 'D1' (Koller and Milch, 2001)
        
        A node D2 is s-reachable from a node D1 in a MACID M if there is some utility node U ∈ U_D
        such that if a new parent D2' were added to D2, there would be an active path in M from
        D2′ to U given Pa(D)∪{D}, where a path is active in a MAID if it is active in the same graph, viewed as a BN.

        """
        mg = self.mechanism_graph()
        agent = mg.whose_node[d1]
        agent_utilities = mg.utility_nodes_agent[agent]
        con_nodes = [d1] + mg.get_parents(d1) 
        is_active_trail = any([mg.is_active_trail(d2+"mec", u_node, con_nodes) for u_node in agent_utilities])
        return is_active_trail

    def strategic_rel_graph(self) -> nx.DiGraph:
        """
        Find the strategic relevance graph of the MAID
        - an edge D -> D' exists iff D' is s-reachable from D
        """
        G = nx.DiGraph()
        dec_pair_perms = list(itertools.permutations(self.all_decision_nodes, 2))
        for dec_pair in dec_pair_perms:
            if self.is_s_reachable(dec_pair[0], dec_pair[1]):
                G.add_edge(dec_pair[0], dec_pair[1])
        return G

    def draw_strategic_rel_graph(self) -> None:
        """
        Draw the MACID's strategic relevance graph
        """
        rg = self.strategic_rel_graph()
        nx.draw_networkx(rg, node_size=400, arrowsize=20, node_color='k', font_color='w', edge_color='k', with_labels=True)
        plt.figure()
        plt.draw()
        

    def is_strategically_acyclic(self) -> bool:
        """
        Find whether the MACID has an acyclic strategic relevance graph.
        """
        rg = self.strategic_rel_graph()
        return nx.is_directed_acyclic_graph(rg)
        

    def get_valid_acyclic_dec_node_ordering(self) -> List[str]:
        """
        Return a topological ordering (which might not be unique) of the decision nodes 
        if the strategic relevance graph is acyclic

        """
        rg = self.strategic_rel_graph()
        if not self.is_strategically_acyclic():
            raise Exception('The strategic relevance graph for this MACID is not acyclic and so \
                        no topological ordering can be immediately given.')
        else:
            return list(nx.topological_sort(rg))

    