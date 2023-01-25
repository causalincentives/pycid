import itertools
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx

from pycid.core.macid_base import AgentLabel, MACIDBase


class MechanisedGraph:
    def __init__(self, cid: MACIDBase):
        super().__init__()
        self.graph = nx.DiGraph()
        self.agent_decision_mechanisms = {
            agent: [decision + "_mec" for decision in decisions] for agent, decisions in cid.agent_decisions.items()
        }

        # add agents

        # initialize the graph #
        self.graph.add_nodes_from(cid.nodes)
        self.graph.add_edges_from(cid.edges)
        for node in cid.nodes:
            self.graph.add_node(node + "_mec")
            self.graph.add_edge(node + "_mec", node)
        self.mechanism_nodes = [node + "_mec" for node in cid.nodes]

        # add edges for r-reachable mechanisms #
        # combinations of decisions and all nodes
        for (decision, node) in itertools.product(cid.decisions, cid.nodes):
            if decision == node:
                continue
            if not cid.is_r_reachable(decision, node):
                continue
            self.graph.add_edge(node + "_mec", decision + "_mec")

    def is_sufficient_recall(self, agent: Optional[AgentLabel] = None) -> bool:
        """
        Check if the specified agent or all agents have sufficient recall.
        Sufficient recall is defined as the mechanized graph restricted to mechanisms
        for decision variables belonging to the agent being acyclic.

        Parameters
        ----------
        agent : Optional[AgentLabel]
            The agent to check for sufficient recall. If not specified, all agents are checked.

        Returns
        -------
        bool
            True if the specified agent or all agents have sufficient recall, False otherwise.
        """

        if agent:
            return self._is_sufficient_recall_single(agent)
        return all(self._is_sufficient_recall_single(agent) for agent in self.agent_decision_mechanisms.keys())

    def _is_sufficient_recall_single(self, agent: AgentLabel) -> bool:
        """
        Calculates sufficient recall for a single agent.
        """

        # define a subgraph of the mechanised graph that only contains mechanisms for decision variables
        decision_mechanisms = self.agent_decision_mechanisms[agent]
        # restrict graph to just decision mechanisms
        decision_mechanised_graph = self.graph.subgraph(decision_mechanisms)
        is_dag = nx.is_directed_acyclic_graph(decision_mechanised_graph) # type: bool
        return is_dag 

    def is_sufficient_information(self) -> bool:
        """
        this checks whether the mechanised graph restricted to just mechanisms for decision variables is acyclic
        """
        # define a subgraph of the mechanised graph that only contains mechanisms for decision variables
        decision_mechanisms = list(itertools.chain(*self.agent_decision_mechanisms.values()))
        # TODO: Fix problem with inheritance
        decision_mechanised_graph = self.graph.subgraph(decision_mechanisms)
        is_dag = nx.is_directed_acyclic_graph(decision_mechanised_graph) # type: bool
        return is_dag

    def draw(self) -> None:
        """Draws full mechanised graph"""
        nx.draw(self.graph, with_labels=True)
        plt.show()
