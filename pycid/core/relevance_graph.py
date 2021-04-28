from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from pycid.core.macid_base import MACIDBase


class RelevanceGraph(nx.DiGraph):
    """
    The relevance graph for a set of decision nodes in the (MA)CID
    see: Hammond, L., Fox, J., Everitt, T., Abate, A., & Wooldridge, M. (2021).
    Equilibrium Refinements for Multi-Agent Influence Diagrams: Theory and Practice.
    Default: the set of decision nodes is all decision nodes in the MAID.
    - an edge D -> D' exists iff D' is r-reachable from D (ie D strategically or probabilistically relies on D')
    """

    def __init__(self, cid: MACIDBase, decisions: Iterable[str] = None):
        super().__init__()
        if decisions is None:
            decisions = cid.decisions
        self.add_nodes_from(decisions)
        dec_pair_perms = list(itertools.permutations(decisions, 2))
        for dec_pair in dec_pair_perms:
            if cid.is_s_reachable(dec_pair[0], dec_pair[1]):
                self.add_edge(dec_pair[0], dec_pair[1])

    def is_acyclic(self) -> bool:
        """
        Find whether the relevance graph for all of the decision nodes in the MACID is acyclic.
        """
        return nx.is_directed_acyclic_graph(self)  # type: ignore

    def get_sccs(self) -> List[set]:
        """
        Return a list with the maximal strongly connected components of the MACID's
        full strategic relevance graph.
        Uses Tarjan’s algorithm with Nuutila’s modifications
        - complexity is linear in the number of edges and nodes"""
        return list(nx.strongly_connected_components(self))

    def _set_color_scc(self, node: str, sccs: Sequence[Any]) -> np.ndarray:
        "Assign a unique color to the set of nodes in each SCC."
        colors = cm.rainbow(np.linspace(0, 1, len(sccs)))
        scc_index = 0
        for idx, scc in enumerate(sccs):
            if node in scc:
                scc_index = idx
                break
        return colors[scc_index]  # type: ignore

    def draw_sccs(self) -> None:
        """
        Show the SCCs for the MACID's full strategic relevance graph
        """
        sccs = list(nx.strongly_connected_components(self))
        layout = nx.kamada_kawai_layout(self)
        colors = [self._set_color_scc(node, sccs) for node in self.nodes]
        nx.draw_networkx(self, pos=layout, node_size=400, arrowsize=20, edge_color="g", node_color=colors)
        plt.show()

    def draw(self) -> None:
        """
        Draw the MACID's relevance graph for the given set of decision nodes.
        Default: draw the relevance graph for all decision nodes in the MACID.
        """
        nx.draw_networkx(
            self, node_size=400, arrowsize=20, node_color="k", font_color="w", edge_color="k", with_labels=True
        )
        plt.show()


class CondensedRelevanceGraph(nx.DiGraph):
    """
    The nodes of a condensed_relevance graph are the maximal sccs of the full relevance graph
    of the original MAID.

    The condensed_relevance graph will always be acyclic. Therefore, we can return a topological ordering.
    """

    def __init__(self, macid: MACIDBase):
        super().__init__()
        rg = RelevanceGraph(macid)
        con_rel = nx.condensation(rg)
        self.add_nodes_from(con_rel.nodes)
        self.add_edges_from(con_rel.edges)
        # this generates a dictionary matching each decision node
        # in rg to the node of con_rel that it's in.
        self.graph["mapping"] = con_rel.graph["mapping"]

    def draw(self) -> None:
        """
        Draw the Condensed Relevance graph of a (MA)CID.
        """
        nx.draw_networkx(self, with_labels=True)
        plt.show()

    def get_scc_topological_ordering(self) -> List[List[str]]:
        """
        Return a topological ordering (which might not be unique) of the SCCs as
        a list of decision nodes in each SCC.
        """
        decs_in_each_scc = [self.get_decisions_in_scc()[scc] for scc in list(nx.topological_sort(self))]
        return decs_in_each_scc

    def get_decisions_in_scc(self) -> Dict[int, List[str]]:
        """Return a dictionary matching each SCC with a list of decision nodes that it contains"""
        scc_dec_mapping: Dict[int, List[str]] = {}
        # invert the dictionary to match each scc with the decision nodes in it
        for k, v in self.graph["mapping"].items():
            scc_dec_mapping[v] = scc_dec_mapping.get(v, []) + [k]
        return scc_dec_mapping
