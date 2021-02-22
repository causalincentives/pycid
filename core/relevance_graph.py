from __future__ import annotations
import itertools
from typing import List

import networkx as nx
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.macid_base import MACIDBase


class RelevanceGraph(nx.DiGraph):
    """
    The relevance graph for a set of decision nodes in the (MA)CID
    see: Hammond, L., Fox, J., Everitt, T., Abate, A., & Wooldridge, M. (2021).
    Equilibrium Refinements for Multi-Agent Influence Diagrams: Theory and Practice.
    Default: the set of decision nodes is all decision nodes in the MAID.
    - an edge D -> D' exists iff D' is r-reachable from D (ie D strategically or probabilistically relies on D')
    """

    def __init__(self, cid: MACIDBase, decisions: List[str] = None):
        super().__init__()
        if decisions is None:
            decisions = cid.all_decision_nodes
        self.add_nodes_from(decisions)
        dec_pair_perms = list(itertools.permutations(decisions, 2))
        for dec_pair in dec_pair_perms:
            if cid.is_s_reachable(dec_pair[0], dec_pair[1]):
                self.add_edge(dec_pair[0], dec_pair[1])

    def is_acyclic(self) -> bool:
        """
        Finds whether the relevance graph for all of the decision nodes in the MACID is acyclic.
        """
        return nx.is_directed_acyclic_graph(self)  # type: ignore

    def draw(self) -> None:
        """
        Draw the MACID's relevance graph for the given set of decision nodes.
        Default: draw the relevance graph for all decision nodes in the MACID.
        """
        nx.draw_networkx(self, node_size=400, arrowsize=20, node_color='k', font_color='w',
                         edge_color='k', with_labels=True)
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
        self.graph['mapping'] = con_rel.graph['mapping']

    def draw(self) -> None:
        nx.draw_networkx(self, with_labels=True)
        plt.show()
