# -------------Methods for finding the MAID's strategic relevance graph and checking cyclicity ------------------------------------------------------------

def _is_s_reachable(self, d1: str, d2: str):
    """

    - this method determines whether 'D2' is s-reachable from 'D1' (Koller and Milch 2001)

    A node D2 is s-reachable from a node D1 in a MACID M if there is some utility node U ∈ U_D
    such that if a new parent D2' were added to D2, there would be an active path in M from
    D2′ to U given Pa(D)∪{D}, where a path is active in a MAID if it is active in the same graph, viewed as a BN.

    """
    self.add_edge('temp_par', d2)
    agent = self._get_dec_agent(d1)
    agent_utilities = self.all_utility_nodes[agent]
    con_nodes = [d1] + self.get_parents(d1)
    is_active_trail = any([self.is_active_trail('temp_par', u_node, con_nodes) for u_node in agent_utilities])
    self.remove_node('temp_par')
    return is_active_trail


def strategic_rel_graph(self):
    """
    finds the strategic relevance graph of the MAID
    an edge D' -> D exists iff D' is s-reachable from D
    """
    G = nx.DiGraph()
    dec_pair_perms = list(itertools.permutations(self.all_decision_nodes, 2))
    for dec_pair in dec_pair_perms:
        if self._is_s_reachable(dec_pair[0], dec_pair[1]):
            G.add_edge(dec_pair[1], dec_pair[0])
    return G


def draw_strategic_rel_graph(self):
    """
    draws a MACID's strategic relevance graph
    """
    rg = self.strategic_rel_graph()
    nx.draw_networkx(rg, node_size=400, arrowsize=20, edge_color='g', with_labels=True)
    plt.figure(2)
    plt.draw()


def strategically_acyclic(self):
    """
    finds whether the MACID has an acyclic strategic relevance graph.
    """
    rg = self.strategic_rel_graph()
    if nx.is_directed_acyclic_graph(rg):
        return True
    else:
        return False


def get_acyclic_topological_ordering(self):
    """
    first checks whether the strategic relevance graph is acyclic
    returns a topological ordering (which might not be unique) of the decision nodes
    """
    rg = self.strategic_rel_graph()
    if self.strategically_acyclic():
        return list(nx.topological_sort(rg))
    else:
        return 'The strategic relevance graph for this MACID is not acyclic and so \
                       no topological ordering can be immediately given. Use .. method instead.'

