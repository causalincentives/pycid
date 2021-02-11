from analyze.value_of_information import admits_voi
from core.cid import CID


def nonrequisite(cid: CID, decision: str, node: str) -> bool:
    """Returns true if node -> decision is a nonrequisite observation link.
    - A nonrequisite observation link in a single decision CID G is any edge X → D where:
    X ⊥ U(D) | (Pa_D ∪ {D} \ {X} (where U(D) are the utility nodes downstream of D)
    ("Agent Incentives: a Causal Perspective" by Everitt, Carey, Langlois, Ortega, and Legg, 2020)
    """
    return not admits_voi(cid, decision, node) and node in cid.get_parents(decision)


def requisite_graph(cid: CID) -> CID:
    """Return the requisite graph of the original CID, also called
    a minimal_reduction, d_reduction, or the trimmed graph.
    - The requisite graph G∗ of a single-decision CID G is the result of removing from G
    all nonrequisite observation links.
    ("Agent Incentives: a Causal Perspective" by Everitt, Carey, Langlois, Ortega, and Legg, 2020)
    """
    requisite_graph = cid.copy_without_cpds()
    decisions = cid.all_decision_nodes
    while True:
        removed = 0
        for decision in decisions:
            non_requisite_nodes = [node for node in cid.nodes if nonrequisite(requisite_graph, decision, node)]
            for nr in non_requisite_nodes:
                removed += 1
                requisite_graph.remove_edge(nr, decision)
        if removed == 0:
            break
    return requisite_graph
