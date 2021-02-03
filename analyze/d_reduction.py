from analyze.value_of_information import admits_voi
from core.cid import CID


def nonrequisite(cid: CID, decision: str, node: str) -> bool:
    """Returns true if node -> deision is a nonrequisite observation link
    - A nonrequisite observation link in a single decision CID G is any edge X → D where:
        X ⊥ U(D) | (PaD ∪ {D} \ {X} (where U(D) are the utility nodes downstream of D)
    ("Agent Incentives: a Causal Perspective" by Everitt, Carey, Langlois, Ortega, and Legg, 2020)
    """
    return not admits_voi(cid, decision, node) and node in cid.get_parents(decision)


def d_reduction(cid: CID) -> CID:
    """Return the d-reduced version of the CID's graph, sometimes called the trimmed graph.
    - The d-reduced graph G∗ of a single-decision CID G is the result of removing from G 
    all nonrequisite observation links.
    ("Agent Incentives: a Causal Perspective" by Everitt, Carey, Langlois, Ortega, and Legg, 2020)
    Based on algorithm from Sect 4.5 of Lauritzen and Nilsson 2001, but simplified
    using the assumption that the graph is soluble
    """
    reduced_cid = cid.copy_without_cpds()
    decisions = cid.all_decision_nodes
    while True:
        removed = 0
        for decision in decisions:
            non_requisite_nodes = [node for node in cid.nodes if nonrequisite(reduced_cid, decision, node)]
            for nr in non_requisite_nodes:
                removed += 1
                reduced_cid.remove_edge(nr, decision)
        if removed == 0:
            break
    return reduced_cid
