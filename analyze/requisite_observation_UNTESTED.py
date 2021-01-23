from analyze.value_of_information import admits_voi
from core.macid_base import MACIDBase


def requisite(cid: MACIDBase, decision: str, node: str) -> bool:
    """Returns true if node is a requisite observation"""
    return admits_voi(cid, decision, node) and node in cid.get_parents(decision)


def trim(cid: MACIDBase) -> MACIDBase:
    """Return the trimmed version of the graph, sometimes called d-reduction

    Based on algorithm from Sect 4.5 of Lauritzen and Nilsson 2011, but simplified
    using the assumption that the graph is soluble"""
    new = cid.copy()
    decisions = cid.all_decision_nodes
    while True:
        removed = 0
        for decision in decisions:
            non_requisite_nodes = [node for node in cid.nodes if requisite(cid, decision, node)]
            for nr in non_requisite_nodes:
                removed += 1
                new.remove_edge(nr, decision)
        if removed == 0:
            break
    return new
