from typing import List

from analyze.value_of_information import admits_voi_list
from cid import CID


def total_effect(cid: CID, A: str, X: str, a1, a2) -> float:
    "the total effect on X from intervening on A with a2 rather than a1"
    return cid.expected_value([X], {}, intervene={A: a2})[0] - \
           cid.expected_value([X], {}, intervene={A: a1})[0]


def introduced_total_effect(cid: CID, A: str, D: str, Y: str, a1, a2) -> float:
    """The total introduced effect, comparing the effect of A on D and Y """
    teD = total_effect(cid, A, D, a1, a2)
    teY = total_effect(cid, A, Y, a1, a2)
    return teD - teY


# TODO find a better place to put this
def trimmed(cid: CID) -> CID:
    """Return the trimmed version of the graph

    Based on algorithm from Sect 4.5 of Lauritzen and Nilsson 2011, but simplified
    uusing the assumption that the graph is soluble"""
    new = cid.copy()
    decisions = cid.decision_nodes
    while True:
        removed = 0
        for decision in decisions:
            nonrequisite = admits_voi_list(cid, decision)
            for nr in nonrequisite:
                removed += 1
                new.remove_edge(nr, decision)
        if removed == 0:
            break
    return new
