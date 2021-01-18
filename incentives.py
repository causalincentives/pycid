from typing import List
from cid import CID


def total_effect(cid: CID, A: str, X: str, a1, a2) -> float:
    "the total effect on X from intervening on A with a2 rather than a1"
    return cid.expected_value(X, {}, intervene={A: a2}) - \
           cid.expected_value(X, {}, intervene={A: a1})


def introduced_total_effect(cid: CID, A: str, D: str, Y: str, a1, a2) -> float:
    """The total introduced effect, comparing the effect of A on D and Y """
    teD = total_effect(cid, A, D, a1, a2)
    teY = total_effect(cid, A, Y, a1, a2)
    return teD - teY


def trimmed(cid: CID) -> CID:
    """Return the trimmed version of the graph

    Based on algorithm from Sect 4.5 of Lauritzen and Nilsson 2011, but simplified
    uusing the assumption that the graph is soluble"""

    def nr_observations(d: str) -> List[str]:
        """Get nonrequisite observations"""
        nonrequisite = []
        parents = cid.get_parents(d)
        for obs in parents:
            observed = list(set(parents + [d]) - set([obs]))
            connected = set(cid.active_trail_nodes([obs], observed=observed)[obs])
            downstream_utilities = set([i for i in cid.utility_nodes if d in cid._get_ancestors_of(i)])
            # if len([u for u in downstream_utilities if u in connected])==0:
            # import ipdb; ipdb.set_trace()
            if not connected.intersection(downstream_utilities):
                nonrequisite.append(obs)
        return nonrequisite

    new = cid.copy()
    decisions = cid.decision_nodes
    while True:
        removed = 0
        for decision in decisions:
            nonrequisite = new.nr_observations(decision)
            for nr in nonrequisite:
                removed += 1
                new.remove_edge(nr, decision)
        if removed == 0:
            break
    return new
