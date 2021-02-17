from analyze.value_of_information import admits_voi_list
from core.cid import CID


def total_effect(cid: CID, A: str, X: str, a1: int, a2: int) -> float:
    "the total effect on X from intervening on A with a2 rather than a1"
    return cid.expected_value([X], {}, intervene={A: a2})[0] - \
        cid.expected_value([X], {}, intervene={A: a1})[0]


def introduced_total_effect(cid: CID, A: str, D: str, Y: str, a1: int, a2: int) -> float:
    """The total introduced effect, comparing the effect of A on D and Y """
    teD = total_effect(cid, A, D, a1, a2)
    teY = total_effect(cid, A, Y, a1, a2)
    return teD - teY
