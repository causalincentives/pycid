from pycid.core.cid import CID


def total_effect(cid: CID, a: str, x: str, a1: int, a2: int) -> float:
    "the total effect on x from intervening on a with a2 rather than a1"
    total_effect = cid.expected_value([x], {}, intervene={a: a2})[0] - cid.expected_value([x], {}, intervene={a: a1})[0]
    return total_effect  # type: ignore


def introduced_total_effect(cid: CID, a: str, d: str, y: str, a1: int, a2: int) -> float:
    """The total introduced effect, comparing the effect of a on d and y """
    te_d = total_effect(cid, a, d, a1, a2)
    te_y = total_effect(cid, a, y, a1, a2)
    return te_d - te_y
