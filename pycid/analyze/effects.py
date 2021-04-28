from pycid.core.cid import CID


def total_effect(cid: CID, a: str, x: str, a0: int = 0, a1: int = 1) -> float:
    "the total effect on x from intervening on a with a2 rather than a1"
    total_effect = (
        cid.expected_value([x], {}, intervention={a: a1})[0] - cid.expected_value([x], {}, intervention={a: a0})[0]
    )
    return total_effect  # type: ignore


def introduced_total_effect(
    cid: CID, a: str, d: str, y: str, a0: int = 0, a1: int = 1, adapt_marginalized: bool = True
) -> float:
    """The total introduced effect, comparing the effect of a on d and y"""
    te_d = total_effect(cid, a, d, a0, a1)
    te_y = total_effect(cid, a, y, a0, a1)
    if te_y < 0 and adapt_marginalized:
        te_y = -te_y
        te_d = -te_d
    return te_d - te_y
