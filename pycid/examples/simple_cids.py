from pycid.core.cid import CID
from pycid.core.cpd import discrete_uniform, noisy_copy


def get_minimal_cid() -> CID:
    cid = CID([("A", "B")], decisions=["A"], utilities=["B"])
    cid.add_cpds(A=[0, 1], B=lambda a: a)
    return cid


def get_3node_cid() -> CID:
    cid = CID([("S", "D"), ("S", "U"), ("D", "U")], decisions=["D"], utilities=["U"])
    cid.add_cpds(S=discrete_uniform([-1, 1]), U=lambda s, d: s * d, D=[-1, 1])
    return cid


def get_5node_cid() -> CID:
    cid = CID(
        [("S1", "D"), ("S1", "U1"), ("S2", "D"), ("S2", "U2"), ("D", "U1"), ("D", "U2")],
        decisions=["D"],
        utilities=["U1", "U2"],
    )
    cid.add_cpds(
        S1=discrete_uniform([0, 1]),
        S2=discrete_uniform([0, 1]),
        U1=lambda s1, d: int(s1 == d),
        U2=lambda s2, d: int(s2 == d),
        D=[0, 1],
    )
    return cid


def get_5node_cid_with_scaled_utility() -> CID:
    cid = CID(
        [("S1", "D"), ("S1", "U1"), ("S2", "D"), ("S2", "U2"), ("D", "U1"), ("D", "U2")],
        decisions=["D"],
        utilities=["U1", "U2"],
    )
    cid.add_cpds(
        S1=discrete_uniform([0, 1]),
        S2=discrete_uniform([0, 1]),
        U1=lambda s1, d: 10 * int(s1 == d),
        U2=lambda s2, d: 2 * int(s2 == d),
        D=[0, 1],
    )
    return cid


def get_2dec_cid() -> CID:
    cid = CID(
        [("S1", "S2"), ("S1", "D1"), ("D1", "S2"), ("S2", "U"), ("S2", "D2"), ("D2", "U")],
        decisions=["D1", "D2"],
        utilities=["U"],
    )
    cid.add_cpds(
        S1=discrete_uniform([0, 1]),
        D1=[0, 1],
        D2=[0, 1],
        S2=lambda s1, d1: int(s1 == d1),
        U=lambda s2, d2: int(s2 == d2),
    )
    return cid


def get_quantitative_voi_cid() -> CID:
    cid = CID([("S", "X"), ("X", "D"), ("D", "U"), ("S", "U")], decisions=["D"], utilities=["U"])
    cid.add_cpds(
        S=discrete_uniform([-1, 1]),
        X=lambda s: noisy_copy(s, probability=0.8, domain=[-1, 1]),
        D=[-1, 0, 1],
        U=lambda s, d: int(s) * int(d),
    )
    return cid


def get_sequential_cid() -> CID:
    """
    This CID is a subtle case of sufficient recall, as the decision rule for D1 influences
    the expected utility of D2, but D2 can still be chosen without knowing D1, since
    D1 does not influence any utility nodes descending from D2.
    """
    cid = CID(
        [
            ("S1", "D1"),
            ("D1", "U1"),
            ("S1", "U1"),
            ("D1", "S2"),
            ("S2", "D2"),
            ("D2", "U2"),
            ("S2", "U2"),
        ],
        decisions=["D1", "D2"],
        utilities=["U1", "U2"],
    )

    cid.add_cpds(
        S1=discrete_uniform([0, 1]),
        D1=[0, 1],
        U1=lambda s1, d1: int(s1 == d1),
        S2=lambda d1: d1,
        D2=[0, 1],
        U2=lambda s2, d2: int(s2 == d2),
    )
    return cid


def get_insufficient_recall_cid() -> CID:
    cid = CID([("A", "U"), ("B", "U")], decisions=["A", "B"], utilities=["U"])
    cid.add_cpds(A=[0, 1], B=[0, 1], U=lambda a, b: a * b)
    return cid


def get_trim_example_cid() -> CID:
    cid = CID(
        [
            ("Y1", "D1"),
            ("Y1", "Y2"),
            ("Y1", "D2"),
            ("Y2", "D2"),
            ("Y2", "U"),
            ("D1", "Y2"),
            ("D1", "D2"),
            ("Z1", "D1"),
            ("Z1", "D2"),
            ("Z1", "Z2"),
            ("Z2", "D2"),
            ("D2", "U"),
        ],
        decisions=["D1", "D2"],
        utilities=["U"],
    )
    return cid
