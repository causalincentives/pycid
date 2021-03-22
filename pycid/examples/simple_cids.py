from pycid.core.cid import CID
from pycid.core.cpd import DecisionDomain, FunctionCPD, StochasticFunctionCPD, UniformRandomCPD


def get_minimal_cid() -> CID:
    cid = CID([("A", "B")], decisions=["A"], utilities=["B"])
    cpd_a = DecisionDomain("A", [0, 1])
    cpd_b = FunctionCPD("B", lambda a: a)  # type: ignore
    cid.add_cpds(cpd_a, cpd_b)
    return cid


def get_3node_cid() -> CID:
    cid = CID([("S", "D"), ("S", "U"), ("D", "U")], decisions=["D"], utilities=["U"])
    cpd_s = UniformRandomCPD("S", [-1, 1])
    cpd_u = FunctionCPD("U", lambda s, d: s * d)  # type: ignore
    cpd_d = DecisionDomain("D", [-1, 1])
    cid.add_cpds(cpd_d, cpd_s, cpd_u)
    return cid


def get_5node_cid() -> CID:
    cid = CID(
        [("S1", "D"), ("S1", "U1"), ("S2", "D"), ("S2", "U2"), ("D", "U1"), ("D", "U2")],
        decisions=["D"],
        utilities=["U1", "U2"],
    )
    cpd_s1 = UniformRandomCPD("S1", [0, 1])
    cpd_s2 = UniformRandomCPD("S2", [0, 1])
    cpd_u1 = FunctionCPD("U1", lambda s1, d: int(s1 == d))  # type: ignore
    cpd_u2 = FunctionCPD("U2", lambda s2, d: int(s2 == d))  # type: ignore
    cpd_d = DecisionDomain("D", [0, 1])
    cid.add_cpds(cpd_d, cpd_s1, cpd_s2, cpd_u1, cpd_u2)
    return cid


def get_5node_cid_with_scaled_utility() -> CID:
    cid = CID(
        [("S1", "D"), ("S1", "U1"), ("S2", "D"), ("S2", "U2"), ("D", "U1"), ("D", "U2")],
        decisions=["D"],
        utilities=["U1", "U2"],
    )
    cpd_s1 = UniformRandomCPD("S1", [0, 1])
    cpd_s2 = UniformRandomCPD("S2", [0, 1])
    cpd_u1 = FunctionCPD("U1", lambda s1, d: 10 * int(s1 == d))  # type: ignore
    cpd_u2 = FunctionCPD("U2", lambda s2, d: 2 * int(s2 == d))  # type: ignore
    cpd_d = DecisionDomain("D", [0, 1])
    cid.add_cpds(cpd_d, cpd_s1, cpd_s2, cpd_u1, cpd_u2)
    return cid


def get_2dec_cid() -> CID:
    cid = CID(
        [("S1", "S2"), ("S1", "D1"), ("D1", "S2"), ("S2", "U"), ("S2", "D2"), ("D2", "U")],
        decisions=["D1", "D2"],
        utilities=["U"],
    )
    cpd_s1 = UniformRandomCPD("S1", [0, 1])
    cpd_d1 = DecisionDomain("D1", [0, 1])
    cpd_d2 = DecisionDomain("D2", [0, 1])
    cpd_s2 = FunctionCPD("S2", lambda s1, d1: int(s1 == d1))  # type: ignore
    cpd_u = FunctionCPD("U", lambda s2, d2: int(s2 == d2))  # type: ignore
    cid.add_cpds(cpd_s1, cpd_d1, cpd_s2, cpd_d2, cpd_u)
    return cid


def get_quantitative_voi_cid() -> CID:
    cid = CID([("S", "X"), ("X", "D"), ("D", "U"), ("S", "U")], decisions=["D"], utilities=["U"])
    cpd_s = UniformRandomCPD("S", [-1, 1])
    # X takes the value of S with probability 0.8
    cpd_x = StochasticFunctionCPD("X", lambda s: {s: 0.8}, domain=[-1, 1])
    cpd_d = DecisionDomain("D", [-1, 0, 1])
    cpd_u = FunctionCPD("U", lambda s, d: int(s) * int(d))  # type: ignore
    cid.add_cpds(cpd_s, cpd_x, cpd_d, cpd_u)
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
        UniformRandomCPD("S1", [0, 1]),
        DecisionDomain("D1", [0, 1]),
        FunctionCPD("U1", lambda s1, d1: int(s1 == d1)),  # type: ignore
        FunctionCPD("S2", lambda d1: d1),  # type: ignore
        DecisionDomain("D2", [0, 1]),
        FunctionCPD("U2", lambda s2, d2: int(s2 == d2)),  # type: ignore
    )
    return cid


def get_insufficient_recall_cid() -> CID:
    cid = CID([("A", "U"), ("B", "U")], decisions=["A", "B"], utilities=["U"])
    cid.add_cpds(
        DecisionDomain("A", [0, 1]),
        DecisionDomain("B", [0, 1]),
        FunctionCPD("U", lambda a, b: a * b),  # type: ignore
    )
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
