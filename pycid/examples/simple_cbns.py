from pycid.core.causal_bayesian_network import CausalBayesianNetwork
from pycid.core.cpd import FunctionCPD, UniformRandomCPD


def get_3node_cbn() -> CausalBayesianNetwork:
    cbn = CausalBayesianNetwork([("S", "D"), ("S", "U"), ("D", "U")])
    cpd_s = UniformRandomCPD("S", [-1, 1])
    cpd_u = FunctionCPD("U", lambda s, d: s * d)  # type: ignore
    cpd_d = FunctionCPD("D", lambda s: s + 1)  # type: ignore
    cbn.add_cpds(cpd_d, cpd_s, cpd_u)
    return cbn


def get_3node_uniform_cbn() -> CausalBayesianNetwork:
    cbn = CausalBayesianNetwork([("A", "C"), ("A", "B"), ("B", "C")])
    cpd_a = UniformRandomCPD("A", [0, 1])
    cpd_b = UniformRandomCPD("B", [0, 1])
    cpd_c = FunctionCPD("C", lambda a, b: a * b)  # type: ignore
    cbn.add_cpds(cpd_a, cpd_b, cpd_c)
    return cbn


def get_minimal_cbn() -> CausalBayesianNetwork:
    cbn = CausalBayesianNetwork([("A", "B")])
    cpd_a = UniformRandomCPD("A", [0, 1])
    cpd_b = FunctionCPD("B", lambda a: a)  # type: ignore
    cbn.add_cpds(cpd_a, cpd_b)
    return cbn


def get_fork_cbn() -> CausalBayesianNetwork:
    cbn = CausalBayesianNetwork([("A", "C"), ("B", "C")])
    cpd_a = UniformRandomCPD("A", [1, 2])
    cpd_b = UniformRandomCPD("B", [3, 4])
    cpd_c = FunctionCPD("C", lambda a, b: a * b)  # type: ignore
    cbn.add_cpds(cpd_a, cpd_b, cpd_c)
    return cbn
