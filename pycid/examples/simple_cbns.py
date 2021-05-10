from pycid.core.causal_bayesian_network import CausalBayesianNetwork
from pycid.core.cpd import bernoulli, discrete_uniform


def get_3node_cbn() -> CausalBayesianNetwork:
    cbn = CausalBayesianNetwork([("S", "D"), ("S", "U"), ("D", "U")])
    cbn.add_cpds(S=discrete_uniform([-1, 1]), D=lambda s: s + 1, U=lambda s, d: s * d)
    return cbn


def get_3node_uniform_cbn() -> CausalBayesianNetwork:
    cbn = CausalBayesianNetwork([("A", "C"), ("A", "B"), ("B", "C")])
    cbn.add_cpds(A=bernoulli(0.5), B=bernoulli(0.5), C=lambda a, b: a * b)
    return cbn


def get_minimal_cbn() -> CausalBayesianNetwork:
    cbn = CausalBayesianNetwork([("A", "B")])
    cbn.add_cpds(A=discrete_uniform([0, 1]), B=lambda a: a)
    return cbn


def get_fork_cbn() -> CausalBayesianNetwork:
    cbn = CausalBayesianNetwork([("A", "C"), ("B", "C")])
    cbn.add_cpds(A=discrete_uniform([1, 2]), B=discrete_uniform([3, 4]), C=lambda a, b: a * b)
    return cbn
