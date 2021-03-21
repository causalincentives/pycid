from pycid.core.causal_bayesian_network import CausalBayesianNetwork
from pycid.core.cpd import FunctionCPD, UniformRandomCPD


def get_3node_cbn() -> CausalBayesianNetwork:
    cbn = CausalBayesianNetwork([("S", "D"), ("S", "U"), ("D", "U")])
    cpd_s = UniformRandomCPD("S", [-1, 1])
    cpd_u = FunctionCPD("U", lambda s, d: s * d)
    cpd_d = FunctionCPD("D", lambda s: s + 1)
    cbn.add_cpds(cpd_d, cpd_s, cpd_u)
    return cbn
