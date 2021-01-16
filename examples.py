# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.

from pgmpy.factors.discrete import TabularCPD
import numpy as np
from cid import CID
from cpd import NullCPD, FunctionCPD


def get_minimal_cid():
    cid = CID([('A', 'B')],
              decision_nodes=['A'],
              utility_nodes=['B'])
    cpd_a = NullCPD('A', 2)
    # cpd_a = TabularCPD('B',2,[[1., 0.], [0., 1.]], evidence=['A'], evidence_card = [2])
    cpd_b = FunctionCPD('B', lambda a: a, evidence=['A'])
    cid.add_cpds(cpd_a, cpd_b)
    return cid


def get_3node_cid() -> CID:
    cid = CID([('S', 'D'), ('S', 'U'), ('D', 'U')],
              decision_nodes=['D'],
              utility_nodes=['U'])
    cpd_s = TabularCPD('S', 2, np.array([[.5], [.5]]))
    # mat = np.array([[0,1, 1,0], [1,0,0,1]])
    # cpd_u = TabularCPD('U', 2, mat, evidence=['S', 'D'], evidence_card=[2,2])
    cpd_u = FunctionCPD('U', lambda s, d: int(s == d), evidence=['S', 'D'])
    cpd_d = NullCPD('D', 2)
    cid.add_cpds(cpd_d, cpd_s, cpd_u)
    return cid


def get_5node_cid() -> CID:
    cid = CID([
        ('S1', 'D'),
        ('S1', 'U1'),
        ('S2', 'D'),
        ('S2', 'U2'),
        ('D', 'U1'),
        ('D', 'U2')],
        decision_nodes=['D'],
        utility_nodes=['U1', 'U2'])
    cpd_s1 = TabularCPD('S1', 2, np.array([[.5], [.5]]))
    cpd_s2 = TabularCPD('S2', 2, np.array([[.5], [.5]]))
    # mat = np.array([[0, 1, 1, 0], [1, 0, 0, 1]])
    # cpd_u1 = TabularCPD('U1', 2, mat, evidence=['S1', 'D'], evidence_card=[2, 2])
    # cpd_u2 = TabularCPD('U2', 2, mat, evidence=['S2', 'D'], evidence_card=[2, 2])
    cpd_u1 = FunctionCPD('U1', lambda s1, d: int(s1 == d), evidence=['S1', 'D'])
    cpd_u2 = FunctionCPD('U2', lambda s2, d: int(s2 == d), evidence=['S2', 'D'])
    cpd_d = NullCPD('D', 2)
    cid.add_cpds(cpd_d, cpd_s1, cpd_s2, cpd_u1, cpd_u2)
    return cid


def get_5node_cid_with_scaled_utility() -> CID:
    cid = CID([
        ('S1', 'D'),
        ('S1', 'U1'),
        ('S2', 'D'),
        ('S2', 'U2'),
        ('D', 'U1'),
        ('D', 'U2')],
        decision_nodes=['D'],
        utility_nodes=['U1', 'U2'])
    cpd_s1 = TabularCPD('S1', 2, np.array([[.5], [.5]]))
    cpd_s2 = TabularCPD('S2', 2, np.array([[.5], [.5]]))
    # mat = np.array([[0, 1, 1, 0], [1, 0, 0, 1]])
    # cpd_u1 = TabularCPD('U1', 2, mat, evidence=['S1', 'D'], evidence_card=[2, 2])
    # cpd_u2 = TabularCPD('U2', 2, mat, evidence=['S2', 'D'], evidence_card=[2, 2])
    cpd_u1 = FunctionCPD('U1', lambda s1, d: 10*int(s1 == d), evidence=['S1', 'D'])
    cpd_u2 = FunctionCPD('U2', lambda s2, d: 2*int(s2 == d), evidence=['S2', 'D'])
    cpd_d = NullCPD('D', 2)
    cid.add_cpds(cpd_d, cpd_s1, cpd_s2, cpd_u1, cpd_u2)
    return cid


# def get_5node_cid_with_scaled_utility():
#     from pgmpy.factors.discrete.CPD import TabularCPD
#     cid = CID([
#         ('S1', 'D'),
#         ('S1', 'U1'),
#         ('S2', 'D'),
#         ('S2', 'U2'),
#         ('D', 'U1'),
#         ('D', 'U2')
#     ],
#         decision_nodes=['D'],
#         utility_nodes=['U1', 'U2'])
#     s1cpd = TabularCPD('S1', 2, np.array([[.5], [.5]]))
#     s2cpd = TabularCPD('S2', 2, np.array([[.5], [.5]]))
#     mat = np.array([[0, 1, 1, 0], [1, 0, 0, 1]])
#     u1cpd = TabularCPD('U1', 2, mat, evidence=['S1', 'D'], evidence_card=[2, 2], state_names={'U1': [0, 10]})
#     u2cpd = TabularCPD('U2', 2, mat, evidence=['S2', 'D'], evidence_card=[2, 2], state_names={'U2': [0, 2]})
#     nullcpd = NullCPD('D', 2)
#     cid.add_cpds(nullcpd, s1cpd, s2cpd, u1cpd, u2cpd)
#     return cid


def get_2dec_cid() -> CID:
    from pgmpy.factors.discrete.CPD import TabularCPD
    cid = CID([
        ('S1', 'S2'),
        ('S1', 'D1'),
        ('D1', 'S2'),
        ('S2', 'U'),
        ('S2', 'D2'),
        ('D2', 'U')
    ],
        decision_nodes=['D1', 'D2'],
        utility_nodes=['U'])
    cpd_s1 = TabularCPD('S1', 2, np.array([[.5], [.5]]))
    cpd_d1 = NullCPD('D1', 2)
    cpd_d2 = NullCPD('D2', 2)
    # mat1 = np.array([[0, 1, 1, 0], [1, 0, 0, 1]])
    # cpd_s2 = TabularCPD('S2', 2, mat1, evidence=['S1', 'D1'], evidence_card=[2, 2])
    # mat2 = np.array([[0, 1, 1, 0], [1, 0, 0, 1]])
    # cpd_u = TabularCPD('U', 2, mat2, evidence=['S2', 'D2'], evidence_card=[2, 2])
    cpd_s2 = FunctionCPD('S2', lambda s2, d1: int(s2 == d1), evidence=['S1', 'D1'])
    cpd_u = FunctionCPD('U', lambda s2, d2: int(s2 == d2), evidence=['S2', 'D2'])
    cid.add_cpds(cpd_s1, cpd_d1, cpd_s2, cpd_d2, cpd_u)
    return cid


def get_insufficient_recall_cid() -> CID:
    cid = CID([('A', 'U'), ('B', 'U')], decision_nodes=['A', 'B'], utility_nodes=['U'])
    tabcpd = TabularCPD('U', 2, np.random.randn(2, 4), evidence=['A', 'B'], evidence_card=[2, 2])
    cid.add_cpds(NullCPD('A', 2), NullCPD('B', 2), tabcpd)
    return cid


def get_nested_cid() -> CID:
    cid = CID([
        ('S1', 'D1'),
        ('S1', 'S3'),
        ('D1', 'S3'),
        ('S2', 'S4'),
        ('S2', 'U'),
        ('S3', 'S4'),
        ('S3', 'D2'),
        ('S4', 'D2'),
        ('D2', 'U')],
        decision_nodes=['D1', 'D2'],
        utility_nodes=['U'])
    cpds = [
        TabularCPD('S1', 2, np.array([[.5], [.5]])),
        TabularCPD('S2', 2, np.array([[.5], [.5]])),
        TabularCPD('S3', 2, np.array([[0, 1, 1, 0], [1, 0, 0, 1]]), evidence=['S1', 'D1'], evidence_card=[2, 2]),
        TabularCPD('S4', 2, np.array([[.5, 0, .5, 1], [.5, 0, 0.5, 1]]), evidence=['S2', 'S3'], evidence_card=[2, 2]),
        NullCPD('D1', 2),
        NullCPD('D2', 2),
        TabularCPD('U', 2, np.array([[1, 0, 0, 1], [0, 1, 1, 0]]), evidence=['D2', 'S2'], evidence_card=[2, 2])
    ]
    cid.add_cpds(*cpds)
    return cid


def get_introduced_bias() -> CID:

    cid = CID([
        ('A', 'X'),  # defining the graph's nodes and edges
        ('Z', 'X'),
        ('Z', 'Y'),
        ('X', 'D'),
        ('X', 'Y'),
        ('D', 'U'),
        ('Y', 'U')
    ],
        decision_nodes=['D'],
        utility_nodes=['U'])

    cpd_A = TabularCPD('A', 2, np.array([[.5], [.5]]))
    cpd_Z = TabularCPD('Z', 2, np.array([[.5], [.5]]))
    cpd_X = FunctionCPD('X', lambda a, z: a*z, evidence=['A', 'Z'])
    cpd_D = NullCPD('D', 2)
    cpd_Y = FunctionCPD('Y', lambda x, z: x + z, evidence=['X', 'Z'])
    cpd_U = FunctionCPD('U', lambda d, y: -(d - y) ** 2, evidence=['D', 'Y'])

    cid.add_cpds(cpd_A, cpd_D, cpd_Z, cpd_X, cpd_Y, cpd_U)

    return cid

# cpd = FunctionCPD('A', lambda : 0, evidence=[])
# cid = get_minimal_cid()
# cpd.initializeTabularCPD(cid)