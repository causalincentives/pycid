#Licensed to the Apache Software Foundation (ASF) under one or more contributor license 
#agreements; and to You under the Apache License, Version 2.0.

from pgmpy.factors.discrete import TabularCPD
import numpy as np
from cid import CID
from cpd import NullCPD

def get_minimal_cid():
    from pgmpy.factors.discrete.CPD import TabularCPD
    cid = CID([('A', 'B')],
              decision_nodes=['A'],
              utility_nodes=['B'])
    cpd = TabularCPD('B',2,[[1., 0.], [0., 1.]], evidence=['A'], evidence_card = [2])
    nullcpd = NullCPD('A', 2)
    cid.add_cpds(nullcpd, cpd)
    return cid

def get_3node_cid():
    from pgmpy.factors.discrete.CPD import TabularCPD
    cid = CID([('S', 'D'), ('S', 'U'), ('D', 'U')],
              decision_nodes=['D'],
              utility_nodes=['U'])
    scpd = TabularCPD('S',2,np.array([[.5],[.5]]))#, evidence=[], evidence_card = [])
    mat = np.array([[0,1, 1,0], [1,0,0,1]])
    ucpd = TabularCPD('U', 2, mat, evidence=['S', 'D'], evidence_card=[2,2])
    nullcpd = NullCPD('D', 2)
    cid.add_cpds(nullcpd, scpd, ucpd)
    return cid

def get_5node_cid():
    from pgmpy.factors.discrete.CPD import TabularCPD
    cid = CID([
        ('S1', 'D'),
        ('S1', 'U1'),
        ('S2', 'D'),
        ('S2', 'U2'),
        ('D', 'U1'),
        ('D', 'U2')
        ],
        decision_nodes = ['D'],
        utility_nodes = ['U1', 'U2'])
    s1cpd = TabularCPD('S1',2,np.array([[.5],[.5]]))
    s2cpd = TabularCPD('S2',2,np.array([[.5],[.5]]))
    mat = np.array([[0,1, 1,0], [1,0,0,1]])
    u1cpd = TabularCPD('U1', 2, mat, evidence=['S1', 'D'], evidence_card=[2,2])
    u2cpd = TabularCPD('U2', 2, mat, evidence=['S2', 'D'], evidence_card=[2,2])
    nullcpd = NullCPD('D', 2)
    cid.add_cpds(nullcpd, s1cpd, s2cpd, u1cpd, u2cpd)
    return cid

def get_2dec_cid():
    from pgmpy.factors.discrete.CPD import TabularCPD
    cid = CID([
        ('S1', 'S2'), 
        ('S1','D1'), 
        ('D1','S2'),
        ('S2', 'U'), 
        ('S2', 'D2'), 
        ('D2', 'U')
        ],
        decision_nodes=['D1', 'D2'],
        utility_nodes=['U'])
    s1cpd = TabularCPD('S1',2,np.array([[.5],[.5]]))#, evidence=[], evidence_card = [])
    d1cpd = NullCPD('D1', 2)
    d2cpd = NullCPD('D2', 2)
    mat1 = np.array([[0,1, 1,0], [1,0,0,1]])
    s2cpd = TabularCPD('S2', 2, mat1, evidence=['S1', 'D1'], evidence_card=[2,2])
    mat2 = np.array([[0,1, 1,0], [1,0,0,1]])
    ucpd = TabularCPD('U', 2, mat2, evidence=['S2', 'D2'], evidence_card=[2,2])
    cid.add_cpds(s1cpd, d1cpd, s2cpd, d2cpd, ucpd)
    return cid


def get_insufficient_recall_cid():
    cid = CID([('A','U'),('B','U')], decision_nodes=['A', 'B'], utility_nodes = ['U'])
    tabcpd = TabularCPD('U', 2, np.random.randn(2,4), evidence=['A','B'], evidence_card=[2,2])
    cid.add_cpds(NullCPD('A', 2), NullCPD('B', 2), tabcpd)
    return cid
        
def get_nested_cid():
    from pgmpy.factors.discrete.CPD import TabularCPD
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
        decision_nodes = ['D1', 'D2'],
        utility_nodes = ['U'])
    cpds = [
        TabularCPD('S1',2,np.array([[.5],[.5]])),
        TabularCPD('S2',2,np.array([[.5],[.5]])),
        TabularCPD('S3',2,np.array([[0, 1, 1, 0],[1, 0, 0, 1]]), evidence=['S1', 'D1'], evidence_card=[2,2]),
        TabularCPD('S4',2,np.array([[.5, 0, .5, 1],[.5, 0, 0.5, 1]]), evidence=['S2', 'S3'], evidence_card=[2,2]),
        NullCPD('D1', 2),
        NullCPD('D2', 2),
        TabularCPD('U', 2, np.array([[1, 0, 0, 1], [0, 1, 1, 0]]), evidence=['D2', 'S2'], evidence_card=[2,2])
    ]
    cid.add_cpds(*cpds)
    return cid

