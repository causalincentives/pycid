#Licensed to the Apache Software Foundation (ASF) under one or more contributor license
#agreements; and to You under the Apache License, Version 2.0.


from cid import CID
from pgmpy.factors.discrete import TabularCPD


class TwoDecisions(CID):

    def __init__(self):
        super().__init__([('S1', 'D1'),
                          ('S1', 'S2'),
                          ('D1', 'S2'),
                          ('S2', 'D2'),
                          ('D2', 'U'),
                          ('S2', 'U')],
                         decision_nodes = ['D1', 'D2'],
                         utility_nodes = ['U'])

        cpd_S1 = TabularCPD(variable='S1', variable_card=2,
                            values=[[0.5],
                                    [0.5]])
        cpd_D1= TabularCPD(variable='D1', variable_card=2,
                           values=[[0.5, 0.5],
                                   [0.5, 0.5]],
                           evidence=['S1'], evidence_card=[2])
        cpd_D2= TabularCPD(variable='D2', variable_card=2,
                           values=[[0.99, 0.02],
                                   [0.01, 0.98]],
                           evidence=['S2'], evidence_card=[2])
        cpd_S2 = TabularCPD(variable='S2', variable_card=2,
                            values=[[0, 1, 1, 0],
                                    [1, 0, 0, 1]],
                            evidence=['S1', 'D1'],
                            evidence_card=[2, 2])
        cpd_U = TabularCPD(variable='U', variable_card=2,
                           values=[[0, 1, 1, 0],
                                   [1, 0, 0, 1]],
                           evidence=['S2', 'D2'],
                           evidence_card=[2, 2])
        self.add_cpds(cpd_S1, cpd_D1, cpd_S2, cpd_D2, cpd_U)
