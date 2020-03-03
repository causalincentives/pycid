#Licensed to the Apache Software Foundation (ASF) under one or more contributor license
#agreements; and to You under the Apache License, Version 2.0.

from cid import CID
from pgmpy.factors.discrete import TabularCPD


class ThreeNode(CID):

    def __init__(self):
        super().__init__([('S', 'D'),
                          ('S', 'U'),
                          ('D', 'U')],
                           decision_nodes = ['D'],
                           utility_nodes = ['U'])

        cpd_S = TabularCPD(variable='S', variable_card=2,
                           values=[[0.5],
                                   [0.5]])
        cpd_D= TabularCPD(variable='D', variable_card=2,
                          values=[[0.5, 0.5],
                                  [0.5, 0.5]],
                          evidence=['S'], evidence_card=[2])
        cpd_U = TabularCPD(variable='U', variable_card=2,
                           values=[[0, 1, 1, 0],
                                   [1, 0, 0, 1]],
                           evidence=['S', 'D'],
                           evidence_card=[2, 2])
        self.add_cpds(cpd_S, cpd_D, cpd_U)

        self.assign_cpd('D')


three_node = ThreeNode()