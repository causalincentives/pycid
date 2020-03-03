# Copyright 2019 DeepMind Technologies Limited.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from cid import CID
from pgmpy.factors.discrete import TabularCPD


class FiveNode(CID):
    def __init__(self):
        super().__init__([('S1', 'D'),
                          ('S2', 'D'),
                          ('S1', 'U1'),
                          ('S2', 'U2'),
                          ('D', 'U1'),
                          ('D', 'U2')],
                         decision_nodes = ['D'],
                         utility_nodes = ['U1', 'U2'])

        cpd_S1 = TabularCPD(variable='S1', variable_card=2,
                            values=[[0.5],
                                    [0.5]])
        cpd_S2 = TabularCPD(variable='S2', variable_card=2,
                            values=[[0.5],
                                    [0.5]])
        cpd_U1 = TabularCPD(variable='U1', variable_card=2,
                            values=[[0, 1, 1, 0],
                                    [1, 0, 0, 1]],
                            evidence=['S1', 'D'],
                            evidence_card=[2, 2])
        cpd_U2 = TabularCPD(variable='U2', variable_card=2,
                            values=[[0, 1, 1, 0],
                                    [1, 0, 0, 1]],
                            evidence=['S2', 'D'],
                            evidence_card=[2, 2])
        cpd_D= TabularCPD(variable='D', variable_card=2,
                          values=[[0.5, 0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0.5, 0.5]],
                          evidence=['S1', 'S2'],
                          evidence_card=[2, 2])

        self.add_cpds(cpd_S1, cpd_S2, cpd_D, cpd_U1, cpd_U2)
