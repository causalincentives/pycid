# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.
#%%
import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
import unittest
import numpy as np
from examples.simple_cids import get_3node_cid, get_5node_cid, get_5node_cid_with_scaled_utility, get_2dec_cid, \
    get_minimal_cid
from examples.story_cids import get_introduced_bias
from pgmpy.factors.discrete import TabularCPD

from examples.simple_macids import get_basic2agent, basic2agent_2

class TestMACID(unittest.TestCase):

    # @unittest.skip("")
    def test_create_macid(self):

        basic2agent = get_basic2agent()
        basic2agent.draw()


    # @unittest.skip("")
    def test_assign_cpd(self):
        example = basic2agent_2()
        example.add_cpds(TabularCPD('D1', 2, np.array([[0], [1]])))
        #example.add_cpds(TabularCPD('D2', 2, [0,1]))
        cpd = example.get_cpds('D1')
        print(cpd)
       

       
       
       
       
       
       
       
       
       
        #self.assertTrue(np.array_equal(cpd, np.array([[1, 0], [0, 1]])))

#     # @unittest.skip("")
#     def test_assign_cpd(self):
#         basic2 = basic2agent_2()
#         three_node.add_cpds(TabularCPD('D', 2, np.eye(2), evidence=['S'], evidence_card=[2]))
#         three_node.check_model()
#         print(three_node.all_utility_nodes)
#         three_node.draw()
#         cpd = three_node.get_cpds('D').values
#         self.assertTrue(np.array_equal(cpd, np.array([[1, 0], [0, 1]])))

# basic2agent_2()

    # # @unittest.skip("")
    # def test_assign_cpd(self):
    #     basic2agent = get_basic2agent()
    #     basic2agent.draw()
    #     basic2agent.add_cpds(TabularCPD('D', 2, np.eye(2), evidence=['S'], evidence_card=[2]))
    #     three_node.check_model()
    #     print(three_node.all_utility_nodes)
    #     three_node.draw()
    #     cpd = three_node.get_cpds('D').values
    #     self.assertTrue(np.array_equal(cpd, np.array([[1, 0], [0, 1]])))



if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMACID)
    unittest.TextTestRunner().run(suite)



