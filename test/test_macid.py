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

from examples.simple_macids import get_basic2agent, get_basic2agent2

class TestMACID(unittest.TestCase):

    # @unittest.skip("")
    def test_create_macid(self):
        basic2agent = get_basic2agent()
        basic2agent.draw()
        basic2agent.draw_strategic_rel_graph()

    # @unittest.skip("")
    def test_is_s_reachable(self):
        example = get_basic2agent()
        self.assertTrue(example.is_s_reachable('D1','D2'))
        self.assertFalse(example.is_s_reachable('D2','D1'))

    # @unittest.skip("") 
    def test_is_strategically_acyclic(self):
        example = get_basic2agent()
        self.assertTrue(example.is_strategically_acyclic())
        
        example2 = get_basic2agent2()
        self.assertFalse(example2.is_strategically_acyclic())
        
    # @unittest.skip("")

    def test_get_acyclic_topological_ordering(self):
        example = get_basic2agent()
        self.assertEqual(example.get_acyclic_topological_ordering(), ['D1', 'D2'])
          
        example2 = get_basic2agent2()
        with self.assertRaises(Exception):
            example2.get_acyclic_topological_ordering()


       
       
       
       
       
       
       
       
       
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



