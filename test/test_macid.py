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

from examples.simple_macids import get_basic2agent, get_basic2agent2, get_basic_subgames, get_basic_subgames2

class TestMACID(unittest.TestCase):

    @unittest.skip("")
    def test_create_macid(self):
        example = get_basic_subgames()
        example.draw()
        print(example.get_SCCs()[0])
        print(example.get_SCCs()[1])
        print(example.get_SCCs()[2])

    # @unittest.skip("")
    def test_get_SCCs(self):
        example = get_basic_subgames2()
        #self.assertTrue(len(example.get_SCCs())==3)
        example.draw()
        example.draw_strategic_rel_graph()

        example.draw_SCCs()
        # # TODO: change subgame example
        # mg = example.mechanism_graph()
        # print(path_d_separated_by_Z(mg, ['D11mec', 'D11', 'X1', 'U11'], ['D12', 'D11', 'X2', 'D12mec']))
        # print(path_d_separated_by_Z(mg, ['D11mec', 'D11', 'X1', 'U11'], ['D11']))


        # example3 = example_temp()
        # example3.draw()
        

        # print(path_d_separated_by_Z(example3, ['D1mec', 'D1', 'X1', 'U1'], ['D1']))








    


    

       



       
       
       
       
       
       
       
       
       
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



