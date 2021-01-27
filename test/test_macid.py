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

from examples.simple_macids import get_basic2agent_acyclic, get_basic2agent_cyclic, get_basic_subgames, \
get_basic_subgames2, get_basic_subgame3
from examples.story_macids import subgame_difference
from core.get_paths import find_active_path
import networkx as nx
import itertools
import copy


class TestMACID(unittest.TestCase):

    @unittest.skip("")
    def test_create_macid(self):
        example = get_basic_subgames()
        example.draw()
        print(example.get_SCCs()[0])
        print(example.get_SCCs()[1])
        print(example.get_SCCs()[2])

    # # @unittest.skip("")
    # def test_get_SCCs(self):
    #     example = get_basic2agent_cyclic()
    #     self.assertEqual(*example.get_SCCs(), {'D1', 'D2'})
    #     example2 = get_basic_subgames2()
    #     self.assertTrue(len(example2.get_SCCs())==3)

    # # @unittest.skip("")
    # def test_condensed_relevance_graph(self):
    #     example = get_basic_subgames2()
    #     a = example.condensed_relevance_graph()
    #     print(a.graph['mapping'])
    
    def test_subgame(self):
        example = get_basic_subgames2()
        example.draw()
        example.draw_strategic_rel_graph()
        example.draw_SCCs()
        example.decision_nodes_in_subgames()
        # crg = example.condensed_relevance_graph()

        # crg_map = crg.graph['mapping']
        # scc_map = {}
        # for k, v in crg_map.items():
        #     scc_map[v] = scc_map.get(v, []) + [k]
        # print(f"SCC map = {scc_map}")
        # tpg_list = list(nx.topological_sort(crg))
        # print(tpg_list)

        # print(crg.nodes)
        # s = crg.nodes
        # #poweset excluding the empty set
        # powerset_nodes = itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s)+1))

        # new = list(powerset_nodes)
        # new2 = copy.deepcopy(new)

        # print(new)
        # for subset in new:

        #    # print(f"subset: {subset}")
        #     for node in subset:
        #         # print(f"node: {node} subset: {subset}")
        #         # print(f"desc: {nx.descendants(crg, node)}")
        #         if nx.descendants(crg, node).issubset(subset):
        #             print("is subset")
        #         else:
        #             #  print("is not subset")
        #             if subset in new2:
        #                 new2.remove(subset)
        #             #break
        # print(new2)

    @unittest.skip("")   
    def test_temp(self):
        example = get_basic_subgames2()
        # example.draw()

        # example.draw_strategic_rel_graph()
        # example.draw_SCCs()
        l = example.condensed_relevance_graph()
        print(l.graph['mapping'])
        a = l.graph['mapping']

        scc_map = {}
        for k, v in a.items():
            scc_map[v] = scc_map.get(v, []) + [k]
        print(f"SCC map = {scc_map}")

        print(l.nodes)
        s = l.nodes
        #poweset excluding the empty set
        powerset_nodes = itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s)+1))

        new = list(powerset_nodes)

        # for item in new:
        #     print(item)

        print(new)

        # l2 = nx.to_undirected(l)
        print(list(nx.weakly_connected_components(l)))

        # for subset in new:

        #     print(f"subset: {subset}")
        #     for node in subset:
        #         print(f"node: {node} subset: {subset}")
        #         print(f"desc: {nx.descendants(l, node)}")
        #         if nx.descendants(l, node).issubset(subset):
        #             print("is subset")
        #         else:
        #             print("is not subset")


        # dec_nodes = []
        # for subset in new:
        #     print(f"subset: {subset}")
        #     for node in subset:
        #         print(f"{node} got here")
        #         print(f"desc: {nx.descendants(l, node)}")
        #         print(nx.descendants(l, node).issubset(set(subset)))
        #         if not nx.descendants(l, node).issubset(set(subset)):
        #             print(f"node: {node} breaking")
        #             new.remove(subset)
            
        # print(new)

        # dec_nodes = []
        # for subset in new:
        #     print(f"subset: {subset}")
        #     for node in subset:
        #         print(f"{node} got here")
        #         if nx.descendants(l, node).issubset(set(subset)):
        #             print(f"node: {node} breaking")
        #             continue
        #         new.remove(subset)
            

        # print(new)      



    # def _powerset(iterable):
    #     "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    #     s = list(iterable)
    #     return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

        for k in scc_map.keys():
            print(f"{k}'s descendents are {nx.descendants(l,k)}")

        #subgame_decs = [{k}.union(nx.descendants(l, k)) for k in scc_map.keys()]

        crg_subgames = [{node}.union(nx.descendants(l, node)) for node in l.nodes]

        print(crg_subgames)

        dec_subgames = [[scc_map[i] for i in crg_subgame] for crg_subgame in crg_subgames]
        
        print([set(itertools.chain.from_iterable(i)) for i in dec_subgames])


        #print(dec_subgames)
        

        
        # print(f"keys are {list(inv_map.keys())}")


        # print(f"desc are {nx.descendants(l,2)}") 

        
        # print(list(nx.topological_sort(l)))
        # example2 = example.copy_without_cpds()
        # example2.draw()

        # example3 = example.mechanism_graph()
        # example3.draw()



        # example4 = get_3node_cid()
        # example4.draw()
        # example5 = example4.mechanism_graph()
        # example5.draw()


        # example2 = example.mechanism_graph()
        # example2.draw()
        # example2.remove_nodes_from(['U1_Amec', 'U2_Amec', 'U1_Bmec', 'U2_Bmec', 'Nmec'])
        # example2.draw()
        # print(find_active_path(example2, 'D2mec', 'U1_B', ['D1', 'N', 'D1mec']))
        # print(find_active_path(example2, 'D1mec', 'U2_A', ['D2', 'D1', 'D2mec']))




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



