# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.
# %%
import sys
import os

from numpy.lib.function_base import append
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
import unittest
from examples.simple_macids import basic2agent, basic2agent_tie_break, basic_different_dec_cardinality, battle_of_the_sexes, \
    get_basic2agent_cyclic, get_basic_subgames, get_basic_subgames2, get_basic_subgames3, matching_pennies, modified_taxi_competition, prisoners_dilemma, prisoners_dilemma2, taxi_competition, two_agent_no_pne, \
    two_agent_one_pne, two_agent_two_pne, two_agents_three_actions

import numpy as np
from core.relevance_graph import RelevanceGraph, CondensedRelevanceGraph
import networkx as nx

class TestMACID(unittest.TestCase):

    @unittest.skip("")
    def test_get_sccs(self) -> None:
        macid = get_basic2agent_cyclic()
        self.assertEqual(macid.get_sccs(), [{'D1', 'D2'}])
        macid = get_basic_subgames2()
        self.assertTrue(len(macid.get_sccs()) == 3)

    @unittest.skip("")
    def test_all_maid_subgames(self) -> None:
        macid = get_basic2agent_cyclic()
        self.assertCountEqual(macid.all_maid_subgames(), [{'D1', 'D2'}])
        macid = get_basic_subgames()
        self.assertTrue(len(macid.all_maid_subgames()) == 4)
        macid = get_basic_subgames3()
        self.assertTrue(len(macid.all_maid_subgames()) == 5)

    @unittest.skip("")
    def test_get_all_pure_spe(self) -> None:
        macid = basic2agent_tie_break()
        self.assertTrue(len(macid.get_all_pure_spe()) == 2)
        macid2 = basic2agent()
        self.assertEqual(macid2.get_all_pure_spe(), [[('D1', [], 1), ('D2', [('D1', 0)], 1), ('D2', [('D1', 1)], 0)]])
        macid3 = basic_different_dec_cardinality()
        self.assertEqual(macid3.get_all_pure_spe(), [[('D1', [], 1), ('D2', [('D1', 0)], 1), ('D2', [('D1', 1)], 2)]])

    @unittest.skip("")
    def test_policy_profile_assignment(self) -> None:
        macid = taxi_competition()
        macid.impute_random_decision('D1')
        cpd = macid.get_cpds('D1')
        partial_policy = [cpd]
        policy_assignment = macid.policy_profile_assignment(partial_policy)
        self.assertTrue(policy_assignment['D1'])
        self.assertFalse(policy_assignment['D2'])
        macid.impute_fully_mixed_policy_profile()
        joint_policy = [macid.get_cpds(d) for d in macid.all_decision_nodes]
        joint_policy_assignment = macid.policy_profile_assignment(joint_policy)
        self.assertTrue(joint_policy_assignment['D1'])
        self.assertTrue(joint_policy_assignment['D2'])
        d1_cpd = joint_policy_assignment['D1']
        self.assertEqual(d1_cpd.state_names, {'D1': ['e', 'c']})
        print(d1_cpd.state_names)  # can put this in the notebook too
        self.assertTrue(np.array_equal(d1_cpd.values, np.array([0.5, 0.5])))

    @unittest.skip("")
    def test_get_all_pure_ne(self) -> None:
        macid = prisoners_dilemma()
        self.assertEqual(len(macid.get_all_pure_ne()), 1)
        pne = macid.get_all_pure_ne()[0]
        macid.add_cpds(*pne)
        self.assertEqual(macid.expected_utility({}, agent=1), -2)
        self.assertEqual(macid.expected_utility({}, agent=2), -2)   
        
        macid2 = battle_of_the_sexes()
        self.assertEqual(len(macid2.get_all_pure_ne()), 2)
        
        macid3 = matching_pennies()
        self.assertEqual(len(macid3.get_all_pure_ne()), 0)

        macid4 = two_agents_three_actions()
        self.assertEqual(len(macid4.get_all_pure_ne()), 1)     
        
    @unittest.skip("")
    def test_get_all_pure_ne_in_sg(self):
        macid = taxi_competition()
        ne_in_subgame = macid.get_all_pure_ne_in_sg(decisions_in_sg=['D2'])
        policy_assignment = macid.policy_profile_assignment(ne_in_subgame[0])
        cpd_d2 = policy_assignment['D2']
        self.assertTrue(np.array_equal(cpd_d2.values, np.array([[0,1], [1,0]])))
        self.assertFalse(policy_assignment['D1'])
        ne_in_full_macid = macid.get_all_pure_ne_in_sg()
        self.assertEqual(len(ne_in_full_macid), 3)
        with self.assertRaises(Exception):
            macid.get_all_pure_ne_in_sg(decisions_in_sg=['D3'])

    

        
        
        # all_spe = macid.get_all_pure_spe()
        # print(f"len is {len(all_spe)}")
        # macid.add_cpds(*all_spe[0])
        






        # print(policy_assignment)



        # print(bool(policy_assignment['D1']))
        # print(bool(policy_assignment['D2']))




        # macid = prisoners_dilemma()
        # pne = macid.get_all_pure_ne()
        # jp = macid.policy_profile_assignment(pne[0])
        # cpd = jp['D1']
        # self.assertTrue(np.array_equal(cpd.values, np.array([0, 1])))
        # cpd = jp['D2']
        # self.assertTrue(np.array_equal(cpd.values, np.array([0, 1])))



        # # put this in the notebook:
        # macid = taxi_competition()
        # macid.impute_fully_mixed_policy_profile()
        # macid.intervene({'D1': 'c'})
        # macid.intervene({'D2': 'e'})
        # cpd = macid.get_cpds('D1')
        # print(f"D1 is {cpd.values}")
        # cpd = macid.get_cpds('D2')

        # print(f"D2 is {cpd.values}")
        # print("util 1")
        # print(macid.expected_utility({}, agent = 1))
        # print("util 2")
        # print(macid.expected_utility({}, agent = 2))

        # print("-------------")
        # new_macid = prisoners_dilemma2()
        # # new_macid.impute_fully_mixed_policy_profile()
        # # print(new_macid.expected_utility({'D1': 'd'}, agent = 1))
        # print(new_macid.expected_utility({'D1': 0, 'D2': 0}, agent = 2))
        # print(new_macid.expected_utility({'D1': 0, 'D2': 0}, agent = 1))









        #print({d: macid.get_cpds(d) for d in macid.all_decision_nodes})

       # cpd_b = FunctionCPD('D1', lambda a: a, evidence=['A'])



        # decisions_in_subgame = ['D2']
        # pne = macid.get_all_pure_ne_in_sg(decisions_in_subgame)
        # jp = macid.policy_profile_assignment(pne[0])
        
        # cpd = jp['D2']
        # print(f"vallues 1b = {cpd.values}")
        # print(f"jp is {jp}")
        
        
        # jp = macid.joint_policy_assignment(pne[1])
        # cpd = jp['D1']
        # print(f"vallues 2 = {cpd.values}")

        # cpd = jp['D2']
        # print(f"vallues 2b = {cpd.values}")
        # jp = macid.joint_policy_assignment(pne[2])
        # cpd = jp['D1']

        # print(f"vallues 3 = {cpd.values}")
        # cpd = jp['D2']
        # print(f"vallues 3b = {cpd.values}")








        # macid = taxi_competition()
        # pne = macid.get_all_pure_ne2(['D2'])

        # new_macid = macid.copy_without_cpds()
        # jp = new_macid.joint_policy_assignment(pne[0])
        # cpd = jp['D2']
        # print(f"new = {cpd.values}")
        # print(f"jp = {jp}")


        # macid = taxi_competition()
        # partial_ne = macid.get_all_pure_ne2(['D2'])
        # pp = macid.partial_policy_assignment(partial_ne[0])
        # cpd = pp['D2']
        # print(f"new = {cpd.values}")
        # print(f"pp = {pp}")

        

        
        # new_macid = macid.copy_without_cpds()
        # new_macid.add_cpds(*pne[0])
        # print({d: new_macid.get_cpds(d) for d in new_macid.all_decision_nodes})
        # a = [d for d in new_macid.all_decision_nodes if not new_macid.get_cpds(d)]
        # print(f"a is {a}")
        # list_cpds = [new_macid.get_cpds(d) for d in new_macid.all_decision_nodes]
        # print(f"{list_cpds[0]}")
        # print(list_cpds)






        # jp = macid.joint_policy_assignment(pne[0])
        # print(jp)
        # for k, v in jp.items():
        #     if '<UniformRandomCPD' in str(v):
        #         print(k)


        # cpd = jp['D2']
        # print(f"new = {cpd.values}")

        #pne = macid.get_all_pure_ne()[0]
        #macid.add_cpds(*pne)
        #print(macid.expected_utility({}, agent=1))
        #self.assertEqual(macid.expected_utility({}, agent=1), -2)





        # self.assertTrue(np.array_equal(cpd.values, np.array([0, 1])))
        # cpd = jp['D2']
        # self.assertTrue(np.array_equal(cpd.values, np.array([[0, 0],
        #                                                     [1, 1]])))


    # @unittest.skip("")
    def test_get_all_pure_spe(self) -> None:
        macid = taxi_competition()
        all_spe = macid.get_all_pure_spe()
        self.assertTrue(len(all_spe)==1)
        spe = all_spe[0]
        joint_policy = macid.policy_profile_assignment(spe)
        cpd_d1 = joint_policy['D1']
        cpd_d2 = joint_policy['D2']
        self.assertTrue(np.array_equal(cpd_d1.values, np.array([1,0])))
        self.assertTrue(np.array_equal(cpd_d2.values, np.array([[0,1], [1,0]])))

        macid = prisoners_dilemma()
        all_spe = macid.get_all_pure_spe()
        self.assertTrue(len(all_spe)==1)

        macid = battle_of_the_sexes()
        all_spe = macid.get_all_pure_spe()
        self.assertTrue(len(all_spe)==2)




        


        # spe = all_spe[1]
        # jp = macid.joint_policy_assignment(spe)
        # cpd = jp['D2']
        # print(f"new D2 = {cpd.values}")

        # cpd = jp['D1']
        # print(f"new_D1 = {cpd.values}")


        


        # rg = RelevanceGraph(macid)
        # rg.draw()
        # crg = CondensedRelevanceGraph(macid)
        # crg.draw()
        # print(crg.graph['mapping'])

        # dec_scc_mapping = crg.graph['mapping']
        # scc_dec_mapping = {}
        # for k, v in dec_scc_mapping.items():
        #     scc_dec_mapping[v] = scc_dec_mapping.get(v, []) + [k]
        # print(scc_dec_mapping)
        # print(list(nx.topological_sort(crg)))

        # rev_top_order = reversed(list(nx.topological_sort(crg)))
        # spes = [[]]

        # for scc in rev_top_order:
        #     extended_spes = []
        #     for partial_profile in spes:
        #         macid.add_cpds(*partial_profile)
        #         for ne in macid.get_all_pure_ne(scc): 
        #             extended_spes.append(partial_profile + ne)
        #     spes = extended_spes
        # return spes    




        # matching = two_agents_three_actions()
        # matching.impute_fully_mixed_policy_profile()
        # print(f" ------------ {matching.get_cpds('D1').values}")
        # pne_new = matching.get_all_pure_ne()[0]
        # print(f"len is {len(pne_new)}")
        # pne = matching.get_all_pure_ne()[0]
        # matching.add_cpds(*pne)

        # print(matching.expected_utility({}, agent=1))
        # print(matching.expected_utility({}, agent=2))
        # print(matching.expected_value(['U1'], {}))
        # print(matching.get_cpds('D1').possible_values(matching))
        # print(matching.get_cpds('D1').get_values())
        # print(matching.get_cpds('D1'))
        # print(matching.get_cpds('D1').state_names)

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMACID)
    unittest.TextTestRunner().run(suite)
