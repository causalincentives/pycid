# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.
# %%
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
import unittest
from examples.simple_macids import basic2agent, basic2agent_tie_break, basic_different_dec_cardinality, battle_of_the_sexes, \
    get_basic2agent_cyclic, get_basic_subgames, get_basic_subgames2, get_basic_subgames3, matching_pennies, prisoners_dilemma, taxi_competition, two_agent_no_pne, \
    two_agent_one_pne, two_agent_two_pne, two_agents_three_actions

import numpy as np

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
    def test_pure_ne(self) -> None:
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

        

    # @unittest.skip("")
    def test_joint_policy_assignment(self) -> None:
        macid = prisoners_dilemma()
        pne = macid.get_all_pure_ne()
        jp = macid.joint_policy_assignment(pne[0])
        cpd = jp['D1']
        self.assertTrue(np.array_equal(cpd.values, np.array([0, 1])))
        cpd = jp['D2']
        self.assertTrue(np.array_equal(cpd.values, np.array([0, 1])))

        macid = taxi_competition()
        pne = macid.get_all_pure_ne()
        jp = macid.joint_policy_assignment(pne[0])
        cpd = jp['D1']
        self.assertTrue(np.array_equal(cpd.values, np.array([1, 0])))
        cpd = jp['D2']
        self.assertTrue(np.array_equal(cpd.values, np.array([[0, 0],
                                                            [1, 1]])))






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
