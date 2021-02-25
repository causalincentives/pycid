# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.
# %%
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
import unittest
from examples.simple_macids import basic2agent, basic2agent_tie_break, basic_different_dec_cardinality, \
    get_basic2agent_cyclic, get_basic_subgames, get_basic_subgames2, get_basic_subgames3, two_agent_no_pne, two_agent_one_pne, two_agent_two_pne

import itertools

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


    # def test_pure_ne(self):
    #     ne = []
    #     macid = two_agent_two_pne()
    #     all_dec_decision_rules = list(map(macid.possible_decision_rules, macid.all_decision_nodes))
    #     #possible_decision_rules = [[1,2,3], ['a','b','c']]
    #     #print(list(itertools.product(*dec_decision_rules)))
    #     for jp in list(itertools.product(*all_dec_decision_rules)):
    #         macid.add_cpds(*jp)
    #         print(f"here 0 {macid.expected_utility({}, agent=0)}")
    #         print(f"here 1 {macid.expected_utility({}, agent=1)}")
    #         for idx, decision in enumerate(macid.all_decision_nodes):
    #             agent_util = macid.expected_utility({}, agent=macid.whose_node[decision])
    #             opt_dr = macid.optimal_decision_rules(decision)
    #             macid.add_cpds(*opt_dr)
    #             opt_agent_util = macid.expected_utility({}, agent=macid.whose_node[decision])
    #             if agent_util == opt_agent_util:
    #                 print(f"they're the same")
    #             if agent_util != opt_agent_util:
    #                 break
    #         else:
    #             ne.append(jp)
    #     print(f"len is {len(ne)}")
    #     ne_jp = ne[1]
    #     macid.add_cpds(*ne_jp)
    #     print(macid.expected_utility({}, agent=0))
    #     print(macid.expected_utility({}, agent=1))






    def test_pure_ne2(self):
        
        #ne = []

        macid = two_agent_two_pne()
        

        print(len(macid.get_all_pure_ne()))
        





if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMACID)
    unittest.TextTestRunner().run(suite)
