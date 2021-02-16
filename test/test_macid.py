# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.
# %%
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
import unittest
from examples.simple_macids import basic2agent, basic2agent_tie_break, basic_different_dec_cardinality, \
    get_basic2agent_cyclic, get_basic_subgames, get_basic_subgames2, get_basic_subgames3


class TestMACID(unittest.TestCase):

    # @unittest.skip("")
    def test_get_sccs(self) -> None:
        macid = get_basic2agent_cyclic()
        self.assertEqual(*macid.get_sccs(), {'D1', 'D2'})
        macid = get_basic_subgames2()
        self.assertTrue(len(macid.get_sccs()) == 3)

    # @unittest.skip("")
    def test_all_maid_subgames(self) -> None:
        macid = get_basic2agent_cyclic()
        self.assertCountEqual(*macid.all_maid_subgames(), {'D1', 'D2'})
        macid = get_basic_subgames()
        self.assertTrue(len(macid.all_maid_subgames()) == 4)
        macid = get_basic_subgames3()
        self.assertTrue(len(macid.all_maid_subgames()) == 5)

    def test_get_all_pure_spe(self) -> None:
        macid = basic2agent_tie_break()
        self.assertTrue(len(macid.get_all_pure_spe()) == 2)
        macid2 = basic2agent()
        self.assertEqual(macid2.get_all_pure_spe(), [[('D1', [], 1), ('D2', [('D1', 0)], 1), ('D2', [('D1', 1)], 0)]])
        macid3 = basic_different_dec_cardinality()
        self.assertEqual(macid3.get_all_pure_spe(), [[('D1', [], 1), ('D2', [('D1', 0)], 1), ('D2', [('D1', 1)], 2)]])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMACID)
    unittest.TextTestRunner().run(suite)
