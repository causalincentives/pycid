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
get_basic_subgames2, get_basic_subgames3
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

    # @unittest.skip("")
    def test_get_SCCs(self):    
        example = get_basic2agent_cyclic()
        self.assertEqual(*example.get_SCCs(), {'D1', 'D2'})
        example2 = get_basic_subgames2()
        self.assertTrue(len(example2.get_SCCs())==3)

    # @unittest.skip("")
    def test_all_maid_subgames(self):
        example = get_basic2agent_cyclic()
        self.assertCountEqual(*example.all_maid_subgames(), {'D1', 'D2'})
        example2 = get_basic_subgames()
        self.assertTrue(len(example2.all_maid_subgames()) == 4)
        example3 = get_basic_subgames3()
        self.assertTrue(len(example3.all_maid_subgames()) == 5)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMACID)
    unittest.TextTestRunner().run(suite)



