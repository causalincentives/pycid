# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.
#%%
import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
import unittest
import numpy as np
from examples.simple_cids import get_3node_cid, get_5node_cid_with_scaled_utility, get_2dec_cid, \
    get_sequential_cid
from examples.story_cids import get_introduced_bias
from examples.simple_macids import get_basic2agent, get_basic_subgames
from analyze.get_paths import find_active_path, get_motifs, get_motif, find_all_dir_paths

class TestPATHS(unittest.TestCase):

    # @unittest.skip("")
    def test_find_active_path(self):
        example = get_basic2agent()
        self.assertEqual(find_active_path(example, 'D1', 'U1', ['D2']), ['D1', 'U1'])
        self.assertFalse(find_active_path(example, 'D1', 'U1', ['D2', 'U1']))

    # @unittest.skip("")
    def test_find_get_motif(self):
        example = get_basic_subgames()
        self.assertEqual(get_motif(example, ['D3', 'D2', 'U2', 'D11', 'D12', 'U3'], 0), 'backward')
        self.assertEqual(get_motif(example, ['D3', 'D2', 'U2', 'D11', 'D12', 'U3'], 1), 'fork')
        self.assertEqual(get_motif(example, ['D3', 'D2', 'U2', 'D11', 'D12', 'U3'], 2), 'collider')
        self.assertEqual(get_motif(example, ['D3', 'D2', 'U2', 'D11', 'D12', 'U3'], 4), 'forward')
        self.assertEqual(get_motif(example, ['D3', 'D2', 'U2', 'D11', 'D12', 'U3'], 5), 'endpoint')

    # @unittest.skip("")
    def test_find_get_motifs(self):
        example = get_basic_subgames()
        motifs = get_motifs(example, ['D3', 'D2', 'U2', 'D11', 'D12', 'U3'])
        self.assertEqual(motifs, ['backward', 'fork', 'collider', 'fork', 'forward', 'endpoint'])

    # @unittest.skip("")
    def test_find_all_dir_paths(self):
        example = get_basic_subgames()
        self.assertEqual(*find_all_dir_paths(example, 'D11', 'U3'), ['D11', 'D12', 'U3'])
        self.assertFalse(find_all_dir_paths(example, 'U2', 'D2'))
        self.assertTrue(len(find_all_dir_paths(example, 'D2', 'U2'))==2)

    



if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPATHS)
    unittest.TextTestRunner().run(suite)
