# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.
import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
import unittest
from examples.simple_cids import get_3node_cid
from examples.simple_macids import get_basic2agent, get_basic_subgames, get_path_example
from analyze.get_paths import backdoor_path_active_when_conditioning_on_W, find_active_path, get_motifs, \
    get_motif, find_all_dir_paths, find_all_undir_paths, directed_decision_free_path, path_d_separated_by_Z, \
    frontdoor_indirect_path_not_blocked_by_W, parents_of_Y_not_descended_from_X
from core.macid import MACID


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
        self.assertTrue(len(find_all_dir_paths(example, 'D2', 'U2')) == 2)

    # @unittest.skip("")
    def test_find_all_undir_paths(self):
        example = get_3node_cid()
        self.assertTrue(len(find_all_undir_paths(example, 'S', 'U')) == 2)

        example2 = MACID([
            ('X1', 'D'),
            ('X2', 'U')],
            {1: {'D': ['D'], 'U': ['U']}})
        self.assertEqual(*find_all_undir_paths(example2, 'X1', 'D'), ['X1', 'D'])
        self.assertFalse(find_all_undir_paths(example2, 'X1', 'U'))

    # @unittest.skip("")
    def test_directed_decision_free_path(self):
        example = get_basic_subgames()
        self.assertTrue(directed_decision_free_path(example, 'X1', 'U11'))
        self.assertTrue(directed_decision_free_path(example, 'X2', 'U22'))
        self.assertFalse(directed_decision_free_path(example, 'X2', 'U3'))
        self.assertFalse(directed_decision_free_path(example, 'X2', 'U2'))
        self.assertFalse(directed_decision_free_path(example, 'U22', 'U3'))

    # @unittest.skip("")
    def test_path_d_seperated_by_Z(self):
        example = get_path_example()
        self.assertFalse(path_d_separated_by_Z(example, ['X1', 'D', 'U']))
        self.assertTrue(path_d_separated_by_Z(example, ['X1', 'D', 'U'], ['D']))
        self.assertTrue(path_d_separated_by_Z(example, ['X1', 'D', 'X2']))
        self.assertFalse(path_d_separated_by_Z(example, ['X1', 'D', 'X2'], ['D']))
        self.assertFalse(path_d_separated_by_Z(example, ['X1', 'D', 'X2'], ['U']))

    # @unittest.skip("")
    def test_frontdoor_indirect_path_not_blocked_by_W(self):
        example = get_path_example()
        self.assertTrue(frontdoor_indirect_path_not_blocked_by_W(example, 'X2', 'X1', ['D']))
        self.assertFalse(frontdoor_indirect_path_not_blocked_by_W(example, 'X2', 'X1'))
        self.assertFalse(frontdoor_indirect_path_not_blocked_by_W(example, 'X3', 'X1', ['D']))
        self.assertFalse(frontdoor_indirect_path_not_blocked_by_W(example, 'X3', 'X1'))
        self.assertFalse(frontdoor_indirect_path_not_blocked_by_W(example, 'X1', 'U'))
        self.assertFalse(frontdoor_indirect_path_not_blocked_by_W(example, 'X1', 'U', ['D', 'X2']))

    # @unittest.skip("")
    def test_parents_of_Y_not_descended_from_X(self):
        example = get_path_example()
        self.assertCountEqual(parents_of_Y_not_descended_from_X(example, 'U', 'X1'), ['X2'])
        self.assertCountEqual(parents_of_Y_not_descended_from_X(example, 'U', 'X2'), ['X2'])
        self.assertCountEqual(parents_of_Y_not_descended_from_X(example, 'U', 'X3'), ['D', 'X2'])

    # @unittest.skip("")
    def test_backdoor_path_active_when_conditioning_on_W(self):
        example = get_path_example()
        self.assertFalse(backdoor_path_active_when_conditioning_on_W(example, 'X3', 'X2'))
        self.assertTrue(backdoor_path_active_when_conditioning_on_W(example, 'X3', 'X2', ['D']))
        self.assertFalse(backdoor_path_active_when_conditioning_on_W(example, 'X1', 'X2'))
        self.assertFalse(backdoor_path_active_when_conditioning_on_W(example, 'X1', 'X2', ['D']))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPATHS)
    unittest.TextTestRunner().run(suite)
