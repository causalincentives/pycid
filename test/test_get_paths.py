# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.
import logging
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
from examples.simple_cids import get_3node_cid
from examples.simple_macids import get_basic_subgames, get_path_example
from examples.story_macids import taxi_competition
from core.get_paths import is_active_backdoor_trail, find_active_path, get_motifs, \
    get_motif, find_all_dir_paths, find_all_undir_paths, directed_decision_free_path, \
    is_active_path, is_active_indirect_frontdoor_trail
from core.macid import MACID


class TestPATHS(unittest.TestCase):

    def setUp(self) -> None:
        logging.disable()

    # @unittest.skip("")
    def test_find_active_path(self) -> None:
        example = taxi_competition()
        self.assertEqual(find_active_path(example, 'D1', 'U1', ['D2']), ['D1', 'U1'])
        self.assertFalse(find_active_path(example, 'D1', 'U1', ['D2', 'U1']))
        with self.assertRaises(Exception):
            find_active_path(example, 'D1', 'U1', ['D3'])
        with self.assertRaises(Exception):
            find_active_path(example, 'D3', 'U1', ['D2'])
        with self.assertRaises(Exception):
            find_active_path(example, 'D1', 'U3', ['D2'])

    # @unittest.skip("")
    def test_get_motif(self) -> None:
        example = get_basic_subgames()
        self.assertEqual(get_motif(example, ['D3', 'D2', 'U2', 'D11', 'D12', 'U3'], 0), 'backward')
        self.assertEqual(get_motif(example, ['D3', 'D2', 'U2', 'D11', 'D12', 'U3'], 1), 'fork')
        self.assertEqual(get_motif(example, ['D3', 'D2', 'U2', 'D11', 'D12', 'U3'], 2), 'collider')
        self.assertEqual(get_motif(example, ['D3', 'D2', 'U2', 'D11', 'D12', 'U3'], 4), 'forward')
        self.assertEqual(get_motif(example, ['D3', 'D2', 'U2', 'D11', 'D12', 'U3'], 5), 'endpoint')
        with self.assertRaises(Exception):
            get_motif(example, ['D3', 'A', 'U2', 'D11', 'D12', 'U3'], 5)
        with self.assertRaises(Exception):
            get_motif(example, ['D3', 'D2', 'U2', 'D11', 'D12', 'U3'], 6)

    # @unittest.skip("")
    def test_get_motifs(self) -> None:
        example = get_basic_subgames()
        motifs = get_motifs(example, ['D3', 'D2', 'U2', 'D11', 'D12', 'U3'])
        self.assertEqual(motifs, ['backward', 'fork', 'collider', 'fork', 'forward', 'endpoint'])
        with self.assertRaises(Exception):
            get_motifs(example, ['D3', 'A', 'U2', 'D11', 'D12', 'U3'])

    # @unittest.skip("")
    def test_find_all_dir_paths(self) -> None:
        example = MACID([
            ('A', 'B'),
            ('B', 'C'),
            ('C', 'D'),
            ('D', 'E'),
            ('B', 'F'),
            ('F', 'E')],
            {1: {'D': ['D'], 'U': ['E']}})
        self.assertEqual(find_all_dir_paths(example, 'A', 'E'), [['A', 'B', 'C', 'D', 'E'], ['A', 'B', 'F', 'E']])
        self.assertEqual(find_all_dir_paths(example, 'C', 'E'), [['C', 'D', 'E']])
        self.assertFalse(find_all_dir_paths(example, 'F', 'A'))
        self.assertTrue(len(find_all_dir_paths(example, 'B', 'E')) == 2)
        with self.assertRaises(Exception):
            find_all_dir_paths(example, 'U2', 'A')

    # @unittest.skip("")
    def test_find_all_undir_paths(self) -> None:
        example = get_3node_cid()
        self.assertTrue(len(find_all_undir_paths(example, 'S', 'U')) == 2)
        with self.assertRaises(Exception):
            find_all_undir_paths(example, 'S', 'A')

        example2 = MACID([
            ('X1', 'D'),
            ('X2', 'U')],
            {1: {'D': ['D'], 'U': ['U']}})
        self.assertEqual(find_all_undir_paths(example2, 'X1', 'D'), [['X1', 'D']])
        self.assertFalse(find_all_undir_paths(example2, 'X1', 'U'))
        example3 = MACID([
            ('A', 'B'),
            ('B', 'C'),
            ('C', 'D'),
            ('D', 'E'),
            ('B', 'F'),
            ('F', 'E')],
            {1: {'D': ['D'], 'U': ['E']}})
        self.assertCountEqual(find_all_undir_paths(example3, 'F', 'A'),
                              [['F', 'E', 'D', 'C', 'B', 'A'], ['F', 'B', 'A']])

    # @unittest.skip("")
    def test_directed_decision_free_path(self) -> None:
        example = get_basic_subgames()
        self.assertTrue(directed_decision_free_path(example, 'X1', 'U11'))
        self.assertTrue(directed_decision_free_path(example, 'X2', 'U22'))
        self.assertFalse(directed_decision_free_path(example, 'X2', 'U3'))
        self.assertFalse(directed_decision_free_path(example, 'X2', 'U2'))
        self.assertFalse(directed_decision_free_path(example, 'U22', 'U3'))
        with self.assertRaises(Exception):
            directed_decision_free_path(example, 'X1', 'A')

    # @unittest.skip("")
    def test_is_active_path(self) -> None:
        example = get_path_example()
        self.assertTrue(is_active_path(example, ['X1', 'D', 'U']))
        self.assertFalse(is_active_path(example, ['X1', 'D', 'U'], ['D']))
        self.assertFalse(is_active_path(example, ['X1', 'D', 'X2']))
        self.assertTrue(is_active_path(example, ['X1', 'D', 'X2'], ['D']))
        self.assertTrue(is_active_path(example, ['X1', 'D', 'X2'], ['U']))
        with self.assertRaises(Exception):
            is_active_path(example, ['X1', 'D', 'A'], ['U'])
        with self.assertRaises(Exception):
            is_active_path(example, ['X1', 'D', 'X2'], ['A'])

    # @unittest.skip("")
    def test_is_active_indirect_frontdoor_trail(self) -> None:
        example = get_path_example()
        self.assertTrue(is_active_indirect_frontdoor_trail(example, 'X2', 'X1', ['D']))
        self.assertFalse(is_active_indirect_frontdoor_trail(example, 'X2', 'X1'))
        self.assertFalse(is_active_indirect_frontdoor_trail(example, 'X3', 'X1', ['D']))
        self.assertFalse(is_active_indirect_frontdoor_trail(example, 'X3', 'X1'))
        self.assertFalse(is_active_indirect_frontdoor_trail(example, 'X1', 'U'))
        self.assertFalse(is_active_indirect_frontdoor_trail(example, 'X1', 'U', ['D', 'X2']))
        with self.assertRaises(Exception):
            is_active_indirect_frontdoor_trail(example, 'A', 'U', ['D', 'X2'])
        with self.assertRaises(Exception):
            is_active_indirect_frontdoor_trail(example, 'X1', 'U', ['A', 'X2'])

    # @unittest.skip("")
    def test_is_active_backdoor_trail(self) -> None:
        example = get_path_example()
        self.assertFalse(is_active_backdoor_trail(example, 'X3', 'X2'))
        self.assertTrue(is_active_backdoor_trail(example, 'X3', 'X2', ['D']))
        self.assertFalse(is_active_backdoor_trail(example, 'X1', 'X2'))
        self.assertFalse(is_active_backdoor_trail(example, 'X1', 'X2', ['D']))
        with self.assertRaises(Exception):
            self.assertTrue(is_active_backdoor_trail(example, 'A', 'X2', ['D']))
        with self.assertRaises(Exception):
            self.assertTrue(is_active_backdoor_trail(example, 'X3', 'X2', ['A']))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPATHS)
    unittest.TextTestRunner().run(suite)

# %%
