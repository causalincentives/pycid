# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.
#%%
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
from core.macid import MACID
from analyze.reasoning_patterns import direct_effect, find_motivations, manipulation, revealing_or_denying, signaling
import unittest


class TestReasoning(unittest.TestCase):

    def test_direct_effect(self):
        macid = MACID([('D1', 'U'), ('D2', 'D1')],
                      {1: {'D': ['D1', 'D2'], 'U': ['U']}})
        self.assertTrue(direct_effect(macid, 'D1'))
        self.assertFalse(direct_effect(macid, 'D2'))
        with self.assertRaises(Exception):
            direct_effect(macid, 'D3')

    def test_manipulation(self):
        macid = MACID([('D1', 'U2'), ('D1', 'D2'), ('D2', 'U1'), ('D2', 'U2')],
                      {1: {'D': ['D1'], 'U': ['U1']}, 2: {'D': ['D2'], 'U': ['U2']}})
        effective_set = ['D2']  # by direct effect
        self.assertTrue(manipulation(macid, 'D1', effective_set))
        self.assertFalse(manipulation(macid, 'D2', effective_set))
        with self.assertRaises(Exception):
            manipulation(macid, 'D3', effective_set)
        effective_set2 = ['A']
        with self.assertRaises(Exception):
            manipulation(macid, 'D1', effective_set2)

    def test_signaling(self):
        macid = MACID([('X', 'U1'), ('X', 'U2'),
                       ('X', 'D1'), ('D1', 'D2'),
                       ('D2', 'U1'), ('D2', 'U2')],
                      {1: {'D': ['D1'], 'U': ['U1']}, 2: {'D': ['D2'], 'U': ['U2']}})
        effective_set = ['D2']  # by direct effect
        self.assertTrue(signaling(macid, 'D1', effective_set))
        self.assertFalse(signaling(macid, 'D2', effective_set))
        with self.assertRaises(Exception):
            signaling(macid, 'D3', effective_set)
        effective_set2 = ['A']
        with self.assertRaises(Exception):
            signaling(macid, 'D1', effective_set2)

    def test_revealing_or_denying(self):
        macid = MACID([('D1', 'X2'), ('X1', 'X2'),
                       ('X2', 'D2'), ('D2', 'U1'),
                       ('D2', 'U2'), ('X1', 'U2')],
                      {1: {'D': ['D1'], 'U': ['U1']}, 2: {'D': ['D2'], 'U': ['U2']}})
        effective_set = ['D2']  # by direct effect
        self.assertTrue(revealing_or_denying(macid, 'D1', effective_set))
        self.assertFalse(revealing_or_denying(macid, 'D2', effective_set))
        with self.assertRaises(Exception):
            revealing_or_denying(macid, 'D3', effective_set)
        effective_set2 = ['A']
        with self.assertRaises(Exception):
            revealing_or_denying(macid, 'D1', effective_set2)

    def test_motivations(self):
        macid = MACID([('D1', 'U'), ('D2', 'D1')],
                      {1: {'D': ['D1', 'D2'], 'U': ['U']}})
        self.assertEqual(find_motivations(macid)['dir_effect'], ['D1'])

        macid2 = MACID([('D1', 'U2'), ('D1', 'D2'), ('D2', 'U1'), ('D2', 'U2')],
                       {1: {'D': ['D1'], 'U': ['U1']}, 2: {'D': ['D2'], 'U': ['U2']}})
        self.assertEqual(find_motivations(macid2)['dir_effect'], ['D2'])
        self.assertEqual(find_motivations(macid2)['manip'], ['D1'])

        macid3 = MACID([('X', 'U1'), ('X', 'U2'),
                       ('X', 'D1'), ('D1', 'D2'),
                       ('D2', 'U1'), ('D2', 'U2')],
                       {1: {'D': ['D1'], 'U': ['U1']}, 2: {'D': ['D2'], 'U': ['U2']}})
        self.assertEqual(find_motivations(macid3)['dir_effect'], ['D2'])
        self.assertEqual(find_motivations(macid3)['sig'], ['D1'])

        macid4 = MACID([('D1', 'X2'), ('X1', 'X2'),
                       ('X2', 'D2'), ('D2', 'U1'),
                       ('D2', 'U2'), ('X1', 'U2')],
                       {1: {'D': ['D1'], 'U': ['U1']}, 2: {'D': ['D2'], 'U': ['U2']}})
        self.assertEqual(find_motivations(macid4)['dir_effect'], ['D2'])
        self.assertEqual(find_motivations(macid4)['rev_den'], ['D1'])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestReasoning)
    unittest.TextTestRunner().run(suite)
