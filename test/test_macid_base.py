# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.

import sys, os
sys.path.insert(0, os.path.abspath('.'))
import unittest
import numpy as np
from examples.simple_cids import get_3node_cid, get_5node_cid, get_5node_cid_with_scaled_utility, get_2dec_cid, \
    get_minimal_cid
from examples.story_cids import get_introduced_bias
from pgmpy.factors.discrete import TabularCPD


class TestBASE(unittest.TestCase):

    # @unittest.skip("")
    def test_assign_cpd(self):
        three_node = get_3node_cid()
        three_node.add_cpds(TabularCPD('D', 2, np.eye(2), evidence=['S'], evidence_card=[2]))
        three_node.check_model()
        cpd = three_node.get_cpds('D').values
        self.assertTrue(np.array_equal(cpd, np.array([[1, 0], [0, 1]])))

    def test_query(self):
        three_node = get_3node_cid()
        with self.assertRaises(Exception):
            three_node._query(['U'], {})
        with self.assertRaises(Exception):
            three_node._query(['U'], {'D': 0})

    # @unittest.skip("")
    def test_expected_utility(self):
        three_node = get_3node_cid()
        five_node = get_5node_cid()
        eu00 = three_node.expected_utility({'D': 0, 'S': 0})
        self.assertEqual(eu00, 1)
        eu10 = three_node.expected_utility({'D': 1, 'S': 0})
        self.assertEqual(eu10, 0)
        eu000 = five_node.expected_utility({'D': 0, 'S1': 0, 'S2': 0})
        self.assertEqual(eu000, 2)
        eu001 = five_node.expected_utility({'D': 0, 'S1': 0, 'S2': 1})
        self.assertEqual(eu001, 1)

    # @unittest.skip("")
    def test_intervention(self):
        cid = get_minimal_cid()
        cid.impute_random_policy()
        self.assertEqual(cid.expected_value(['B'], {})[0], 0.5)
        for a in [0, 1, 2]:
            cid.intervene({'A': a})
            self.assertEqual(cid.expected_value(['B'], {})[0], a)
        self.assertEqual(cid.expected_value(['B'], {}, intervene={'A': 1})[0], 1)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestBASE)
    unittest.TextTestRunner().run(suite)
