# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.
#%%
import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
import unittest
import numpy as np
from examples.simple_cids import get_3node_cid, get_5node_cid, get_minimal_cid
from examples.simple_macids import get_basic2agent_acyclic, get_basic2agent_cyclic
from pgmpy.factors.discrete import TabularCPD
from examples.story_macids import forgetful_movie_star, subgame_difference


class TestBASE(unittest.TestCase):

    # @unittest.skip("")
    def test_assign_cpd(self):
        three_node = get_3node_cid()
        three_node.add_cpds(TabularCPD('D', 2, np.eye(2), evidence=['S'], evidence_card=[2]))
        three_node.check_model()
        cpd = three_node.get_cpds('D').values
        self.assertTrue(np.array_equal(cpd, np.array([[1, 0], [0, 1]])))

    # @unittest.skip("")
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

    # @unittest.skip("")
    def test_is_s_reachable(self):
        example = get_basic2agent_acyclic()
        self.assertTrue(example.is_s_reachable('D1', 'D2'))
        self.assertFalse(example.is_s_reachable('D2', 'D1'))

        example2 = subgame_difference()
        example2.draw()
        example2.draw_strategic_rel_graph()
        self.assertTrue(example2.is_s_reachable('D1', 'D2'))
        self.assertFalse(example2.is_s_reachable('D2', 'D1'))

    # @unittest.skip("")
    def test_is_full_rg_strategically_acyclic(self):
        example = get_basic2agent_acyclic()
        self.assertTrue(example.is_full_rg_strategically_acyclic())

        example2 = get_basic2agent_cyclic()
        self.assertFalse(example2.is_full_rg_strategically_acyclic())

    # @unittest.skip("")
    def test_get_valid_acyclic_dec_node_ordering(self):
        example = get_basic2agent_acyclic()
        self.assertEqual(example.get_valid_acyclic_dec_node_ordering(), ['D1', 'D2'])

        example2 = get_basic2agent_cyclic()
        with self.assertRaises(Exception):
            example2.get_valid_acyclic_dec_node_ordering()

    # @unittest.skip("")
    def test_mechanism_graph(self):
        example = get_basic2agent_acyclic()
        mg = example.mechanism_graph()
        self.assertCountEqual(mg.all_decision_nodes, ['D1', 'D2'])
        self.assertCountEqual(mg.all_utility_nodes, ['U1', 'U2'])
        self.assertEqual(len(mg.nodes()), len(example.nodes()) * 2)

    # @unittest.skip("")
    def test_copy_without_cpds(self):
        cid = get_3node_cid()
        cid_no_cpds = cid.copy_without_cpds()
        self.assertTrue(len(cid_no_cpds.cpds) == 0)

    # @unittest.skip("")
    def test_sufficient_recall(self):
        example = forgetful_movie_star()
        self.assertFalse(example.sufficient_recall(1))
        self.assertTrue(example.sufficient_recall(2))

        example2 = get_basic2agent_acyclic()
        self.assertTrue(example2.sufficient_recall(1))
        self.assertTrue(example2.sufficient_recall(2))
        with self.assertRaises(Exception):
            self.assertTrue(example2.sufficient_recall(3))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestBASE)
    unittest.TextTestRunner().run(suite)
