# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.
# %%
import logging
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
from core.relevance_graph import CondensedRelevanceGraph, RelevanceGraph
import unittest
import numpy as np
from examples.simple_cids import get_3node_cid, get_5node_cid, get_minimal_cid
from examples.story_macids import prisoners_dilemma, taxi_competition
from pgmpy.factors.discrete import TabularCPD  # type: ignore
from examples.story_macids import forgetful_movie_star, subgame_difference
from core.macid_base import MechanismGraph


class TestBASE(unittest.TestCase):

    def setUp(self) -> None:
        logging.disable()

    # @unittest.skip("")
    def test_assign_cpd(self) -> None:
        three_node = get_3node_cid()
        three_node.add_cpds(TabularCPD('D', 2, np.eye(2), evidence=['S'], evidence_card=[2]))
        three_node.check_model()
        cpd = three_node.get_cpds('D').values
        self.assertTrue(np.array_equal(cpd, np.array([[1, 0], [0, 1]])))

    # @unittest.skip("")
    def test_query(self) -> None:
        three_node = get_3node_cid()
        with self.assertRaises(Exception):
            three_node.query(['U'], {})
        with self.assertRaises(Exception):
            three_node.query(['U'], {'D': 0})
        three_node.impute_random_policy()
        with self.assertRaises(Exception):
            three_node.query(['U'], {'S': 0})

    # @unittest.skip("")
    def test_expected_utility(self) -> None:
        three_node = get_3node_cid()
        five_node = get_5node_cid()
        eu00 = three_node.expected_utility({'D': -1, 'S': -1})
        self.assertEqual(eu00, 1)
        eu10 = three_node.expected_utility({'D': 1, 'S': -1})
        self.assertEqual(eu10, -1)
        eu000 = five_node.expected_utility({'D': 0, 'S1': 0, 'S2': 0})
        self.assertEqual(eu000, 2)
        eu001 = five_node.expected_utility({'D': 0, 'S1': 0, 'S2': 1})
        self.assertEqual(eu001, 1)
        macid_example = prisoners_dilemma()
        eu_agent0 = macid_example.expected_utility({'D1': 'd', 'D2': 'c'}, agent=1)
        self.assertEqual(eu_agent0, 0)
        eu_agent1 = macid_example.expected_utility({'D1': 'd', 'D2': 'c'}, agent=2)
        self.assertEqual(eu_agent1, -3)

    # @unittest.skip("")
    def test_get_valid_order(self) -> None:
        macid = prisoners_dilemma()
        self.assertEqual(macid.get_valid_order(), ['D2', 'D1'])
        rg = RelevanceGraph(macid)
        with self.assertRaises(Exception):
            rg.get_valid_order()
        with self.assertRaises(Exception):
            macid.get_valid_order(['D3'])

    # @unittest.skip("")
    def test_intervention(self) -> None:
        cid = get_minimal_cid()
        cid.impute_random_policy()
        self.assertEqual(cid.expected_value(['B'], {})[0], 0.5)
        for a in [0, 1, 2]:
            cid.intervene({'A': a})
            self.assertEqual(cid.expected_value(['B'], {})[0], a)
        self.assertEqual(cid.expected_value(['B'], {}, intervene={'A': 1})[0], 1)
        macid = taxi_competition()
        macid.impute_fully_mixed_policy_profile()
        self.assertEqual(macid.expected_value(['U1'], {}, intervene={'D1': 'c', 'D2': 'e'})[0], 3)
        self.assertEqual(macid.expected_value(['U2'], {}, intervene={'D1': 'c', 'D2': 'e'})[0], 5)

    # @unittest.skip("")
    def test_possible_pure_decision_rules(self) -> None:
        cid = get_minimal_cid()
        possible_pure_decision_rules = cid.possible_pure_decision_rules('A')
        self.assertEqual(len(possible_pure_decision_rules), 2)
        expected_utilities = []
        for decision_rule in possible_pure_decision_rules:
            cid.add_cpds(decision_rule)
            cid.check_model()
            expected_utilities.append(cid.expected_utility({}))
        self.assertEqual(set(expected_utilities), {0, 1})

        cid = get_3node_cid()
        possible_pure_decision_rules = cid.possible_pure_decision_rules('D')
        self.assertEqual(len(possible_pure_decision_rules), 4)
        expected_utilities = []
        matrices = set()
        for decision_rule in possible_pure_decision_rules:
            cid.add_cpds(decision_rule)
            matrices.add(tuple(cid.get_cpds('D').values.flatten()))
            cid.check_model()
            expected_utilities.append(cid.expected_utility({}))
        self.assertEqual(set(expected_utilities), {-1, 0, 1})
        self.assertEqual(len(matrices), 4)

        five_node = get_5node_cid()
        possible_pure_decision_rules = five_node.possible_pure_decision_rules('D')
        self.assertEqual(len(possible_pure_decision_rules), 16)
        expected_utilities = []
        for decision_rule in possible_pure_decision_rules:
            five_node.add_cpds(decision_rule)
            five_node.check_model()
            expected_utilities.append(five_node.expected_utility({}))
        self.assertEqual(set(expected_utilities), {0.5, 1.0, 1.5})

    # @unittest.skip("")
    def test_optimal_decision_rules(self) -> None:
        cid = get_minimal_cid()
        optimal_decision_rules = cid.optimal_decision_rules('A')
        self.assertEqual(len(optimal_decision_rules), 1)
        for cpd in optimal_decision_rules:
            cid.add_cpds(cpd)
            self.assertEqual(cid.expected_utility({}), 1)

        cid = get_3node_cid()
        optimal_decision_rules = cid.optimal_decision_rules('D')
        self.assertEqual(len(optimal_decision_rules), 1)
        for cpd in optimal_decision_rules:
            cid.add_cpds(cpd)
            self.assertEqual(cid.expected_utility({}), 1)

        five_node = get_5node_cid()
        optimal_decision_rules = five_node.optimal_decision_rules('D')
        self.assertEqual(len(optimal_decision_rules), 4)
        five_node.impute_optimal_policy()
        for cpd in optimal_decision_rules:
            five_node.add_cpds(cpd)
            self.assertEqual(five_node.expected_utility({}), 1.5)

    # @unittest.skip("")
    def test_is_s_reachable(self) -> None:
        example = taxi_competition()
        self.assertTrue(example.is_s_reachable('D1', 'D2'))
        self.assertFalse(example.is_s_reachable('D2', 'D1'))

        example2 = subgame_difference()
        self.assertTrue(example2.is_s_reachable('D1', 'D2'))
        self.assertFalse(example2.is_s_reachable('D2', 'D1'))

    # @unittest.skip("")
    def test_is_r_reachable(self) -> None:
        example = subgame_difference()
        self.assertFalse(example.is_r_reachable('D2', 'D1'))
        self.assertFalse(example.is_r_reachable('D2', 'N'))
        self.assertFalse(example.is_r_reachable('D1', 'N'))
        self.assertTrue(example.is_r_reachable('D1', 'D2'))

    # @unittest.skip("")
    def test_relevance_graph(self) -> None:
        example = taxi_competition()
        rg = RelevanceGraph(example)
        self.assertTrue(rg.is_acyclic())
        example2 = prisoners_dilemma()
        rg2 = RelevanceGraph(example2)
        self.assertFalse(rg2.is_acyclic())
        self.assertTrue(len(rg.get_sccs()) == 2)
        self.assertEqual(rg2.get_sccs(), [{'D1', 'D2'}])

    # @unittest.skip("")
    def test_condensed_relevance_graph(self) -> None:
        example = taxi_competition()
        crg = CondensedRelevanceGraph(example)
        self.assertEqual(crg.get_scc_topological_ordering(), [['D1'], ['D2']])
        self.assertEqual(crg.get_decisions_in_scc()[0], ['D2'])

    # @unittest.skip("")
    def test_mechanism_graph(self) -> None:
        example = taxi_competition()
        mg = MechanismGraph(example)
        self.assertCountEqual(mg.all_decision_nodes, ['D1', 'D2'])
        self.assertCountEqual(mg.all_utility_nodes, ['U1', 'U2'])
        self.assertEqual(len(mg.nodes()), len(example.nodes()) * 2)

    # @unittest.skip("")
    def test_copy_without_cpds(self) -> None:
        cid = get_3node_cid()
        cid_no_cpds = cid.copy_without_cpds()
        self.assertTrue(len(cid_no_cpds.cpds) == 0)

    # @unittest.skip("")
    def test_sufficient_recall(self) -> None:
        example = forgetful_movie_star()
        self.assertFalse(example.sufficient_recall(1))
        self.assertTrue(example.sufficient_recall(2))

        example2 = taxi_competition()
        self.assertTrue(example2.sufficient_recall(1))
        self.assertTrue(example2.sufficient_recall(2))
        with self.assertRaises(Exception):
            self.assertTrue(example2.sufficient_recall(3))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestBASE)
    unittest.TextTestRunner().run(suite)

# %%
