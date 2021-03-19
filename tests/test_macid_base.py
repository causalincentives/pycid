import sys
import unittest

import numpy as np
import pytest
from pgmpy.factors.discrete import TabularCPD  # type: ignore

from pycid.core.macid_base import MechanismGraph
from pycid.core.relevance_graph import CondensedRelevanceGraph, RelevanceGraph
from pycid.examples.simple_cids import get_3node_cid, get_5node_cid, get_minimal_cid
from pycid.examples.story_macids import forgetful_movie_star, prisoners_dilemma, subgame_difference, taxi_competition


class TestBASE(unittest.TestCase):
    # @unittest.skip("")
    def test_remove_add_edge(self) -> None:
        cid = get_3node_cid()
        cid.remove_edge("S", "D")
        self.assertTrue(cid.check_model())
        cid.add_edge("S", "D")
        self.assertTrue(cid.check_model())

    def test_make_decision(self) -> None:
        cid = get_3node_cid()
        self.assertCountEqual(cid.decisions, ["D"])
        cid.make_decision("S")
        self.assertCountEqual(cid.decisions, ["D", "S"])
        self.assertEqual(cid.decision_agent["S"], 0)
        self.assertCountEqual(cid.agent_decisions[0], ["D", "S"])
        cid2 = cid.copy_without_cpds()
        with self.assertRaises(ValueError):
            cid2.make_decision("S")

    def test_make_chance(self) -> None:
        cid = get_3node_cid()
        self.assertCountEqual(cid.decisions, ["D"])
        cid.make_decision("S")
        self.assertCountEqual(cid.decisions, ["D", "S"])
        cid.make_chance("S")
        self.assertCountEqual(cid.decisions, ["D"])

    # @unittest.skip("")
    def test_assign_cpd(self) -> None:
        three_node = get_3node_cid()
        three_node.add_cpds(TabularCPD("D", 2, np.eye(2), evidence=["S"], evidence_card=[2]))
        three_node.check_model()
        cpd = three_node.get_cpds("D").values
        self.assertTrue(np.array_equal(cpd, np.array([[1, 0], [0, 1]])))

    # @unittest.skip("")
    def test_query(self) -> None:
        three_node = get_3node_cid()
        three_node.query(["U"], {"D": -1}, intervention={"S": 1})
        # The following queries should not be allowed before a policy is specified
        with self.assertRaises(ValueError):
            three_node.query(["U"], {})
        with self.assertRaises(ValueError):
            three_node.query(["U"], {"D": 1})
        with self.assertRaises(ValueError):
            three_node.query(["U"], {"S": 1})
        # but should be allowed after
        three_node.impute_random_policy()
        three_node.query(["U"], {})
        three_node.query(["U"], {"D": 1})
        three_node.query(["U"], {"S": 1})
        # contexts still need be within the domain of the variable
        with self.assertRaises(ValueError):
            three_node.query(["U"], {"S": 0})

    # @unittest.skip("")
    def test_expected_utility(self) -> None:
        three_node = get_3node_cid()
        five_node = get_5node_cid()
        eu00 = three_node.expected_utility({"D": -1, "S": -1})
        self.assertEqual(eu00, 1)
        eu10 = three_node.expected_utility({"D": 1, "S": -1})
        self.assertEqual(eu10, -1)
        eu000 = five_node.expected_utility({"D": 0, "S1": 0, "S2": 0})
        self.assertEqual(eu000, 2)
        eu001 = five_node.expected_utility({"D": 0, "S1": 0, "S2": 1})
        self.assertEqual(eu001, 1)
        macid_example = prisoners_dilemma()
        eu_agent0 = macid_example.expected_utility({"D1": "d", "D2": "c"}, agent=1)
        self.assertEqual(eu_agent0, 0)
        eu_agent1 = macid_example.expected_utility({"D1": "d", "D2": "c"}, agent=2)
        self.assertEqual(eu_agent1, -3)

    # @unittest.skip("")
    def test_get_valid_order(self) -> None:
        macid = prisoners_dilemma()
        self.assertEqual(macid.get_valid_order(), ["D2", "D1"])
        rg = RelevanceGraph(macid)
        with self.assertRaises(AttributeError):
            rg.get_valid_order()
            # TODO we're checking that the relevance graph doesn't have a valid order method? why?
        with self.assertRaises(KeyError):
            macid.get_valid_order(["D3"])

    # @unittest.skip("")
    def test_intervention(self) -> None:
        cid = get_minimal_cid()
        cid.impute_random_policy()
        self.assertEqual(cid.expected_value(["B"], {})[0], 0.5)
        for a in [0, 1]:
            cid.intervene({"A": a})
            self.assertEqual(cid.expected_value(["B"], {})[0], a)
        self.assertEqual(cid.expected_value(["B"], {}, intervention={"A": 1})[0], 1)
        macid = taxi_competition()
        macid.impute_fully_mixed_policy_profile()
        self.assertEqual(macid.expected_value(["U1"], {}, intervention={"D1": "c", "D2": "e"})[0], 3)
        self.assertEqual(macid.expected_value(["U2"], {}, intervention={"D1": "c", "D2": "e"})[0], 5)

    # @unittest.skip("")
    def test_possible_pure_decision_rules(self) -> None:
        cid = get_minimal_cid()
        possible_pure_decision_rules = cid.pure_decision_rules("A")
        self.assertEqual(len(possible_pure_decision_rules), 2)
        expected_utilities = []
        for decision_rule in possible_pure_decision_rules:
            cid.add_cpds(decision_rule)
            cid.check_model()
            expected_utilities.append(cid.expected_utility({}))
        self.assertEqual(set(expected_utilities), {0, 1})

        cid = get_3node_cid()
        possible_pure_decision_rules = cid.pure_decision_rules("D")
        self.assertEqual(len(possible_pure_decision_rules), 4)
        expected_utilities = []
        matrices = set()
        for decision_rule in possible_pure_decision_rules:
            cid.add_cpds(decision_rule)
            matrices.add(tuple(cid.get_cpds("D").values.flatten()))
            cid.check_model()
            expected_utilities.append(cid.expected_utility({}))
        self.assertEqual(set(expected_utilities), {-1, 0, 1})
        self.assertEqual(len(matrices), 4)

        five_node = get_5node_cid()
        possible_pure_decision_rules = five_node.pure_decision_rules("D")
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
        optimal_decision_rules = cid.optimal_pure_decision_rules("A")
        self.assertEqual(len(optimal_decision_rules), 1)
        for cpd in optimal_decision_rules:
            cid.add_cpds(cpd)
            self.assertEqual(cid.expected_utility({}), 1)

        cid = get_3node_cid()
        optimal_decision_rules = cid.optimal_pure_decision_rules("D")
        self.assertEqual(len(optimal_decision_rules), 1)
        for cpd in optimal_decision_rules:
            cid.add_cpds(cpd)
            self.assertEqual(cid.expected_utility({}), 1)

        five_node = get_5node_cid()
        optimal_decision_rules = five_node.optimal_pure_decision_rules("D")
        self.assertEqual(len(optimal_decision_rules), 4)
        five_node.impute_optimal_policy()
        for cpd in optimal_decision_rules:
            five_node.add_cpds(cpd)
            self.assertEqual(five_node.expected_utility({}), 1.5)

    # @unittest.skip("")
    def test_is_s_reachable(self) -> None:
        example = taxi_competition()
        self.assertTrue(example.is_s_reachable("D1", "D2"))
        self.assertFalse(example.is_s_reachable("D2", "D1"))

        example2 = subgame_difference()
        self.assertTrue(example2.is_s_reachable("D1", "D2"))
        self.assertFalse(example2.is_s_reachable("D2", "D1"))

    # @unittest.skip("")
    def test_is_r_reachable(self) -> None:
        example = subgame_difference()
        self.assertFalse(example.is_r_reachable("D2", "D1"))
        self.assertFalse(example.is_r_reachable("D2", "N"))
        self.assertFalse(example.is_r_reachable("D1", "N"))
        self.assertTrue(example.is_r_reachable("D1", "D2"))

    # @unittest.skip("")
    def test_relevance_graph(self) -> None:
        example = taxi_competition()
        rg = RelevanceGraph(example)
        self.assertTrue(rg.is_acyclic())
        example2 = prisoners_dilemma()
        rg2 = RelevanceGraph(example2)
        self.assertFalse(rg2.is_acyclic())
        self.assertTrue(len(rg.get_sccs()) == 2)
        self.assertEqual(rg2.get_sccs(), [{"D1", "D2"}])

    # @unittest.skip("")
    def test_condensed_relevance_graph(self) -> None:
        example = taxi_competition()
        crg = CondensedRelevanceGraph(example)
        self.assertEqual(crg.get_scc_topological_ordering(), [["D1"], ["D2"]])
        self.assertEqual(crg.get_decisions_in_scc()[0], ["D2"])

    # @unittest.skip("")
    def test_mechanism_graph(self) -> None:
        example = taxi_competition()
        mg = MechanismGraph(example)
        self.assertCountEqual(mg.decisions, ["D1", "D2"])
        self.assertCountEqual(mg.utilities, ["U1", "U2"])
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
        with self.assertRaises(ValueError):
            self.assertTrue(example2.sufficient_recall(3))


if __name__ == "__main__":
    pytest.main(sys.argv)
