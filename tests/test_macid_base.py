import sys
import unittest

import pytest

from pycid.core.cpd import DecisionDomain
from pycid.core.macid_base import MechanismGraph
from pycid.core.relevance_graph import CondensedRelevanceGraph, RelevanceGraph
from pycid.examples.simple_cids import get_3node_cid, get_5node_cid, get_minimal_cid
from pycid.examples.story_macids import forgetful_movie_star, prisoners_dilemma, subgame_difference, taxi_competition


class TestBASE(unittest.TestCase):
    # @unittest.skip("")
    def test_make_decision(self) -> None:
        cid = get_3node_cid()
        self.assertCountEqual(cid.decisions, ["D"])
        cid.make_decision("S")
        self.assertCountEqual(cid.decisions, ["D", "S"])
        self.assertEqual(cid.decision_agent["S"], 0)
        self.assertCountEqual(cid.agent_decisions[0], ["D", "S"])

    def test_make_utility(self) -> None:
        cid = get_3node_cid()
        self.assertCountEqual(cid.utilities, ["U"])
        cid.make_utility("S")
        self.assertCountEqual(cid.utilities, ["U", "S"])
        self.assertEqual(cid.utility_agent["S"], 0)
        self.assertCountEqual(cid.agent_utilities[0], ["U", "S"])

    def test_make_chance(self) -> None:
        cid = get_3node_cid()
        self.assertCountEqual(cid.decisions, ["D"])
        cid.make_decision("S")
        self.assertCountEqual(cid.decisions, ["D", "S"])
        cid.make_chance("S")
        self.assertCountEqual(cid.decisions, ["D"])

    # @unittest.skip("")
    def test_query(self) -> None:
        three_node = get_3node_cid()
        three_node.query(["U"], {"D": -1}, intervention={"S": 1})
        self.assertTrue(three_node.query(["U"], {"D": -1}, intervention={"S": 1}).values[0] == float(1.0))
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
        macid = taxi_competition()
        self.assertEqual(macid.get_valid_order(), ["D1", "D2"])
        rg = RelevanceGraph(macid)
        with self.assertRaises(AttributeError):
            rg.get_valid_order()
            # TODO we're checking that the relevance graph doesn't have a valid order method? why?
        with self.assertRaises(KeyError):
            macid.get_valid_order(["D3"])

    # @unittest.skip("")
    def test_possible_pure_decision_rules(self) -> None:
        cid = get_minimal_cid()
        possible_pure_decision_rules = list(cid.pure_decision_rules("A"))
        self.assertEqual(len(possible_pure_decision_rules), 2)
        expected_utilities = []
        for decision_rule in possible_pure_decision_rules:
            cid.add_cpds(decision_rule)
            cid.check_model()
            expected_utilities.append(cid.expected_utility({}))
        self.assertEqual(set(expected_utilities), {0, 1})

        cid = get_3node_cid()
        possible_pure_decision_rules = list(cid.pure_decision_rules("D"))
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
        possible_pure_decision_rules = list(five_node.pure_decision_rules("D"))
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
    def test_copy_without_cpds(self) -> None:
        cid = get_3node_cid()
        cid_no_cpds = cid.copy_without_cpds()
        self.assertTrue(len(cid_no_cpds.cpds) == 0)

    def test_remove_all_decision_rules(self) -> None:
        macid = prisoners_dilemma()
        self.assertTrue(isinstance(macid.get_cpds("D1"), DecisionDomain))
        macid.remove_all_decision_rules()
        self.assertTrue(isinstance(macid.get_cpds("D1"), DecisionDomain))
        macid.impute_fully_mixed_policy_profile()
        self.assertFalse(isinstance(macid.get_cpds("D1"), DecisionDomain))
        macid.remove_all_decision_rules()
        self.assertTrue(isinstance(macid.get_cpds("D1"), DecisionDomain))
        macid_copy = macid.copy_without_cpds()
        with self.assertRaises(KeyError):
            macid_copy.remove_all_decision_rules()

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

    # @unittest.skip("")
    def test_mechanism_graph(self) -> None:
        example = taxi_competition()
        mg = MechanismGraph(example)
        self.assertCountEqual(mg.decisions, ["D1", "D2"])
        self.assertCountEqual(mg.utilities, ["U1", "U2"])
        self.assertEqual(len(mg.nodes()), len(example.nodes()) * 2)


if __name__ == "__main__":
    pytest.main(sys.argv)
