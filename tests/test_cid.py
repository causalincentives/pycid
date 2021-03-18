import sys
import unittest

import numpy as np
import pytest

from pycid.examples.simple_cids import (
    get_2dec_cid,
    get_3node_cid,
    get_5node_cid_with_scaled_utility,
    get_insufficient_recall_cid,
    get_sequential_cid,
)
from pycid.examples.story_cids import get_introduced_bias


class TestCID(unittest.TestCase):
    # @unittest.skip("")
    def test_sufficient_recall(self) -> None:
        two_decisions = get_2dec_cid()
        self.assertTrue(two_decisions.sufficient_recall())
        sequential = get_sequential_cid()
        self.assertTrue(sequential.sufficient_recall())
        two_decisions.remove_edge("S2", "D2")
        self.assertFalse(two_decisions.sufficient_recall())

    # @unittest.skip("")
    def test_solve(self) -> None:
        three_node = get_3node_cid()
        three_node.solve()
        solution = three_node.solve()  # check that it can be solved repeatedly
        cpd2 = solution["D"]
        self.assertTrue(np.array_equal(cpd2.values, np.array([[1, 0], [0, 1]])))
        three_node.add_cpds(cpd2)
        self.assertEqual(three_node.expected_utility({}), 1)

        two_decisions = get_2dec_cid()
        solution = two_decisions.solve()
        cpd = solution["D2"]
        self.assertTrue(np.array_equal(cpd.values, np.array([[1, 0], [0, 1]])))
        two_decisions.add_cpds(*list(solution.values()))
        self.assertEqual(two_decisions.expected_utility({}), 1)

        sequential = get_sequential_cid()
        sequential.solve()

        cid = get_insufficient_recall_cid()
        cid.impute_optimal_policy()
        self.assertEqual(cid.expected_utility({}), 1)

    # @unittest.skip("")
    def test_scaled_utility(self) -> None:
        cid = get_5node_cid_with_scaled_utility()
        cid.impute_random_policy()
        self.assertEqual(cid.expected_utility({}), 6.0)

    # @unittest.skip("")
    def test_impute_cond_expectation_decision(self) -> None:
        cid = get_introduced_bias()
        cid.impute_conditional_expectation_decision("D", "Y")
        eu_ce = cid.expected_utility({})
        self.assertAlmostEqual(eu_ce, -0.1666, 2)
        cid.impute_optimal_policy()
        eu_opt = cid.expected_utility({})
        self.assertEqual(eu_ce, eu_opt)


if __name__ == "__main__":
    pytest.main(sys.argv)
