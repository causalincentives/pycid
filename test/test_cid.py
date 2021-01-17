# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; and to You under the Apache License, Version 2.0.

import sys, os

sys.path.insert(0, os.path.abspath('.'))
import unittest
import numpy as np
from examples import get_3node_cid, get_5node_cid, get_5node_cid_with_scaled_utility, get_2dec_cid, get_nested_cid, \
    get_introduced_bias
from pgmpy.factors.discrete import TabularCPD


class TestCID(unittest.TestCase):

    def test_assign_cpd(self):
        three_node = get_3node_cid()
        three_node.add_cpds(TabularCPD('D', 2, np.eye(2), evidence=['S'], evidence_card=[2]))
        three_node.check_model()
        cpd = three_node.get_cpds('D').values
        self.assertTrue(np.array_equal(cpd, np.array([[1, 0], [0, 1]])))

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

    def test_sufficient_recall(self):
        two_decisions = get_2dec_cid()
        self.assertEqual(two_decisions.check_sufficient_recall(), True)
        two_decisions.remove_edge('S2', 'D2')
        self.assertEqual(two_decisions.check_sufficient_recall(), False)

    def test_solve(self):
        three_node = get_3node_cid()
        soln2 = three_node.solve()
        soln2 = three_node.solve()  # check that it can be solved repeatedly
        cpd2 = soln2['D']
        self.assertTrue(np.array_equal(cpd2.values, np.array([[1, 0], [0, 1]])))
        three_node.add_cpds(cpd2)
        self.assertEqual(three_node.expected_utility({}), 1)

        two_decisions = get_2dec_cid()
        soln = two_decisions.solve()
        cpd = soln['D2']
        self.assertTrue(np.array_equal(cpd.values, np.array([[1, 0], [0, 1]])))
        two_decisions.add_cpds(cpd)
        self.assertEqual(two_decisions.expected_utility({}), 1)

    def test_scaled_utility(self):
        cid = get_5node_cid_with_scaled_utility()
        self.assertEqual(cid.expected_utility({}), 6.0)

    def test_impute_cond_expectation_decision(self):
        cid = get_introduced_bias()
        cid.impute_conditional_expectation_decision('D', 'Y')
        eu_ce = cid.expected_utility({})
        self.assertGreater(eu_ce, -0.2)
        cid.impute_optimal_policy()
        eu_opt = cid.expected_utility({})
        self.assertEqual(eu_ce, eu_opt)

    def test_updated_decision_names(self):
        cid = get_introduced_bias()
        self.assertEqual(cid.get_cpds('D').state_names['D'], [0, 1])
        cid.impute_conditional_expectation_decision('D', 'Y')
        self.assertNotEqual(cid.get_cpds('D').state_names['D'], [0, 1])
        cid.impute_random_policy()
        self.assertNotEqual(cid.get_cpds('D').state_names['D'], [0, 1])
        cid.impute_optimal_policy()
        eu = cid.expected_utility({})
        self.assertGreater(eu, -0.2)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCID)
    unittest.TextTestRunner().run(suite)
