#Licensed to the Apache Software Foundation (ASF) under one or more contributor license
#agreements; and to You under the Apache License, Version 2.0.

import unittest

import numpy as np

from models.five_node import FiveNode
from models.three_node import ThreeNode
from models.two_decisions import TwoDecisions


class TestCIDClass(unittest.TestCase):

    def test_assign_cpd(self):
        three_node = ThreeNode()
        three_node.assign_cpd('D')
        cpd = three_node.get_cpds('D').values
        three_node.check_model()
        self.assertTrue(np.array_equal(cpd, np.array([[0.5, 0.5], [0.5, 0.5]])))
        three_node.assign_cpd('D', policy=lambda x : [0])
        three_node.check_model()
        cpd = three_node.get_cpds('D').values
        self.assertTrue(np.array_equal(cpd, np.array([[1, 1], [0, 0]])))

    def test_expected_utility(self):
        three_node = ThreeNode()
        five_node = FiveNode()
        eu00 = three_node.expected_utility({'D': 0, 'S': 0})
        self.assertEqual(eu00, 1)
        eu10 = three_node.expected_utility({'D': 1, 'S': 0})
        self.assertEqual(eu10, 0)
        eu000 = five_node.expected_utility({'D': 0, 'S1': 0, 'S2': 0})
        self.assertEqual(eu000, 2)
        eu001 = five_node.expected_utility({'D': 0, 'S1': 0, 'S2': 1})
        self.assertEqual(eu001, 1)

    def test_optimal_decision(self):
        five_node = FiveNode()
        opt00 = five_node.optimal_decisions('D', {'S1': 0, 'S2': 0})
        self.assertEqual(opt00, [0])
        opt01 = five_node.optimal_decisions('D', {'S1': 0, 'S2': 1})
        self.assertEqual(opt01, [0, 1])

    def test_possible_decision_contexts(self):
        three_node = ThreeNode()
        five_node = FiveNode()
        pdc3 = three_node.possible_contexts('D')
        self.assertEqual(pdc3, [{'S': 0}, {'S': 1}])
        pdc5 = five_node.possible_contexts('D')
        self.assertEqual(pdc5, [{'S1': 0, 'S2': 0}, {'S1': 1, 'S2': 0}, {'S1': 0, 'S2': 1}, {'S1': 1, 'S2': 1}])

    def test_sufficient_recall(self):
        two_decisions = TwoDecisions()
        self.assertEqual(two_decisions.check_sufficient_recall(), True)
        two_decisions.remove_edge('S2', 'D2')
        self.assertEqual(two_decisions.check_sufficient_recall(), False)

    def test_solve(self):
        three_node = ThreeNode()
        two_decisions = TwoDecisions()
        three_node.solve()
        cpd = three_node.get_cpds('D').values
        self.assertTrue(np.array_equal(cpd, np.array([[1, 0], [0, 1]])))
        self.assertEqual(three_node.expected_utility(), 1)
        two_decisions.solve()
        two_decisions.solve()  # check that it can be solved repeatedly
        cpd = two_decisions.get_cpds('D2').values
        self.assertTrue(np.array_equal(cpd, np.array([[1, 0], [0, 1]])))
        self.assertEqual(two_decisions.expected_utility(), 1)


if __name__ == '__main__':
    unittest.main()