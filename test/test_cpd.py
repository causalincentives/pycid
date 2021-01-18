import numpy as np
import unittest
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from cpd import NullCPD, FunctionCPD
from examples import get_introduced_bias, get_minimal_cid


class TestCPD(unittest.TestCase):

    def test_initialize_null_cpd(self):
        cid = get_minimal_cid()
        cpd_a = NullCPD('A', 2, state_names={'A': [0, 2]})
        cpd_a.initialize_tabular_cpd(cid)
        self.assertTrue((cpd_a.get_values() == np.array([[0.5], [0.5]])).all())
        self.assertEqual(cpd_a.get_state_names('A', 1), 2)
        cpd_b = NullCPD('B', 2)
        cpd_b.initialize_tabular_cpd(cid)
        self.assertTrue((cpd_a.get_values() == np.array([[0.5, 0.5], [0.5, 0.5]])).all())

    def test_initialize_function_cpd(self):
        cid = get_minimal_cid()
        cpd_a = FunctionCPD('A', lambda : 2, evidence=[])
        cpd_a.initialize_tabular_cpd(cid)
        self.assertTrue(cpd_a.get_values(), np.array([[1]]))
        self.assertEqual(cpd_a.get_cardinality(['A'])['A'], 1)
        self.assertEqual(cpd_a.get_state_names('A', 0), 2)
        cpd_b = FunctionCPD('B', lambda x: x, evidence=['A'])
        cpd_b.initialize_tabular_cpd(cid)
        self.assertTrue(cpd_a.get_values(), np.array([[1]]))
        self.assertEqual(cpd_a.get_cardinality(['A'])['A'], 1)
        self.assertEqual(cpd_a.get_state_names('A', 0), 2)

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
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCPD)
    unittest.TextTestRunner().run(suite)
