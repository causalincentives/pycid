import logging
import unittest

import numpy as np

from examples.simple_cids import get_minimal_cid
from examples.story_cids import get_introduced_bias
from pycid.core.cpd import FunctionCPD, UniformRandomCPD


class TestCPD(unittest.TestCase):
    def setUp(self) -> None:
        logging.disable()

    def test_initialize_uniform_random_cpd(self) -> None:
        cid = get_minimal_cid()
        cpd_a = UniformRandomCPD("A", [0, 2])
        cpd_a.initialize_tabular_cpd(cid)
        self.assertTrue((cpd_a.get_values() == np.array([[0.5], [0.5]])).all())
        self.assertEqual(cpd_a.get_state_names("A", 1), 2)
        cpd_b = UniformRandomCPD("B", [0, 1])
        cpd_b.initialize_tabular_cpd(cid)
        self.assertTrue((cpd_a.get_values() == np.array([[0.5, 0.5], [0.5, 0.5]])).all())

    def test_initialize_function_cpd(self) -> None:
        cid = get_minimal_cid()
        cpd_a = FunctionCPD("A", lambda: 2, evidence=[])
        cpd_a.initialize_tabular_cpd(cid)
        self.assertTrue(cpd_a.get_values(), np.array([[1]]))
        self.assertEqual(cpd_a.get_cardinality(["A"])["A"], 1)
        self.assertEqual(cpd_a.get_state_names("A", 0), 2)
        cpd_b = FunctionCPD("B", lambda x: x, evidence=["A"])
        cpd_b.initialize_tabular_cpd(cid)
        self.assertTrue(cpd_a.get_values(), np.array([[1]]))
        self.assertEqual(cpd_a.get_cardinality(["A"])["A"], 1)
        self.assertEqual(cpd_a.get_state_names("A", 0), 2)

    def test_updated_decision_names(self) -> None:
        cid = get_introduced_bias()
        self.assertEqual(cid.get_cpds("D").state_names["D"], [0, 1])
        cid.impute_conditional_expectation_decision("D", "Y")
        self.assertNotEqual(cid.get_cpds("D").state_names["D"], [0, 1])
        cid.impute_random_policy()
        self.assertNotEqual(cid.get_cpds("D").state_names["D"], [0, 1])
        # TODO: It doesn't always work to impute an optimal policy after imputing a
        #       conditional expectation one, possibly because the real-valued decision domain?
        # cid.impute_optimal_policy()
        # eu = cid.expected_utility({})
        # self.assertGreater(eu, -0.2)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCPD)
    unittest.TextTestRunner().run(suite)
