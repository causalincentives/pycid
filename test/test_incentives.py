import sys, os
import unittest
sys.path.insert(0, os.path.abspath('.'))
from cpd import FunctionCPD
from incentives import total_effect, introduced_total_effect
from examples import get_introduced_bias, get_minimal_cid


class TestIncentives(unittest.TestCase):

    def testTotalEffect(self):
        cid = get_minimal_cid()
        self.assertEqual(total_effect(cid, 'A', 'B', 0, 1), 1)
        cid = get_introduced_bias()
        self.assertEqual(total_effect(cid, 'A', 'X', 0, 1), 0.5)
        self.assertEqual(total_effect(cid, 'A', 'D', 0, 1), 0)
        self.assertEqual(total_effect(cid, 'A', 'Y', 0, 1), 0.5)

    def testIntroducedEffect(self):
        cid = get_introduced_bias()
        self.assertEqual(introduced_total_effect(cid, 'A', 'D', 'Y', 0, 1), -0.5)
        cid.impute_conditional_expectation_decision('D', 'Y')
        self.assertAlmostEqual(introduced_total_effect(cid, 'A', 'D', 'Y', 0, 1), 0.3333, 2)
        # Try modified model where X doesn't depend on Z
        cid = get_introduced_bias()
        cid.add_cpds(FunctionCPD('X', lambda a, z: a, evidence=['A', 'Z']))
        cid.impute_conditional_expectation_decision('D', 'Y')
        self.assertAlmostEqual(introduced_total_effect(cid, 'A', 'D', 'Y', 0, 1), 0, 2)
        # Try modified model where Y doesn't depend on Z
        cid = get_introduced_bias()
        cid.add_cpds(FunctionCPD('Y', lambda x, z: x, evidence=['X', 'Z']))
        cid.impute_conditional_expectation_decision('D', 'Y')
        self.assertAlmostEqual(introduced_total_effect(cid, 'A', 'D', 'Y', 0, 1), 0, 2)
        # Try modified model where Y doesn't depend on X
        cid = get_introduced_bias()
        cid.add_cpds(FunctionCPD('Y', lambda x, z: z, evidence=['X', 'Z']))
        cid.impute_conditional_expectation_decision('D', 'Y')
        self.assertAlmostEqual(introduced_total_effect(cid, 'A', 'D', 'Y', 0, 1), 0.333, 2)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestIncentives)
    unittest.TextTestRunner().run(suite)
