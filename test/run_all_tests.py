import unittest

from test.test_cid import TestCIDClass
from test.test_incentives import TestIncentives
from test.test_parametrize import TestParameterize

if __name__ == '__main__':
    # All tests can also be run with python3 -m unittest
    suiteList = []
    suiteList.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestCIDClass))
    suiteList.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestIncentives))
    suiteList.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestParameterize))
    comboSuite = unittest.TestSuite(suiteList)
    unittest.TextTestRunner(verbosity=0).run(comboSuite)
