import unittest

from test.testNotebooks import TestNotebooks
from test.test_cid import TestCID
from test.test_incentives import TestIncentives
from test.test_parameterize import TestParameterize

if __name__ == '__main__':
    # All tests can also be run with python3 -m unittest
    suiteList = []
    suiteList.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestCID))
    suiteList.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestIncentives))
    suiteList.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestParameterize))
    suiteList.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestNotebooks))
    comboSuite = unittest.TestSuite(suiteList)
    unittest.TextTestRunner(verbosity=0).run(comboSuite)