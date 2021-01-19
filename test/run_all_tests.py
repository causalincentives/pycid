import unittest

from test.test_analyze import TestAnalyze
from test.test_cpd import TestCPD
from test.test_generate import TestGenerate
from test.test_notebooks import TestNotebooks
from test.test_cid import TestCID

if __name__ == '__main__':
    # All tests can also be run with python3 -m unittest
    suiteList = []
    suiteList.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestCID))
    suiteList.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestCPD))
    suiteList.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestGenerate))
    suiteList.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestAnalyze))
    suiteList.append(unittest.defaultTestLoader.loadTestsFromTestCase(TestNotebooks))
    comboSuite = unittest.TestSuite(suiteList)
    unittest.TextTestRunner(verbosity=0).run(comboSuite)