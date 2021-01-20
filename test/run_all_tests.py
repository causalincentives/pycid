import unittest

from test.test_analyze import TestAnalyze
from test.test_cpd import TestCPD
from test.test_generate import TestGenerate
from test.test_notebooks import TestNotebooks
from test.test_cid import TestCID

if __name__ == '__main__':
    # All tests can also be run with python3 -m unittest
    suiteList = [unittest.defaultTestLoader.loadTestsFromTestCase(TestCID),
                 unittest.defaultTestLoader.loadTestsFromTestCase(TestCPD),
                 unittest.defaultTestLoader.loadTestsFromTestCase(TestGenerate),
                 unittest.defaultTestLoader.loadTestsFromTestCase(TestAnalyze),
                 unittest.defaultTestLoader.loadTestsFromTestCase(TestNotebooks)]
    comboSuite = unittest.TestSuite(suiteList)
    unittest.TextTestRunner(verbosity=0).run(comboSuite)
