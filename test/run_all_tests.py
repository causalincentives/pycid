import unittest

from test.test_analyze import TestAnalyze
from test.test_cpd import TestCPD
from test.test_examples import TestExamples
from test.test_notebooks import TestNotebooks
from test.test_cid import TestCID
from test.test_macid import TestMACID
from test.test_get_paths import TestPATHS
from test.test_reasoning_patterns import TestReasoning
from test.test_macid_base import TestBASE


if __name__ == '__main__':
    # All tests can also be run with python3 -m unittest
    suite_list = [unittest.defaultTestLoader.loadTestsFromTestCase(TestCID),
                  unittest.defaultTestLoader.loadTestsFromTestCase(TestCPD),
                  unittest.defaultTestLoader.loadTestsFromTestCase(TestExamples),
                  unittest.defaultTestLoader.loadTestsFromTestCase(TestAnalyze),
                  unittest.defaultTestLoader.loadTestsFromTestCase(TestNotebooks),
                  unittest.defaultTestLoader.loadTestsFromTestCase(TestMACID),
                  unittest.defaultTestLoader.loadTestsFromTestCase(TestPATHS),
                  unittest.defaultTestLoader.loadTestsFromTestCase(TestReasoning),
                  unittest.defaultTestLoader.loadTestsFromTestCase(TestBASE)]
    combo_suite = unittest.TestSuite(suite_list)
    unittest.TextTestRunner(verbosity=0).run(combo_suite)
