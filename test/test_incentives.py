import sys, os
sys.path.insert(0, os.path.abspath('.'))
import unittest
import numpy as np
from examples import get_3node_cid, get_5node_cid, get_5node_cid_with_scaled_utility, get_2dec_cid, get_nested_cid, \
    get_introduced_bias


class TestIncentives(unittest.TestCase):

    def testTotalEffect(self):
        pass
        # TODO


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestIncentives)
    unittest.TextTestRunner().run(suite)