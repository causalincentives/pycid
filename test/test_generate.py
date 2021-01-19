import sys, os
sys.path.insert(0, os.path.abspath('.'))

from generate.random_cid import random_cid, random_cids
import unittest


# @unittest.skip("until Ryan/James fix")
class TestGenerate(unittest.TestCase):

    def test_random_cid(self):
        random_cid(4, 1, 1)
        random_cid(8, 2, 2)
        random_cid(12, 3, 3)
        random_cids(n_cids=1)[0]


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestGenerate)
    unittest.TextTestRunner().run(suite)