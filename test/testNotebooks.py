import unittest
from testbook import testbook
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestNotebooks(unittest.TestCase):

    def test_solve_cpd(self):
        testbook(os.path.join(ROOT_DIR, 'examples/solve_cpd.ipynb'), execute=True)

    def test_generate_cpd(self):
        testbook(os.path.join(ROOT_DIR, 'examples/generate_cid.ipynb'), execute=True)

    def test_MACID_codebase_demonstration(self):
        testbook(os.path.join(ROOT_DIR, 'examples/MACID_codebase_demonstration.ipynb'), execute=True)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestNotebooks)
    unittest.TextTestRunner().run(suite)