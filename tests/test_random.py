import sys
import unittest

import pytest

from pycid.random.random_cid import random_cid, random_cids


class TestRandom(unittest.TestCase):
    def test_random_cid(self) -> None:
        random_cid(4, 1, 1)
        random_cid(8, 2, 2)
        random_cid(12, 3, 3)
        random_cids(n_cids=1)[0]


if __name__ == "__main__":
    pytest.main(sys.argv)
