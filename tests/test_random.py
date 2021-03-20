import sys

import pytest

from pycid.random.random_cid import random_cid, random_cids


@pytest.mark.parametrize("n_all,n_decisions,n_utilities", [(4, 1, 1), (8, 2, 2), (12, 3, 3)])
def test_random_cid_create(n_all: int, n_decisions: int, n_utilities: int) -> None:
    random_cid(n_all, n_decisions, n_utilities).check_model()


def test_random_cids_create_one() -> None:
    for cid in random_cids(n_cids=1):
        cid.check_model()


if __name__ == "__main__":
    pytest.main(sys.argv)
