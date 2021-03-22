from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pycid.examples.simple_cids import (
    get_2dec_cid,
    get_3node_cid,
    get_5node_cid_with_scaled_utility,
    get_insufficient_recall_cid,
    get_sequential_cid,
)
from pycid.examples.story_cids import get_introduced_bias

if TYPE_CHECKING:
    from pycid import CID


@pytest.fixture
def cid_2dec() -> CID:
    return get_2dec_cid()


@pytest.fixture
def cid_3node() -> CID:
    return get_3node_cid()


@pytest.fixture
def cid_sequential() -> CID:
    return get_sequential_cid()


@pytest.fixture
def cid_insufficient_recall() -> CID:
    return get_insufficient_recall_cid()


@pytest.fixture
def cid_5node_scaled_utility() -> CID:
    return get_5node_cid_with_scaled_utility()


@pytest.fixture
def cid_introduced_bias() -> CID:
    return get_introduced_bias()


class TestSufficientRecall:
    @staticmethod
    def test_has_sufficient_recall(cid_2dec: CID) -> None:
        assert cid_2dec.sufficient_recall()

    @staticmethod
    def test_has_no_sufficient_recall(cid_2dec: CID) -> None:
        cid_2dec.remove_edge("S2", "D2")
        assert not cid_2dec.sufficient_recall()


class TestSolve:
    @staticmethod
    def test_solve_3node_cid(cid_3node: CID) -> None:
        cid = cid_3node
        cid.solve()
        solution = cid.solve()  # check that it can be solved repeatedly
        cpd = solution["D"]
        assert np.array_equal(cpd.values, np.array([[1, 0], [0, 1]]))
        cid.add_cpds(cpd)
        assert cid.expected_utility({}) == 1

    @staticmethod
    def test_solve_2dec_cid(cid_2dec: CID) -> None:
        solution = cid_2dec.solve()
        cpd = solution["D2"]
        assert np.array_equal(cpd.values, np.array([[1, 0], [0, 1]]))
        cid_2dec.add_cpds(*list(solution.values()))
        assert cid_2dec.expected_utility({}) == 1

    @staticmethod
    def test_impute_optimal_policy(cid_insufficient_recall: CID) -> None:
        cid_insufficient_recall.impute_optimal_policy()
        assert cid_insufficient_recall.expected_utility({}) == 1

    @staticmethod
    def test_scaled_utility(cid_5node_scaled_utility: CID) -> None:
        cid_5node_scaled_utility.impute_random_policy()
        assert cid_5node_scaled_utility.expected_utility({}) == 6.0


class TestConditionalExpectationDecision:
    @staticmethod
    def test_impute_cond_expectation_decision(cid_introduced_bias: CID) -> None:
        cid_introduced_bias.impute_conditional_expectation_decision("D", "Y")
        eu_ce = cid_introduced_bias.expected_utility({})
        assert eu_ce == pytest.approx(-0.1666, abs=1e-2)


if __name__ == "__main__":
    pytest.main(sys.argv)
