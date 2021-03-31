from __future__ import annotations

import sys

import numpy as np
import pytest
from pgmpy.factors.discrete import TabularCPD  # type: ignore

from pycid import CID, MACID, CausalBayesianNetwork, RandomCPD
from pycid.examples.simple_cbns import get_3node_cbn
from pycid.examples.simple_cids import get_3node_cid, get_minimal_cid
from pycid.examples.story_macids import taxi_competition


@pytest.fixture
def cid_3node() -> CID:
    return get_3node_cid()


@pytest.fixture
def cbn_3node() -> CausalBayesianNetwork:
    return get_3node_cbn()


@pytest.fixture
def cid_minimal() -> CID:
    return get_minimal_cid()


@pytest.fixture
def macid_taxi_comp() -> MACID:
    return taxi_competition()


class TestRemoveAddEdge:
    @staticmethod
    def test_remove_add_edge(cid_3node: CID) -> None:
        cid = cid_3node
        cid.remove_edge("S", "D")
        assert cid.check_model()
        cid.add_edge("S", "D")
        assert cid.check_model()


class TestAssignCpd:
    @staticmethod
    def test_add_cpds(cbn_3node: CausalBayesianNetwork) -> None:
        cbn = cbn_3node
        cbn.add_cpds(TabularCPD("D", 2, np.eye(2), evidence=["S"], evidence_card=[2]))
        assert cbn.check_model()
        cpd = cbn.get_cpds("D").values
        assert np.array_equal(cpd, np.array([[1, 0], [0, 1]]))


class TestQuery:
    @staticmethod
    def test_query(cbn_3node: CausalBayesianNetwork) -> None:
        assert cbn_3node.query(["U"], {"D": 2}).values[2] == float(1.0)

    @staticmethod
    def test_query_disconnected_components() -> None:
        cbn = CausalBayesianNetwork([("A", "B")])
        cbn.add_cpds(RandomCPD("A"), RandomCPD("B"))
        cbn.query(["A"], {}, intervention={"B": 0})  # the intervention separates A and B into separare components

    @staticmethod
    def test_valid_context(cbn_3node: CausalBayesianNetwork) -> None:
        with pytest.raises(ValueError):
            cbn_3node.query(["U"], {"S": 0})


class TestIntervention:
    @staticmethod
    def test_cid_single_intervention(cid_minimal: CID) -> None:
        cid = cid_minimal
        cid.impute_random_policy()
        assert cid.expected_value(["B"], {})[0] == 0.5
        for a in [0, 1]:
            cid.intervene({"A": a})
            assert cid.expected_value(["B"], {})[0] == a
        assert cid.expected_value(["B"], {}, intervention={"A": 1})[0] == 1

    @staticmethod
    def test_macid_double_intervention(macid_taxi_comp: MACID) -> None:
        macid = macid_taxi_comp
        macid.impute_fully_mixed_policy_profile()
        assert macid.expected_value(["U1"], {}, intervention={"D1": "c", "D2": "e"})[0] == 3
        assert macid.expected_value(["U2"], {}, intervention={"D1": "c", "D2": "e"})[0] == 5


class TestCopyWithoutCpds:
    @staticmethod
    def test_copy_without_cpds(cbn_3node: CausalBayesianNetwork) -> None:
        assert len(cbn_3node.copy_without_cpds().cpds) == 0


if __name__ == "__main__":
    pytest.main(sys.argv)
