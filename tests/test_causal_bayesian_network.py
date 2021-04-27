from __future__ import annotations

import sys

import numpy as np
import pytest
from pgmpy.factors.discrete import TabularCPD  # type: ignore

from pycid import CID, MACID, CausalBayesianNetwork, RandomCPD
from pycid.examples.simple_cbns import get_3node_cbn, get_3node_uniform_cbn
from pycid.examples.simple_cids import get_3node_cid, get_minimal_cid
from pycid.examples.story_macids import taxi_competition


@pytest.fixture
def cid_3node() -> CID:
    return get_3node_cid()


@pytest.fixture
def cbn_3node() -> CausalBayesianNetwork:
    return get_3node_cbn()


@pytest.fixture
def cbn_3node_uniform() -> CausalBayesianNetwork:
    return get_3node_uniform_cbn()


@pytest.fixture
def cid_minimal() -> CID:
    return get_minimal_cid()


@pytest.fixture
def macid_taxi_comp() -> MACID:
    return taxi_competition()


class TestRemoveAddEdge:
    @staticmethod
    def test_remove_add_edge(cbn_3node_uniform: CausalBayesianNetwork) -> None:
        cbn = cbn_3node_uniform
        cbn.remove_edge("A", "B")
        assert cbn.check_model()
        cbn.add_edge("A", "B")
        assert cbn.check_model()
        with pytest.raises(ValueError):
            cbn.remove_edge("A", "C")  # the CPD for C relies on knowing the value of A
            assert cbn.check_model()


class TestRemoveNode:
    @staticmethod
    def remove_node(cbn_3node: CausalBayesianNetwork) -> None:
        cbn_3node.remove_node("S")
        cbn_3node.remove_cpds("D")
        cbn_3node.remove_cpds("U")
        assert cbn_3node.nodes == []


class TestAssignCpd:
    @staticmethod
    def test_add_cpds(cbn_3node: CausalBayesianNetwork) -> None:
        cbn = cbn_3node
        cbn.add_cpds(TabularCPD("D", 2, np.eye(2), evidence=["S"], evidence_card=[2]))
        assert cbn.check_model()
        cpd = cbn.get_cpds("D").values
        assert np.array_equal(cpd, np.array([[1, 0], [0, 1]]))

    @staticmethod
    def test_remove_cpds(cbn_3node: CausalBayesianNetwork) -> None:
        cbn_3node.remove_cpds("S")
        assert "S" not in cbn_3node.model
        assert cbn_3node.get_cpds("S") is None
        cbn_3node.remove_cpds("D")
        cbn_3node.remove_cpds("U")


class TestQuery:
    @staticmethod
    def test_query(cbn_3node: CausalBayesianNetwork) -> None:
        assert cbn_3node.query(["U"], {"D": 2}).values[2] == float(1.0)

    @staticmethod
    def test_query_disconnected_components() -> None:
        cbn = CausalBayesianNetwork([("A", "B")])
        cbn.add_cpds(A=RandomCPD(), B=RandomCPD())
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
        # TODO: Ask James about this
        assert macid.expected_value(["U1"], {}, intervention={"D1": "c", "D2": "e"})[0] == 5
        assert macid.expected_value(["U2"], {}, intervention={"D1": "c", "D2": "e"})[0] == 3


class TestCopyWithoutCpds:
    @staticmethod
    def test_copy_without_cpds(cbn_3node: CausalBayesianNetwork) -> None:
        assert len(cbn_3node.copy_without_cpds().cpds) == 0


if __name__ == "__main__":
    pytest.main(sys.argv)
