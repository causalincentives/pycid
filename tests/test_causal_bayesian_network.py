from __future__ import annotations

import sys

import numpy as np
import pytest
from pgmpy.factors.discrete import TabularCPD  # type: ignore

from pycid import CausalBayesianNetwork, RandomCPD
from pycid.examples.simple_cbns import get_3node_cbn, get_3node_uniform_cbn, get_fork_cbn, get_minimal_cbn


@pytest.fixture
def cbn_3node() -> CausalBayesianNetwork:
    return get_3node_cbn()


@pytest.fixture
def cbn_3node_uniform() -> CausalBayesianNetwork:
    return get_3node_uniform_cbn()


@pytest.fixture
def cbn_minimal() -> CausalBayesianNetwork:
    return get_minimal_cbn()


@pytest.fixture
def cbn_fork() -> CausalBayesianNetwork:
    return get_fork_cbn()


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


class TestIsStructuralCausalModel:
    @staticmethod
    def test_is_scm(cbn_3node: CausalBayesianNetwork) -> None:
        assert cbn_3node.is_structural_causal_model()

    @staticmethod
    def test_is_not_scm(cbn_3node_uniform: CausalBayesianNetwork) -> None:
        assert not cbn_3node_uniform.is_structural_causal_model()


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
    def test_cbn_single_intervention(cbn_minimal: CausalBayesianNetwork) -> None:
        cbn = cbn_minimal
        assert cbn.expected_value(["B"], {})[0] == 0.5
        for a in [0, 1]:
            cbn.intervene({"A": a})
            assert cbn.expected_value(["B"], {})[0] == a
        assert cbn.expected_value(["B"], {}, intervention={"A": 1})[0] == 1

    @staticmethod
    def test_cbn_double_intervention(cbn_fork: CausalBayesianNetwork) -> None:
        cbn = cbn_fork
        assert cbn.expected_value(["C"], {}, intervention={"A": 1, "B": 3})[0] == 3
        assert cbn.expected_value(["C"], {}, intervention={"A": 2, "B": 4})[0] == 8


class TestCopyWithoutCpds:
    @staticmethod
    def test_copy_without_cpds(cbn_3node: CausalBayesianNetwork) -> None:
        assert len(cbn_3node.copy_without_cpds().cpds) == 0


if __name__ == "__main__":
    pytest.main(sys.argv)
