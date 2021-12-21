from __future__ import annotations

import sys

import pytest

from pycid import CID, CausalBayesianNetwork
from pycid.analyze.effects import introduced_total_effect, total_effect
from pycid.analyze.instrumental_control_incentive import admits_ici, admits_ici_list
from pycid.analyze.requisite_graph import requisite, requisite_graph
from pycid.analyze.response_incentive import admits_ri, admits_ri_list
from pycid.analyze.value_of_control import (
    admits_dir_voc,
    admits_dir_voc_list,
    admits_indir_voc,
    admits_indir_voc_list,
    admits_voc,
    admits_voc_list,
    quantitative_voc,
)
from pycid.analyze.value_of_information import admits_voi, admits_voi_list, quantitative_voi
from pycid.core.cpd import discrete_uniform
from pycid.core.macid import MACID
from pycid.examples.simple_cids import get_3node_cid, get_minimal_cid, get_quantitative_voi_cid, get_trim_example_cid
from pycid.examples.story_cids import (
    get_content_recommender,
    get_fitness_tracker,
    get_grade_predictor,
    get_introduced_bias,
    get_modified_content_recommender,
)


@pytest.fixture
def cid_3node() -> CID:
    return get_3node_cid()


@pytest.fixture
def cid_minimal() -> CID:
    return get_minimal_cid()


@pytest.fixture
def imputed_cid_minimal(cid_minimal: CID) -> CID:
    cid_minimal.impute_random_policy()
    return cid_minimal


@pytest.fixture
def cid_introduced_bias() -> CID:
    return get_introduced_bias()


@pytest.fixture
def cid_quantitative_voi() -> CID:
    return get_quantitative_voi_cid()


@pytest.fixture
def imputed_cid_introduced_bias(cid_introduced_bias: CID) -> CID:
    cid_introduced_bias.impute_random_policy()
    return cid_introduced_bias


@pytest.fixture
def cid_grade_predictor() -> CID:
    return get_grade_predictor()


@pytest.fixture
def cid_content_recommender() -> CID:
    return get_content_recommender()


@pytest.fixture
def cid_modified_content_recommender() -> CID:
    return get_modified_content_recommender()


@pytest.fixture
def cid_trim_example() -> CID:
    return get_trim_example_cid()


@pytest.fixture
def cid_fitness_tracker() -> CID:
    return get_fitness_tracker()


@pytest.fixture
def macid() -> MACID:
    return MACID(
        [("D1", "D2"), ("D1", "U1"), ("D1", "U2"), ("D2", "U2"), ("D2", "U1")],
        agent_decisions={0: {"D": ["D1"], "U": ["U1"]}, 1: {"D": ["D2"], "U": ["U2"]}},
        agent_utilities={0: {"D": ["D1"], "U": ["U1"]}, 1: {"D": ["D2"], "U": ["U2"]}},
    )


# Note: The classes are not necessary.
# They are used to create a common namespace for tests of the same function.


class TestAdmitsVoi:
    @staticmethod
    def test_has_voi_introduced_bias(cid_introduced_bias: CID) -> None:
        assert admits_voi(cid_introduced_bias, "D", "A")

    @staticmethod
    def test_no_voi_grade_predictor(cid_grade_predictor: CID) -> None:
        assert not admits_voi(cid_grade_predictor, "P", "Ge")

    @staticmethod
    def test_invalid_decision(cid_grade_predictor: CID) -> None:
        with pytest.raises(KeyError):
            admits_voi(cid_grade_predictor, "_", "Ge")

    @staticmethod
    def test_invalid_target(cid_grade_predictor: CID) -> None:
        with pytest.raises(KeyError):
            admits_voi(cid_grade_predictor, "P", "_")


class TestAdmitsVoiList:
    @staticmethod
    def test_introduced_bias(cid_introduced_bias: CID) -> None:
        assert set(admits_voi_list(cid_introduced_bias, "D")) == {"A", "X", "Z", "Y"}

    @staticmethod
    def test_grade_predictor(cid_grade_predictor: CID) -> None:
        assert set(admits_voi_list(cid_grade_predictor, "P")) == {"HS", "E", "Gr"}

    @staticmethod
    def test_grade_predictor_removed_edge(cid_grade_predictor: CID) -> None:
        cid_grade_predictor.remove_edge("HS", "P")
        assert set(admits_voi_list(cid_grade_predictor, "P")) == {"R", "HS", "E", "Gr"}


class TestQuantitativeVoi:
    @staticmethod
    def test_quantitative_voi(cid_quantitative_voi: CID) -> None:
        assert set(admits_voi_list(cid_quantitative_voi, "D")) == {"X", "S"}
        assert quantitative_voi(cid_quantitative_voi, "D", "X") == pytest.approx(0.6)
        assert quantitative_voi(cid_quantitative_voi, "D", "S") == pytest.approx(0.4)

    @staticmethod
    def test_invalid_target(cid_quantitative_voi: CID) -> None:
        with pytest.raises(KeyError):
            quantitative_voi(cid_quantitative_voi, "D", "_")

    @staticmethod
    def test_invalid_node(cid_quantitative_voi: CID) -> None:
        with pytest.raises(ValueError):
            quantitative_voi(cid_quantitative_voi, "D", "U")


class TestTotalEffect:
    @staticmethod
    def test_minimal_cid(imputed_cid_minimal: CID) -> None:
        assert total_effect(imputed_cid_minimal, "A", "B", 0, 1) == 1

    @staticmethod
    def test_introduced_bias(imputed_cid_introduced_bias: CID) -> None:
        assert total_effect(imputed_cid_introduced_bias, "A", "X", 0, 1) == pytest.approx(0.5)
        assert total_effect(imputed_cid_introduced_bias, "A", "D", 0, 1) == pytest.approx(0)
        assert total_effect(imputed_cid_introduced_bias, "A", "Y", 0, 1) == pytest.approx(0.5)


class TestIntroducedTotalEffect:
    @staticmethod
    def test_introduced_bias(imputed_cid_introduced_bias: CID) -> None:
        assert introduced_total_effect(imputed_cid_introduced_bias, "A", "D", "Y", 0, 1) == pytest.approx(-0.5)

        imputed_cid_introduced_bias.impute_conditional_expectation_decision("D", "Y")
        assert introduced_total_effect(imputed_cid_introduced_bias, "A", "D", "Y", 0, 1) == pytest.approx(1 / 3)

    @staticmethod
    def test_introduced_bias_x_nodep_z(imputed_cid_introduced_bias: CID) -> None:
        # Modified model where X doesn't depend on Z
        cid = imputed_cid_introduced_bias
        cid.add_cpds(X=lambda A, Z: A)  # type: ignore
        cid.impute_conditional_expectation_decision("D", "Y")
        assert introduced_total_effect(cid, "A", "D", "Y", 0, 1) == pytest.approx(0)

    @staticmethod
    def test_introduced_bias_y_nodep_z(imputed_cid_introduced_bias: CID) -> None:
        # Modified model where Y doesn't depend on Z
        cid = imputed_cid_introduced_bias
        cid.add_cpds(Y=lambda X, Z: X)  # type: ignore
        cid.impute_conditional_expectation_decision("D", "Y")
        assert introduced_total_effect(cid, "A", "D", "Y", 0, 1) == pytest.approx(0)

    @staticmethod
    def test_introduced_bias_y_nodep_x(imputed_cid_introduced_bias: CID) -> None:
        # Modified model where Y doesn't depend on X
        cid = imputed_cid_introduced_bias
        cid.add_cpds(Y=lambda X, Z: Z)  # type: ignore
        cid.impute_conditional_expectation_decision("D", "Y")
        assert introduced_total_effect(cid, "A", "D", "Y", 0, 1) == pytest.approx(1 / 3)

    def test_introduced_bias_reversed_sign(self) -> None:
        cbn = CausalBayesianNetwork([("A", "D"), ("A", "Y")])
        cbn.add_cpds(A=discrete_uniform([0, 1]), D=lambda A: 0, Y=lambda A: A)
        assert introduced_total_effect(cbn, "A", "D", "Y") == pytest.approx(-1)
        cbn.add_cpds(Y=lambda A: -A)
        assert introduced_total_effect(cbn, "A", "D", "Y", adapt_marginalized=True) == pytest.approx(-1)


class TestRequisite:
    @staticmethod
    def test_requisite_trim_example(cid_trim_example: CID) -> None:
        assert requisite(cid_trim_example, "D2", "Y2")

    @staticmethod
    def test_not_requisite_trim_example(cid_trim_example: CID) -> None:
        assert not requisite(cid_trim_example, "D2", "D1")


class TestRequisiteGraph:
    @staticmethod
    def test_trim_example(cid_trim_example: CID) -> None:
        assert len(cid_trim_example.edges) == 12
        assert set(cid_trim_example.get_parents("D2")) == {"Y1", "Y2", "D1", "Z1", "Z2"}

        req_graph = requisite_graph(cid_trim_example)
        assert len(req_graph.edges) == 7
        assert set(req_graph.get_parents("D2")) == {"Y2"}


class TestAdmitsVoc:
    @staticmethod
    def test_has_voc_modified_content_recommender(cid_modified_content_recommender: CID) -> None:
        assert admits_voc(cid_modified_content_recommender, "M")

    @staticmethod
    def test_no_voc_modified_content_recommender(cid_modified_content_recommender: CID) -> None:
        assert not admits_voc(cid_modified_content_recommender, "I")

    @staticmethod
    def test_invalid_target(cid_content_recommender: CID) -> None:
        with pytest.raises(KeyError):
            admits_voc(cid_content_recommender, "_")

    @staticmethod
    def test_macid_raises(macid: MACID) -> None:
        with pytest.raises(ValueError):
            admits_voc(macid, "U1")


class TestAdmitsVocList:
    @staticmethod
    def test_content_recommender(cid_content_recommender: CID) -> None:
        assert set(admits_voc_list(cid_content_recommender)) == {"O", "I", "M", "C"}

    @staticmethod
    def test_modified_content_recommender(cid_modified_content_recommender: CID) -> None:
        assert set(admits_voc_list(cid_modified_content_recommender)) == {"O", "M", "C"}

    @staticmethod
    def test_macid_raises(macid: MACID) -> None:
        with pytest.raises(ValueError):
            admits_voc_list(macid)


class TestQuantitativeVoc:
    @staticmethod
    def test_no_quantitative_voc(cid_3node: CID) -> None:
        assert set(admits_voc_list(cid_3node)) == {"U", "S"}
        assert quantitative_voc(cid_3node, "S") == pytest.approx(
            0
        )  # in this parameterisation, S has no value of control

    @staticmethod
    def test_positive_quantitative_voc(cid_3node: CID) -> None:
        cid_3node.remove_edge("S", "D")
        assert quantitative_voc(cid_3node, "S") == pytest.approx(1)  # the agent at D no longer knows the value of S

    @staticmethod
    def test_invalid_target(cid_3node: CID) -> None:
        with pytest.raises(KeyError):
            quantitative_voc(cid_3node, "_")


class TestAdmitsIci:
    @staticmethod
    def test_has_ici_content_recommender(cid_content_recommender: CID) -> None:
        assert admits_ici(cid_content_recommender, "P", "I")

    @staticmethod
    def test_no_ici_content_recommender(cid_content_recommender: CID) -> None:
        assert not admits_ici(cid_content_recommender, "P", "O")

    @staticmethod
    def test_invalid_decision(cid_content_recommender: CID) -> None:
        with pytest.raises(KeyError):
            admits_ici(cid_content_recommender, "_", "O")

    @staticmethod
    def test_invalid_target(cid_content_recommender: CID) -> None:
        with pytest.raises(KeyError):
            admits_ici(cid_content_recommender, "P", "_")

    @staticmethod
    def test_macid_raises(macid: MACID) -> None:
        with pytest.raises(ValueError):
            admits_ici(macid, "D2", "D1")


class TestAdmitsIciList:
    @staticmethod
    def test_content_recommender(cid_content_recommender: CID) -> None:
        assert set(admits_ici_list(cid_content_recommender, "P")) == {"I", "P", "C"}

    @staticmethod
    def test_macid_raises(macid: MACID) -> None:
        with pytest.raises(ValueError):
            admits_ici_list(macid, "D2")


class TestAdmitsRi:
    @staticmethod
    def test_has_ri_grade_predictor(cid_grade_predictor: CID) -> None:
        assert admits_ri(cid_grade_predictor, "P", "R")

    @staticmethod
    def test_no_ri_grade_predictor(cid_grade_predictor: CID) -> None:
        assert not admits_ri(cid_grade_predictor, "P", "E")

    @staticmethod
    def test_invalid_decision(cid_grade_predictor: CID) -> None:
        with pytest.raises(KeyError):
            admits_ri(cid_grade_predictor, "_", "E")

    @staticmethod
    def test_invalid_target(cid_grade_predictor: CID) -> None:
        with pytest.raises(KeyError):
            admits_ri(cid_grade_predictor, "P", "_")

    @staticmethod
    def test_macid_raises(macid: MACID) -> None:
        with pytest.raises(ValueError):
            admits_ri(macid, "D2", "D1")


class TestAdmitsRiList:
    @staticmethod
    def test_grade_predictor(cid_grade_predictor: CID) -> None:
        assert set(admits_ri_list(cid_grade_predictor, "P")) == {"R", "HS"}

        cid_grade_predictor.remove_edge("HS", "P")
        assert set(admits_ri_list(cid_grade_predictor, "P")) == set()

    @staticmethod
    def test_macid_raises(macid: MACID) -> None:
        with pytest.raises(ValueError):
            admits_ri_list(macid, "D2")


class TestAdmitsIndirVoc:
    @staticmethod
    def test_has_indir_voc_fitness_tracker(cid_fitness_tracker: CID) -> None:
        assert admits_indir_voc(cid_fitness_tracker, "C", "SC")

    @staticmethod
    def test_no_indir_voc_fitness_tracker(cid_fitness_tracker: CID) -> None:
        assert not admits_indir_voc(cid_fitness_tracker, "C", "TF")

    @staticmethod
    def test_invalid_decision(cid_fitness_tracker: CID) -> None:
        with pytest.raises(KeyError):
            admits_indir_voc(cid_fitness_tracker, "_", "TF")

    @staticmethod
    def test_invalid_target(cid_fitness_tracker: CID) -> None:
        with pytest.raises(KeyError):
            admits_indir_voc(cid_fitness_tracker, "C", "_")

    @staticmethod
    def test_macid_raises(macid: MACID) -> None:
        with pytest.raises(ValueError):
            admits_indir_voc(macid, "D2", "D1")


class TestAdmitsIndirVocList:
    @staticmethod
    def test_fitness_tracker(cid_fitness_tracker: CID) -> None:
        assert set(admits_indir_voc_list(cid_fitness_tracker, "C")) == {"SC"}

    @staticmethod
    def test_macid_raises(macid: MACID) -> None:
        with pytest.raises(ValueError):
            admits_indir_voc_list(macid, "D2")


class TestAdmitsDirVoc:
    @staticmethod
    def test_has_dir_voc_fitness_tracker(cid_fitness_tracker: CID) -> None:
        assert admits_dir_voc(cid_fitness_tracker, "F")

    @staticmethod
    def test_no_dir_voc_fitness_tracker(cid_fitness_tracker: CID) -> None:
        assert not admits_dir_voc(cid_fitness_tracker, "TF")

    @staticmethod
    def test_invalid_target(cid_fitness_tracker: CID) -> None:
        with pytest.raises(KeyError):
            admits_dir_voc(cid_fitness_tracker, "_")

    @staticmethod
    def test_macid_raises(macid: MACID) -> None:
        with pytest.raises(ValueError):
            admits_dir_voc(macid, "D2")


class TestAdmitsDirVocList:
    @staticmethod
    def test_fitness_tracker(cid_fitness_tracker: CID) -> None:
        assert set(admits_dir_voc_list(cid_fitness_tracker)) == {"F", "P"}

    @staticmethod
    def test_macid_raises(macid: MACID) -> None:
        with pytest.raises(ValueError):
            admits_dir_voc_list(macid)


if __name__ == "__main__":
    pytest.main(sys.argv)
