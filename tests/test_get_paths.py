from __future__ import annotations

import sys
import unittest
from typing import TYPE_CHECKING

import pytest

from pycid.core.get_paths import (
    directed_decision_free_path,
    find_active_path,
    find_all_dir_paths,
    find_all_undir_paths,
    get_motif,
    get_motifs,
    is_active_backdoor_trail,
    is_active_indirect_frontdoor_trail,
    is_active_path,
)
from pycid.core.macid import MACID
from pycid.examples.simple_cids import get_3node_cid
from pycid.examples.simple_macids import get_basic_subgames, get_path_example
from pycid.examples.story_macids import taxi_competition

if TYPE_CHECKING:
    from pycid import CID


@pytest.fixture
def macid_taxi_comp() -> MACID:
    return taxi_competition()


@pytest.fixture
def macid_basic_subgames() -> MACID:
    return get_basic_subgames()


@pytest.fixture
def macid_dir_paths() -> MACID:
    return MACID(
        [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("B", "F"), ("F", "E")],
        agent_decisions={1: ["D"]},
        agent_utilities={1: ["E"]},
    )


@pytest.fixture
def macid_undir_paths() -> MACID:
    return MACID(
        [("X1", "D"), ("X2", "U")],
        agent_decisions={1: ["D"]},
        agent_utilities={1: ["U"]},
    )


@pytest.fixture
def macid_undir_paths2() -> MACID:
    return MACID(
        [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("B", "F"), ("F", "E")],
        agent_decisions={1: ["D"]},
        agent_utilities={1: ["E"]},
    )


@pytest.fixture
def cid_3node() -> CID:
    return get_3node_cid()


@pytest.fixture
def macid_path_example() -> MACID:
    return get_path_example()


class TestFindActivePath:
    @staticmethod
    def test_find_active_path(macid_taxi_comp: MACID) -> None:
        assert find_active_path(macid_taxi_comp, "D1", "U1", {"D2"}) == ["D1", "U1"]

    @staticmethod
    def test_no_path(macid_taxi_comp: MACID) -> None:
        with pytest.raises(ValueError):
            find_active_path(macid_taxi_comp, "D1", "U1", {"D2", "U1"})

    @staticmethod
    def test_invalid_target_in_observed(macid_taxi_comp: MACID) -> None:
        with pytest.raises(KeyError):
            find_active_path(macid_taxi_comp, "D1", "U1", {"D3"})

    @staticmethod
    def test_invalid_target_start(macid_taxi_comp: MACID) -> None:
        with pytest.raises(KeyError):
            find_active_path(macid_taxi_comp, "D3", "U1", {"D2"})

    @staticmethod
    def test_invalid_target_end(macid_taxi_comp: MACID) -> None:
        with pytest.raises(KeyError):
            find_active_path(macid_taxi_comp, "D1", "U3", {"D2"})


class TestGetMotif:
    @staticmethod
    def test_get_motif(macid_basic_subgames: MACID) -> None:
        assert get_motif(macid_basic_subgames, ["D3", "D2", "U2", "D11", "D12", "U3"], 0) == "backward"
        assert get_motif(macid_basic_subgames, ["D3", "D2", "U2", "D11", "D12", "U3"], 1) == "fork"
        assert get_motif(macid_basic_subgames, ["D3", "D2", "U2", "D11", "D12", "U3"], 2) == "collider"
        assert get_motif(macid_basic_subgames, ["D3", "D2", "U2", "D11", "D12", "U3"], 4) == "forward"
        assert get_motif(macid_basic_subgames, ["D3", "D2", "U2", "D11", "D12", "U3"], 5) == "endpoint"

    @staticmethod
    def test_invalid_target(macid_basic_subgames: MACID) -> None:
        with pytest.raises(KeyError):
            get_motif(macid_basic_subgames, ["D3", "_", "U2", "D11", "D12", "U3"], 5)

    @staticmethod
    def test_invalid_index(macid_basic_subgames: MACID) -> None:
        with pytest.raises(IndexError):
            get_motif(macid_basic_subgames, ["D3", "D2", "U2", "D11", "D12", "U3"], 6)


class TestGetMotifs:
    @staticmethod
    def test_get_motifs(macid_basic_subgames: MACID) -> None:
        motifs = get_motifs(macid_basic_subgames, ["D3", "D2", "U2", "D11", "D12", "U3"])
        assert motifs == ["backward", "fork", "collider", "fork", "forward", "endpoint"]

    @staticmethod
    def test_invalid_target(macid_basic_subgames: MACID) -> None:
        with pytest.raises(KeyError):
            get_motif(macid_basic_subgames, ["D3", "_", "U2", "D11", "D12", "U3"], 5)


class TestFindAllDirPaths:
    @staticmethod
    def test_find_all_dir_paths(macid_dir_paths: MACID) -> None:
        assert list(find_all_dir_paths(macid_dir_paths, "A", "E")) == [["A", "B", "C", "D", "E"], ["A", "B", "F", "E"]]
        assert list(find_all_dir_paths(macid_dir_paths, "C", "E")) == [["C", "D", "E"]]

    @staticmethod
    def test_no_dir_paths(macid_dir_paths: MACID) -> None:
        assert not list(find_all_dir_paths(macid_dir_paths, "F", "A"))

    @staticmethod
    def test_count_dir_paths(macid_dir_paths: MACID) -> None:
        assert len(list(find_all_dir_paths(macid_dir_paths, "B", "E"))) == 2

    @staticmethod
    def test_invalid_target(macid_dir_paths: MACID) -> None:
        with pytest.raises(KeyError):
            find_all_dir_paths(macid_dir_paths, "U2", "A")


class TestFindAllUndirPaths:
    @staticmethod
    def test_undir_paths_in_3node(cid_3node: CID) -> None:
        assert len(list(find_all_undir_paths(cid_3node, "S", "U"))) == 2

    @staticmethod
    def test_invalid_target(cid_3node: CID) -> None:
        with pytest.raises(KeyError):
            find_all_undir_paths(cid_3node, "S", "A")

    @staticmethod
    def test_undir_example(macid_undir_paths: MACID) -> None:
        assert list(find_all_undir_paths(macid_undir_paths, "X1", "D")) == [["X1", "D"]]

    @staticmethod
    def test_no_undir_paths(macid_undir_paths: MACID) -> None:
        assert not list(find_all_undir_paths(macid_undir_paths, "X1", "U"))

    @staticmethod
    def test_undir_paths2(macid_undir_paths2: MACID) -> None:
        case = unittest.TestCase()

        case.assertCountEqual(
            list(find_all_undir_paths(macid_undir_paths2, "F", "A")), [["F", "E", "D", "C", "B", "A"], ["F", "B", "A"]]
        )


class TestDirectedDecisionFreePath:
    @staticmethod
    def test_directed_decision_free_path_exists(macid_basic_subgames: MACID) -> None:
        assert directed_decision_free_path(macid_basic_subgames, "X1", "U11")
        assert directed_decision_free_path(macid_basic_subgames, "X2", "U22")

    @staticmethod
    def test_no_directed_decision_free_path_exists(macid_basic_subgames: MACID) -> None:
        assert not directed_decision_free_path(macid_basic_subgames, "X2", "U3")
        assert not directed_decision_free_path(macid_basic_subgames, "X2", "U2")
        assert not directed_decision_free_path(macid_basic_subgames, "U22", "U3")

    @staticmethod
    def test_invalid_target(macid_basic_subgames: MACID) -> None:
        with pytest.raises(KeyError):
            directed_decision_free_path(macid_basic_subgames, "X1", "A")


class TestActivePath:
    @staticmethod
    def test_active_path_exists(macid_path_example: MACID) -> None:
        assert is_active_path(macid_path_example, ["X1", "D", "U"])
        assert is_active_path(macid_path_example, ["X1", "D", "X2"], {"D"})
        assert is_active_path(macid_path_example, ["X1", "D", "X2"], {"U"})

    @staticmethod
    def test_no_active_path_exists(macid_path_example: MACID) -> None:
        assert not is_active_path(macid_path_example, ["X1", "D", "U"], {"D"})
        assert not is_active_path(macid_path_example, ["X1", "D", "X2"])

    @staticmethod
    def test_invalid_path(macid_path_example: MACID) -> None:
        with pytest.raises(KeyError):
            is_active_path(macid_path_example, ["X1", "D", "_"], {"U"})

    @staticmethod
    def test_invalid_observed_node(macid_path_example: MACID) -> None:
        with pytest.raises(KeyError):
            is_active_path(macid_path_example, ["X1", "D", "X2"], {"_"})


class TestActiveIndirectFrontdoorTrail:
    @staticmethod
    def test_active_indirect_frontdoor_trail_exists(macid_path_example: MACID) -> None:
        assert is_active_indirect_frontdoor_trail(macid_path_example, "X2", "X1", {"D"})

    @staticmethod
    def test_no_active_indirect_frontdoor_trail_exists(macid_path_example: MACID) -> None:
        assert not is_active_indirect_frontdoor_trail(macid_path_example, "X2", "X1")
        assert not is_active_indirect_frontdoor_trail(macid_path_example, "X3", "X1", {"D"})
        assert not is_active_indirect_frontdoor_trail(macid_path_example, "X3", "X1")
        assert not is_active_indirect_frontdoor_trail(macid_path_example, "X1", "U")
        assert not is_active_indirect_frontdoor_trail(macid_path_example, "X1", "U", {"D", "X2"})

    @staticmethod
    def test_invalid_start_node(macid_path_example: MACID) -> None:
        with pytest.raises(KeyError):
            is_active_indirect_frontdoor_trail(macid_path_example, "_", "U", {"D", "X2"})

    @staticmethod
    def test_invalid_observed_nodes(macid_path_example: MACID) -> None:
        with pytest.raises(KeyError):
            is_active_indirect_frontdoor_trail(macid_path_example, "X1", "U", {"A", "X2"})


class TestActiveBackdoorTrail:
    @staticmethod
    def test_active_backdoor_trail_exists(macid_path_example: MACID) -> None:
        assert is_active_backdoor_trail(macid_path_example, "X3", "X2", {"D"})

    @staticmethod
    def test_no_active_backdoor_trail_exists(macid_path_example: MACID) -> None:
        assert not is_active_backdoor_trail(macid_path_example, "X3", "X2")
        assert not is_active_backdoor_trail(macid_path_example, "X1", "X2")
        assert not is_active_backdoor_trail(macid_path_example, "X1", "X2", {"D"})

    @staticmethod
    def test_invalid_start_node(macid_path_example: MACID) -> None:
        with pytest.raises(KeyError):
            is_active_backdoor_trail(macid_path_example, "_", "X2", {"D"})

    @staticmethod
    def test_invalid_observed_nodes(macid_path_example: MACID) -> None:
        with pytest.raises(KeyError):
            is_active_backdoor_trail(macid_path_example, "X3", "X2", {"_"})


if __name__ == "__main__":
    pytest.main(sys.argv)

# %%
