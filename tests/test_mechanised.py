import sys

import pytest

from pycid.core.macid import MACID
from pycid.core.mechanised_graph import MechanisedGraph
from pycid.examples.story_macids import forgetful_movie_star, taxi_competition


@pytest.fixture
def taxi() -> MACID:
    return taxi_competition()


@pytest.fixture
def movie_star() -> MACID:
    return forgetful_movie_star()


def test_create_mechanised_graph(taxi: MACID) -> None:
    mech_graph = MechanisedGraph(taxi)
    assert len(mech_graph.graph.nodes) == (2 * len(taxi.nodes))
    # Assert edges are correct in terms of r-relevance: There should be an edge from 'D2_mec' to 'D1_mec'
    # because 'D2' is r-reachable from 'D1'
    # print (mech_graph.edges) if fails


def test_create_r_reachable_mechanised_graph(taxi: MACID) -> None:
    mech = MechanisedGraph(taxi)
    # Assert edges are correct in terms of r-relevance: There should be an edge from 'D2_mec' to 'D1_mec'
    # because 'D2' is r-reachable from 'D1'
    assert ("D2_mec", "D1_mec") in mech.graph.edges
    # Check Utility nodes are r-reachable from decisions
    assert ("U1_mec", "D1_mec") in mech.graph.edges
    assert ("U2_mec", "D2_mec") in mech.graph.edges


def test_sufficient_recall_single(movie_star: MACID) -> None:
    mech = MechanisedGraph(movie_star)
    assert mech.is_sufficient_recall(2)
    assert not mech.is_sufficient_recall(1)


def test_sufficient_recall_all(movie_star: MACID) -> None:
    mech = MechanisedGraph(movie_star)
    assert not mech.is_sufficient_recall()


if __name__ == "__main__":
    pytest.main(sys.argv)
