import sys

import pytest

from pycid.core.macid import MACID
from pycid.core.mechanised_graph import MechanisedGraph
from pycid.examples.story_macids import taxi_competition


@pytest.fixture
def taxi() -> MACID:
    return taxi_competition()


def test_create_mechanised_graph(taxi) -> None:
    mech_graph = MechanisedGraph(taxi)
    assert len(mech_graph.nodes) == (2 * len(taxi.nodes))
    # Assert edges are correct in terms of r-relevance: There should be an edge from 'D2_mec' to 'D1_mec'
    # because 'D2' is r-reachable from 'D1'
    # print (mech_graph.edges) if fails


def test_create_r_reachable_mechanised_graph(taxi) -> None:
    mech_graph = MechanisedGraph(taxi)
    # Assert edges are correct in terms of r-relevance: There should be an edge from 'D2_mec' to 'D1_mec'
    # because 'D2' is r-reachable from 'D1'
    assert ("D2_mec", "D1_mec") in mech_graph.edges
    # Check Utility nodes are r-reachable from decisions
    assert ("U1_mec", "D1_mec") in mech_graph.edges
    assert ("U2_mec", "D2_mec") in mech_graph.edges


if __name__ == "__main__":
    pytest.main(sys.argv)
