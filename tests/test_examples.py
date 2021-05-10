from __future__ import annotations

import sys
from typing import Any, Callable

import pytest

from pycid.core.causal_bayesian_network import CausalBayesianNetwork
from pycid.examples import simple_cbns, simple_cids, simple_macids, story_cids, story_macids

CONSTRUCTORS = [
    # Simple CBN
    simple_cbns.get_3node_cbn,
    simple_cbns.get_3node_uniform_cbn,
    # Simple CID
    simple_cids.get_minimal_cid,
    simple_cids.get_3node_cid,
    simple_cids.get_5node_cid,
    simple_cids.get_5node_cid_with_scaled_utility,
    simple_cids.get_2dec_cid,
    simple_cids.get_sequential_cid,
    simple_cids.get_insufficient_recall_cid,
    simple_cids.get_trim_example_cid,
    simple_cids.get_quantitative_voi_cid,
    # Simple MACID
    simple_macids.basic2agent_tie_break,
    simple_macids.basic_different_dec_cardinality,
    simple_macids.get_basic_subgames,
    simple_macids.get_basic_subgames2,
    simple_macids.get_basic_subgames3,
    simple_macids.get_path_example,
    simple_macids.two_agent_no_pne,
    simple_macids.two_agent_one_pne,
    simple_macids.two_agent_two_pne,
    simple_macids.two_agents_three_actions,
    # Story CID
    story_cids.get_fitness_tracker,
    story_cids.get_introduced_bias,
    story_cids.get_car_accident_predictor,
    story_cids.get_content_recommender,
    story_cids.get_content_recommender2,
    story_cids.get_modified_content_recommender,
    story_cids.get_grade_predictor,
    # Story MACID
    story_macids.prisoners_dilemma,
    story_macids.battle_of_the_sexes,
    story_macids.matching_pennies,
    story_macids.taxi_competition,
    story_macids.modified_taxi_competition,
    story_macids.tree_doctor,
    story_macids.forgetful_movie_star,
    story_macids.subgame_difference,
    story_macids.road_example,
    story_macids.politician,
    story_macids.umbrella,
    story_macids.sequential,
    story_macids.signal,
    story_macids.triage,
    story_macids.robot_warehouse,
]


@pytest.fixture(params=CONSTRUCTORS)
def graph_constructor(request: Any) -> Callable[[], CausalBayesianNetwork]:
    return request.param  # type: ignore


def test_constructs_macid_base(graph_constructor: Callable) -> None:
    graph = graph_constructor()
    assert isinstance(graph, CausalBayesianNetwork)
    if graph.get_cpds():
        assert graph.check_model()


if __name__ == "__main__":
    pytest.main(sys.argv)
