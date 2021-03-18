import sys
import unittest

import pytest

from pycid.examples.simple_cids import (
    get_2dec_cid,
    get_3node_cid,
    get_5node_cid,
    get_5node_cid_with_scaled_utility,
    get_insufficient_recall_cid,
    get_minimal_cid,
    get_sequential_cid,
    get_trim_example_cid,
)
from pycid.examples.simple_macids import (
    basic2agent_tie_break,
    basic_different_dec_cardinality,
    get_basic_subgames,
    get_basic_subgames2,
    get_basic_subgames3,
    get_path_example,
    two_agent_no_pne,
    two_agent_one_pne,
    two_agent_two_pne,
    two_agents_three_actions,
)
from pycid.examples.story_cids import (
    get_car_accident_predictor,
    get_content_recommender,
    get_content_recommender2,
    get_fitness_tracker,
    get_grade_predictor,
    get_introduced_bias,
    get_modified_content_recommender,
)
from pycid.examples.story_macids import (
    battle_of_the_sexes,
    forgetful_movie_star,
    matching_pennies,
    modified_taxi_competition,
    politician,
    prisoners_dilemma,
    road_example,
    sequential,
    signal,
    subgame_difference,
    taxi_competition,
    tree_doctor,
    triage,
    umbrella,
)


class TestExamples(unittest.TestCase):
    def test_simple_cid_examples(self) -> None:
        get_minimal_cid()
        get_3node_cid()
        get_5node_cid()
        get_5node_cid_with_scaled_utility()
        get_2dec_cid()
        get_sequential_cid()
        get_insufficient_recall_cid()
        get_trim_example_cid()

    def test_simple_macid_examples(self) -> None:
        get_basic_subgames()
        get_basic_subgames2()
        get_basic_subgames3()
        get_path_example()
        basic2agent_tie_break()
        two_agent_one_pne()
        two_agent_two_pne()
        two_agent_no_pne()
        two_agents_three_actions()
        basic_different_dec_cardinality()

    def test_story_cid_examples(self) -> None:
        get_fitness_tracker()
        get_introduced_bias()
        get_car_accident_predictor()
        get_content_recommender()
        get_content_recommender2()
        get_modified_content_recommender()
        get_grade_predictor()

    def test_story_macid_examples(self) -> None:
        prisoners_dilemma()
        battle_of_the_sexes()
        matching_pennies()
        taxi_competition()
        modified_taxi_competition()
        tree_doctor()
        forgetful_movie_star()
        subgame_difference()
        road_example()
        politician()
        umbrella()
        sequential()
        signal()
        triage()


if __name__ == "__main__":
    pytest.main(sys.argv)
