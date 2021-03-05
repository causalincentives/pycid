import logging
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
from examples.simple_cids import get_minimal_cid, get_3node_cid, get_5node_cid, get_5node_cid_with_scaled_utility, \
    get_2dec_cid, get_sequential_cid, get_insufficient_recall_cid, get_trim_example_cid
from examples.simple_macids import get_basic_subgames, get_basic_subgames2, get_basic_subgames3, get_path_example, \
    basic2agent_tie_break, two_agent_one_pne, two_agent_two_pne, two_agent_no_pne, two_agents_three_actions, \
    basic_different_dec_cardinality
from examples.story_cids import get_introduced_bias, get_car_accident_predictor, get_fitness_tracker, \
    get_content_recommender, get_content_recommender2, get_modified_content_recommender, get_grade_predictor
from examples.story_macids import prisoners_dilemma, battle_of_the_sexes, matching_pennies, \
    taxi_competition, modified_taxi_competition, tree_doctor, forgetful_movie_star, subgame_difference, \
    road_example, politician, umbrella, sequential, signal, triage

from examples.generate import random_cid, random_cids
import unittest


class TestExamples(unittest.TestCase):

    def setUp(self) -> None:
        logging.disable()

    def test_random_cid(self) -> None:
        random_cid(4, 1, 1)
        random_cid(8, 2, 2)
        random_cid(12, 3, 3)
        random_cids(n_cids=1)[0]

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
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestExamples)
    unittest.TextTestRunner().run(suite)

# %%
