import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
from examples.simple_cids import get_5node_cid, get_minimal_cid, get_2dec_cid, get_5node_cid_with_scaled_utility, \
    get_insufficient_recall_cid
from examples.simple_macids import basic2agent, basic2agent_tie_break, basic_different_dec_cardinality, \
    c2d, basic2agent_3, get_basic2agent_acyclic, get_basic2agent_cyclic
from examples.story_cids import get_introduced_bias, get_car_accident_predictor, get_fitness_tracker, \
    get_content_recommender, get_modified_content_recommender
from examples.story_macids import sequential, tree_doctor, road_example, politician, umbrella, signal, triage

from examples.generate import random_cid, random_cids
import unittest


class TestExamples(unittest.TestCase):

    def test_random_cid(self) -> None:
        random_cid(4, 1, 1)
        random_cid(8, 2, 2)
        random_cid(12, 3, 3)
        random_cids(n_cids=1)[0]

    def test_simple_cid_examples(self) -> None:
        get_minimal_cid()
        get_5node_cid()
        get_5node_cid()
        get_5node_cid_with_scaled_utility()
        get_2dec_cid()
        get_insufficient_recall_cid()

    def test_simple_macid_examples(self) -> None:
        get_basic2agent_acyclic
        get_basic2agent_cyclic
        basic2agent
        basic2agent_tie_break
        basic_different_dec_cardinality
        c2d()
        sequential()

    def test_story_cid_examples(self) -> None:
        get_introduced_bias()
        get_car_accident_predictor()
        get_fitness_tracker()
        get_content_recommender()
        get_modified_content_recommender()

    def test_story_macid_examples(self) -> None:
        tree_doctor()
        road_example()
        basic2agent_3()
        politician()
        umbrella()
        signal()
        triage()


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestExamples)
    unittest.TextTestRunner().run(suite)

# %%
