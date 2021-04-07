import sys
import unittest

import numpy as np
import pytest

from pycid.examples.simple_macids import (
    basic_different_dec_cardinality,
    get_basic_subgames,
    get_basic_subgames3,
    two_agents_three_actions,
)
from pycid.examples.story_macids import (
    battle_of_the_sexes,
    matching_pennies,
    modified_taxi_competition,
    prisoners_dilemma,
    taxi_competition,
)


class TestMACID(unittest.TestCase):
    # @unittest.skip("")
    def test_decs_in_each_maid_subgame(self) -> None:
        macid = prisoners_dilemma()
        self.assertCountEqual(macid.decs_in_each_maid_subgame(), [{"D1", "D2"}])
        macid = get_basic_subgames()
        self.assertTrue(len(macid.decs_in_each_maid_subgame()) == 4)
        macid = get_basic_subgames3()
        self.assertTrue(len(macid.decs_in_each_maid_subgame()) == 5)

    # @unittest.skip("")
    def test_policy_profile_assignment(self) -> None:
        macid = taxi_competition()
        macid.impute_random_decision("D1")
        cpd = macid.get_cpds("D1")
        partial_policy = [cpd]
        policy_assignment = macid.policy_profile_assignment(partial_policy)
        self.assertTrue(policy_assignment["D1"])
        self.assertFalse(policy_assignment["D2"])
        macid.impute_fully_mixed_policy_profile()
        joint_policy = [macid.get_cpds(d) for d in macid.decisions]
        joint_policy_assignment = macid.policy_profile_assignment(joint_policy)
        self.assertTrue(joint_policy_assignment["D1"])
        self.assertTrue(joint_policy_assignment["D2"])
        d1_cpd = joint_policy_assignment["D1"]
        self.assertEqual(d1_cpd.domain, ["e", "c"])
        # print(d1_cpd.state_names)  # can put this in the notebook too
        self.assertTrue(np.array_equal(d1_cpd.values, np.array([0.5, 0.5])))

    # @unittest.skip("")
    def test_get_all_pure_ne(self) -> None:
        macid = prisoners_dilemma()
        self.assertEqual(len(macid.get_all_pure_ne()), 1)
        pne = macid.get_all_pure_ne()[0]
        macid.add_cpds(*pne)
        self.assertEqual(macid.expected_utility({}, agent=1), -2)
        self.assertEqual(macid.expected_utility({}, agent=2), -2)

        macid2 = battle_of_the_sexes()
        self.assertEqual(len(macid2.get_all_pure_ne()), 2)

        macid3 = matching_pennies()
        self.assertEqual(len(macid3.get_all_pure_ne()), 0)

        macid4 = two_agents_three_actions()
        self.assertEqual(len(macid4.get_all_pure_ne()), 1)

    # @unittest.skip("")
    def test_get_all_pure_ne_in_sg(self) -> None:
        macid = taxi_competition()
        ne_in_subgame = macid.get_all_pure_ne_in_sg(decisions_in_sg=["D2"])
        policy_assignment = macid.policy_profile_assignment(ne_in_subgame[0])
        cpd_d2 = policy_assignment["D2"]
        self.assertTrue(np.array_equal(cpd_d2.values, np.array([[0, 1], [1, 0]])))
        self.assertFalse(policy_assignment["D1"])
        ne_in_full_macid = macid.get_all_pure_ne_in_sg()
        self.assertEqual(len(ne_in_full_macid), 3)
        with self.assertRaises(KeyError):
            macid.get_all_pure_ne_in_sg(decisions_in_sg=["D3"])

    # @unittest.skip("")
    def test_get_all_pure_spe(self) -> None:
        macid = taxi_competition()
        all_spe = macid.get_all_pure_spe()
        self.assertTrue(len(all_spe) == 1)
        spe = all_spe[0]
        joint_policy = macid.policy_profile_assignment(spe)
        cpd_d1 = joint_policy["D1"]
        cpd_d2 = joint_policy["D2"]
        self.assertTrue(np.array_equal(cpd_d1.values, np.array([1, 0])))
        self.assertTrue(np.array_equal(cpd_d2.values, np.array([[0, 1], [1, 0]])))

        macid = modified_taxi_competition()
        all_spe = macid.get_all_pure_spe()
        self.assertTrue(len(all_spe) == 2)

        macid = prisoners_dilemma()
        all_spe = macid.get_all_pure_spe()
        self.assertTrue(len(all_spe) == 1)

        macid = battle_of_the_sexes()
        all_spe = macid.get_all_pure_spe()
        self.assertTrue(len(all_spe) == 2)

        macid3 = basic_different_dec_cardinality()
        all_spe = macid3.get_all_pure_spe()
        spe = all_spe[0]
        joint_policy = macid3.policy_profile_assignment(spe)
        cpd_d1 = joint_policy["D1"]
        cpd_d2 = joint_policy["D2"]
        self.assertTrue(np.array_equal(cpd_d1.values, np.array([0, 1])))
        self.assertTrue(np.array_equal(cpd_d2.values, np.array([[0, 0], [1, 0], [0, 1]])))


if __name__ == "__main__":
    pytest.main(sys.argv)
