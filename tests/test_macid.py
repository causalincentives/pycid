import sys
import unittest

import numpy as np
import pytest

from pycid.core import MACID
from pycid.examples.simple_macids import (
    basic_different_dec_cardinality,
    five_agent_with_mixed_spe,
    get_basic_subgames,
    get_basic_subgames3,
    three_agent_maid,
)
from pycid.examples.story_macids import (
    battle_of_the_sexes,
    matching_pennies,
    modified_taxi_competition,
    prisoners_dilemma,
    rock_paper_scissors,
    taxi_competition,
)


class TestMACID(unittest.TestCase):
    @unittest.skip("")
    def test_get_ne(self) -> None:
        macid = prisoners_dilemma()
        self.assertEqual(len(macid.get_ne()), 1)
        pne = macid.get_ne()[0]
        macid.add_cpds(*pne)
        self.assertEqual(macid.expected_utility({}, agent=1), -2)
        self.assertEqual(macid.expected_utility({}, agent=2), -2)

        macid2 = battle_of_the_sexes()
        self.assertEqual(len(macid2.get_ne()), 2)
        self.assertEqual(len(macid2.get_ne(mixed_ne=True)), 3)

        macid3 = matching_pennies()
        self.assertEqual(len(macid3.get_ne()), 0)
        mixed_nes = macid3.get_ne(mixed_ne=True)
        self.assertEqual(len(mixed_nes), 1)
        mixed_ne = mixed_nes[0]
        macid3.add_cpds(*mixed_ne)
        self.assertEqual(macid3.expected_utility({}, agent=1), 0)
        self.assertEqual(macid3.expected_utility({}, agent=2), 0)

        macid4 = rock_paper_scissors()
        self.assertEqual(len(macid4.get_ne()), 0)
        mne = macid4.get_ne(mixed_ne=True)
        self.assertEqual(len(mne), 1)
        macid4.add_cpds(*mne[0])
        self.assertEqual(macid4.expected_utility({}, agent=1), 0)
        self.assertEqual(macid4.expected_utility({}, agent=2), 0)

        macid5 = three_agent_maid()
        self.assertEqual(len(macid5.get_ne()), 5)
        with self.assertRaises(ValueError):
            macid5.get_ne(mixed_ne=True)

    @unittest.skip("")
    def test_get_spe(self) -> None:
        macid = taxi_competition()
        all_spe = macid.get_spe()
        self.assertTrue(len(all_spe) == 1)
        spe = all_spe[0]
        joint_policy = macid.policy_profile_assignment(spe)
        cpd_d1 = joint_policy["D1"]
        cpd_d2 = joint_policy["D2"]
        self.assertTrue(np.array_equal(cpd_d1.values, np.array([1, 0])))
        self.assertTrue(np.array_equal(cpd_d2.values, np.array([[0, 1], [1, 0]])))
        self.assertTrue(len(macid.get_spe(mixed_ne=True)) == 1)

        macid = modified_taxi_competition()
        all_spe = macid.get_spe()
        self.assertTrue(len(all_spe) == 2)

        macid = prisoners_dilemma()
        all_spe = macid.get_spe()
        self.assertTrue(len(all_spe) == 1)

        macid = battle_of_the_sexes()
        self.assertTrue(len(macid.get_spe()) == 2)
        self.assertTrue(len(macid.get_spe(mixed_ne=True)) == 3)

        macid3 = basic_different_dec_cardinality()
        all_spe = macid3.get_spe()
        spe = all_spe[0]
        joint_policy = macid3.policy_profile_assignment(spe)
        cpd_d1 = joint_policy["D1"]
        cpd_d2 = joint_policy["D2"]
        self.assertTrue(np.array_equal(cpd_d1.values, np.array([0, 1])))
        self.assertTrue(np.array_equal(cpd_d2.values, np.array([[0, 0], [1, 0], [0, 1]])))

        macid4 = five_agent_with_mixed_spe()
        self.assertTrue(len(macid4.get_ne()) == 0)
        with self.assertRaises(ValueError):
            macid4.get_ne(mixed_ne=True)
        self.assertTrue(len(macid4.get_spe(mixed_ne=True)) == 1)

    @unittest.skip("")
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

    @unittest.skip("")
    def test_get_ne_in_sg(self) -> None:
        macid = taxi_competition()
        ne_in_subgame = macid.get_ne_in_sg(decisions_in_sg=["D2"])
        policy_assignment = macid.policy_profile_assignment(ne_in_subgame[0])
        cpd_d2 = policy_assignment["D2"]
        self.assertTrue(np.array_equal(cpd_d2.values, np.array([[0, 1], [1, 0]])))
        self.assertFalse(policy_assignment["D1"])
        ne_in_full_macid = macid.get_ne_in_sg()
        self.assertEqual(len(ne_in_full_macid), 3)
        with self.assertRaises(KeyError):
            macid.get_ne_in_sg(decisions_in_sg=["D3"])

        macid2 = five_agent_with_mixed_spe()
        self.assertTrue(len(macid2.get_ne_in_sg(["D1", "D2"], mixed_ne=True)) == 1)

    # @unittest.skip("")
    def test_is_nash(self) -> None:
        macid = prisoners_dilemma()
        macid.model["D1"] = {"d": 1, "c": 0}
        macid.model["D2"] = {"d": 1, "c": 0}
        profile = [macid.get_cpds(d) for d in macid.decisions]
        self.assertTrue(macid.is_nash(profile))
        macid.model["D1"] = {"d": 0, "c": 1}
        profile2 = [macid.get_cpds(d) for d in macid.decisions]
        self.assertFalse(macid.is_nash(profile2))

    # @unittest.skip("")
    def test_mixed_policy(self) -> None:
        macid = prisoners_dilemma()
        pure_policies = tuple(macid.pure_policies(macid.agent_decisions[1]))
        mixed_policy = list(macid.mixed_policy(pure_policies, [0.5, 0.5]))
        macid.add_cpds(*mixed_policy)
        self.assertTrue(np.array_equal(macid.get_cpds("D1").values, np.array([0.5, 0.5])))

    @unittest.skip("")
    def test_decs_in_each_maid_subgame(self) -> None:
        macid = prisoners_dilemma()
        self.assertCountEqual(macid.decs_in_each_maid_subgame(), [{"D1", "D2"}])
        macid = get_basic_subgames()
        self.assertTrue(len(macid.decs_in_each_maid_subgame()) == 4)
        macid = get_basic_subgames3()
        self.assertTrue(len(macid.decs_in_each_maid_subgame()) == 5)


if __name__ == "__main__":
    pytest.main(sys.argv)
