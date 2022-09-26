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
    def test_get_ne(self) -> None:
        macid = prisoners_dilemma()
        self.assertEqual(len(macid.get_ne()), 1)
        pne = macid.get_ne()[0]
        macid.add_cpds(*pne)
        self.assertEqual(macid.expected_utility({}, agent=1), -2)
        self.assertEqual(macid.expected_utility({}, agent=2), -2)

        macid2 = battle_of_the_sexes()
        self.assertEqual(len(macid2.get_ne()), 3)
        self.assertEqual(len(macid2.get_ne("enumpure")), 2)
        self.assertEqual(len(macid2.get_ne(solver="enummixed")), 3)
        nes = macid2.get_ne(solver="enummixed")
        joint_policy = macid2.policy_profile_assignment(nes[1])  # mixed ne (order not guaranteed?)
        cpd_df = joint_policy["D_F"]
        cpd_dm = joint_policy["D_M"]
        self.assertTrue(np.allclose(cpd_df.values, np.array([0.6, 0.4])))
        self.assertTrue(np.allclose(cpd_dm.values, np.array([0.4, 0.6])))

        macid3 = matching_pennies()
        self.assertEqual(len(macid3.get_ne()), 1)  # gets the mixed NE by default
        self.assertEqual(len(macid3.get_ne("enumpure")), 0)  # no NE if overriden solver to pure NE only
        self.assertEqual(len(macid3.get_ne(solver="enummixed")), 1)
        joint_policy = macid3.policy_profile_assignment(macid3.get_ne(solver="enummixed")[0])
        cpd_d1 = joint_policy["D1"]
        cpd_d2 = joint_policy["D2"]
        self.assertTrue(np.allclose(cpd_d1.values, np.array([0.5, 0.5])))
        self.assertTrue(np.allclose(cpd_d2.values, np.array([0.5, 0.5])))
        self.assertEqual(len(macid3.get_ne(solver="lp")), 1)
        joint_policy = macid3.policy_profile_assignment(macid3.get_ne(solver="lp")[0])
        cpd_d1 = joint_policy["D1"]
        cpd_d2 = joint_policy["D2"]
        self.assertTrue(np.allclose(cpd_d1.values, np.array([0.5, 0.5])))
        self.assertTrue(np.allclose(cpd_d2.values, np.array([0.5, 0.5])))
        self.assertEqual(len(macid3.get_ne(solver="lcp")), 1)
        joint_policy = macid3.policy_profile_assignment(macid3.get_ne(solver="lcp")[0])
        cpd_d1 = joint_policy["D1"]
        cpd_d2 = joint_policy["D2"]
        self.assertTrue(np.allclose(cpd_d1.values, np.array([0.5, 0.5])))
        self.assertTrue(np.allclose(cpd_d2.values, np.array([0.5, 0.5])))
        self.assertEqual(len(macid3.get_ne(solver="simpdiv")), 1)
        joint_policy = macid3.policy_profile_assignment(macid3.get_ne(solver="simpdiv")[0])
        cpd_d1 = joint_policy["D1"]
        cpd_d2 = joint_policy["D2"]
        self.assertTrue(np.allclose(cpd_d1.values, np.array([0.5, 0.5])))
        self.assertTrue(np.allclose(cpd_d2.values, np.array([0.5, 0.5])))
        self.assertEqual(len(macid3.get_ne(solver="gnm")), 1)
        joint_policy = macid3.policy_profile_assignment(macid3.get_ne(solver="gnm")[0])
        cpd_d1 = joint_policy["D1"]
        cpd_d2 = joint_policy["D2"]
        self.assertTrue(np.allclose(cpd_d1.values, np.array([0.5, 0.5])))
        self.assertTrue(np.allclose(cpd_d2.values, np.array([0.5, 0.5])))
        self.assertEqual(len(macid3.get_ne(solver="ipa")), 1)
        joint_policy = macid3.policy_profile_assignment(macid3.get_ne(solver="ipa")[0])
        cpd_d1 = joint_policy["D1"]
        cpd_d2 = joint_policy["D2"]
        self.assertTrue(np.allclose(cpd_d1.values, np.array([0.5, 0.5])))
        self.assertTrue(np.allclose(cpd_d2.values, np.array([0.5, 0.5])))

        macid4 = two_agents_three_actions()
        self.assertEqual(len(macid4.get_ne()), 1)

    # @unittest.skip("")
    def test_get_ne_in_sg(self) -> None:
        macid = taxi_competition()
        ne_in_subgame = macid.get_ne_in_sg(decisions_in_sg=["D2"])
        policy_assignment = macid.policy_profile_assignment(ne_in_subgame[0])
        cpd_d2 = policy_assignment["D2"]
        self.assertTrue(np.array_equal(cpd_d2.values, np.array([[0, 1], [1, 0]])))
        self.assertFalse(policy_assignment["D1"])
        ne_in_full_macid = macid.get_ne_in_sg()
        self.assertEqual(len(ne_in_full_macid), 4)
        with self.assertRaises(KeyError):
            macid.get_ne_in_sg(decisions_in_sg=["D3"])

        mixed_ne_in_subgame = macid.get_ne_in_sg(decisions_in_sg=["D2"], solver="enummixed")
        self.assertEqual(len(mixed_ne_in_subgame), 1)

    # @unittest.skip("")
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

        macid = modified_taxi_competition()
        all_spe = macid.get_spe("enumpure")
        self.assertTrue(len(all_spe) == 2)

        macid = prisoners_dilemma()
        all_spe = macid.get_spe("enumpure")
        self.assertTrue(len(all_spe) == 1)

        macid = battle_of_the_sexes()
        all_spe = macid.get_spe("enumpure")
        self.assertTrue(len(all_spe) == 2)  # only 2 pure subgame perfect NE

        macid3 = basic_different_dec_cardinality()
        all_spe = macid3.get_spe("enumpure")
        spe = all_spe[0]
        joint_policy = macid3.policy_profile_assignment(spe)
        cpd_d1 = joint_policy["D1"]
        cpd_d2 = joint_policy["D2"]
        self.assertTrue(np.array_equal(cpd_d1.values, np.array([0, 1])))
        self.assertTrue(np.array_equal(cpd_d2.values, np.array([[0, 0], [1, 0], [0, 1]])))

        macid = battle_of_the_sexes()
        all_spe = macid.get_spe(solver="enummixed")
        self.assertTrue(len(all_spe) == 3)
        macid = prisoners_dilemma()
        all_spe = macid.get_spe(solver="enummixed")
        self.assertTrue(len(all_spe) == 1)
        macid = taxi_competition()
        all_spe = macid.get_spe(solver="enummixed")
        self.assertTrue(len(all_spe) == 1)
        macid = modified_taxi_competition()
        all_spe = macid.get_spe(solver="enummixed")
        self.assertTrue(len(all_spe) == 2)
        macid = basic_different_dec_cardinality()
        all_spe = macid.get_spe(solver="enummixed")
        self.assertTrue(len(all_spe) == 1)


if __name__ == "__main__":
    pytest.main(sys.argv)
