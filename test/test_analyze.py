import sys
import os
import unittest
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
from analyze.effects import introduced_total_effect, total_effect
from analyze.value_of_information import admits_voi, admits_voi_list
from core.cpd import FunctionCPD
from examples.simple_cids import get_minimal_cid, get_trim_example_cid
from examples.story_cids import get_fitness_tracker, get_introduced_bias, get_content_recommender, \
    get_modified_content_recommender, get_grade_predictor
from analyze.requisite_graph import nonrequisite, requisite_graph
from analyze.value_of_control import admits_voc, admits_voc_list, admits_indir_voc, admits_indir_voc_list, \
    admits_dir_voc, admits_dir_voc_list
from analyze.response_incentive import admits_ri, admits_ri_list
from analyze.instrumental_control_incentive import admits_ici, admits_ici_list


class TestAnalyze(unittest.TestCase):

    # @unittest.skip("")
    def test_value_of_information(self) -> None:
        cid = get_introduced_bias()
        self.assertTrue(admits_voi(cid, 'D', 'A'))
        self.assertEqual(set(admits_voi_list(cid, 'D')), {'A', 'X', 'Z', 'Y'})
        cid2 = get_grade_predictor()
        self.assertCountEqual(admits_voi_list(cid2, 'P'), ['HS', 'E', 'Gr'])
        self.assertFalse(admits_voi(cid2, 'P', 'Ge'))
        with self.assertRaises(Exception):
            admits_voi(cid2, 'P', 'A')
        with self.assertRaises(Exception):
            admits_voi(cid2, 'B', 'Ge')
        cid2.remove_edge('HS', 'P')
        self.assertCountEqual(admits_voi_list(cid2, 'P'), ['R', 'HS', 'E', 'Gr'])

    # @unittest.skip("")
    def test_total_effect(self) -> None:
        cid = get_minimal_cid()
        cid.impute_random_policy()
        self.assertEqual(total_effect(cid, 'A', 'B', 0, 1), 1)
        cid = get_introduced_bias()
        cid.impute_random_policy()
        self.assertEqual(total_effect(cid, 'A', 'X', 0, 1), 0.5)
        self.assertEqual(total_effect(cid, 'A', 'D', 0, 1), 0)
        self.assertEqual(total_effect(cid, 'A', 'Y', 0, 1), 0.5)

    # @unittest.skip("")
    def test_introduced_total_effect(self) -> None:
        cid = get_introduced_bias()
        cid.impute_random_policy()
        self.assertEqual(introduced_total_effect(cid, 'A', 'D', 'Y', 0, 1), -0.5)
        cid.impute_conditional_expectation_decision('D', 'Y')
        self.assertAlmostEqual(introduced_total_effect(cid, 'A', 'D', 'Y', 0, 1), 0.3333, 2)
        # Try modified model where X doesn't depend on Z
        cid = get_introduced_bias()
        cid.impute_random_policy()
        cid.add_cpds(FunctionCPD('X', lambda a, z: a, evidence=['A', 'Z']))
        cid.impute_conditional_expectation_decision('D', 'Y')
        self.assertAlmostEqual(introduced_total_effect(cid, 'A', 'D', 'Y', 0, 1), 0, 2)
        # Try modified model where Y doesn't depend on Z
        cid = get_introduced_bias()
        cid.impute_random_policy()
        cid.add_cpds(FunctionCPD('Y', lambda x, z: x, evidence=['X', 'Z']))
        cid.impute_conditional_expectation_decision('D', 'Y')
        self.assertAlmostEqual(introduced_total_effect(cid, 'A', 'D', 'Y', 0, 1), 0, 2)
        # Try modified model where Y doesn't depend on X
        cid = get_introduced_bias()
        cid.impute_random_policy()
        cid.add_cpds(FunctionCPD('Y', lambda x, z: z, evidence=['X', 'Z']))
        cid.impute_conditional_expectation_decision('D', 'Y')
        self.assertAlmostEqual(introduced_total_effect(cid, 'A', 'D', 'Y', 0, 1), 0.333, 2)

    def test_requisite_graph(self) -> None:
        cid = get_trim_example_cid()
        self.assertTrue(nonrequisite(cid, 'D2', 'D1'))
        self.assertFalse(nonrequisite(cid, 'D2', 'Y2'))
        self.assertCountEqual(cid.get_parents('D2'), ['Y1', 'Y2', 'D1', 'Z1', 'Z2'])
        self.assertEqual(len(cid.edges), 12)
        req_graph = requisite_graph(cid)
        self.assertEqual(len(req_graph.edges), 7)
        self.assertCountEqual(req_graph.get_parents('D2'), ['Y2'])

    def test_value_of_control(self) -> None:
        cid = get_content_recommender()
        self.assertCountEqual(admits_voc_list(cid, 'P'), ['O', 'I', 'M', 'C'])
        cid2 = get_modified_content_recommender()
        self.assertCountEqual(admits_voc_list(cid2, 'P'), ['O', 'M', 'C'])
        self.assertTrue(admits_voc(cid2, 'P', 'M'))
        self.assertFalse(admits_voc(cid2, 'P', 'I'))
        with self.assertRaises(Exception):
            admits_voc(cid2, 'P', 'A')
        with self.assertRaises(Exception):
            admits_voc(cid2, 'B', 'I')

    def test_instrumental_control_incentive(self) -> None:
        cid = get_content_recommender()
        self.assertTrue(admits_ici(cid, 'P', 'I'))
        self.assertFalse(admits_ici(cid, 'P', 'O'))
        self.assertCountEqual(admits_ici_list(cid, 'P'), ['I', 'P', 'C'])
        with self.assertRaises(Exception):
            admits_voi(cid, 'P', 'A')
        with self.assertRaises(Exception):
            admits_voi(cid, 'B', 'O')

    def test_response_incentive(self) -> None:
        cid = get_grade_predictor()
        self.assertCountEqual(admits_ri_list(cid, 'P'), ['R', 'HS'])
        self.assertFalse(admits_ri(cid, 'P', 'E'))
        self.assertTrue(admits_ri(cid, 'P', 'R'))
        cid.remove_edge('HS', 'P')
        self.assertEqual(admits_ri_list(cid, 'P'), [])
        with self.assertRaises(Exception):
            admits_ri(cid, 'P', 'A')
        with self.assertRaises(Exception):
            admits_ri(cid, 'B', 'E')

    def test_indirect_value_of_control(self) -> None:
        cid = get_fitness_tracker()
        self.assertFalse(admits_indir_voc(cid, 'C', 'TF'))
        self.assertTrue(admits_indir_voc(cid, 'C', 'SC'))
        self.assertCountEqual(admits_indir_voc_list(cid, 'C'), ['SC'])
        with self.assertRaises(Exception):
            admits_voi(cid, 'C', 'A')
        with self.assertRaises(Exception):
            admits_voi(cid, 'B', 'TF')

    def test_direct_value_of_control(self) -> None:
        cid = get_fitness_tracker()
        self.assertFalse(admits_dir_voc(cid, 'C', 'TF'))
        self.assertTrue(admits_dir_voc(cid, 'C', 'F'))
        self.assertCountEqual(admits_dir_voc_list(cid, 'C'), ['F', 'P'])
        with self.assertRaises(Exception):
            admits_voi(cid, 'C', 'A')
        with self.assertRaises(Exception):
            admits_voi(cid, 'B', 'TF')


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestAnalyze)
    unittest.TextTestRunner().run(suite)
