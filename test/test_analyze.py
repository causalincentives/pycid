#%%
import sys, os
import unittest
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
from analyze.effects import introduced_total_effect, total_effect
from analyze.value_of_information import admits_voi, admits_voi_list
from core.cpd import FunctionCPD

from examples.simple_cids import get_minimal_cid, get_trim_example_cid
from examples.story_cids import get_introduced_bias, get_content_recommender, get_content_recommender2, \
    get_modified_content_recommender, get_grade_predictor
from analyze.d_reduction import nonrequisite, d_reduction
from core.get_paths import find_active_path, find_all_dir_paths
from analyze.value_of_information import admits_voi_list
from analyze.value_of_control_UNTESTED import admits_voc, admits_voc_list, admits_ici, admits_ici_list
from analyze.response_incentive_UNTESTED import admits_ri, admits_ri_list

class TestAnalyze(unittest.TestCase):

    @unittest.skip("")
    def test_value_of_information(self):
        cid = get_introduced_bias()
        self.assertTrue(admits_voi(cid, 'D', 'A'))
        self.assertEqual(set(admits_voi_list(cid, 'D')), {'A', 'X', 'Z', 'Y'})

    @unittest.skip("")
    def testTotalEffect(self):
        cid = get_minimal_cid()
        cid.impute_random_policy()
        self.assertEqual(total_effect(cid, 'A', 'B', 0, 1), 1)
        cid = get_introduced_bias()
        cid.impute_random_policy()
        self.assertEqual(total_effect(cid, 'A', 'X', 0, 1), 0.5)
        self.assertEqual(total_effect(cid, 'A', 'D', 0, 1), 0)
        self.assertEqual(total_effect(cid, 'A', 'Y', 0, 1), 0.5)
 
    @unittest.skip("")
    def testIntroducedEffect(self):
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

    # def test_trim(self):
    #     cid = get_trim_example_cid()
    #     cid.draw()
    #     print(admits_voi_list(cid, 'D2'))
    #     print(admits_voi_list(cid, 'D1'))        
       
        # cid2 = trim(cid)


        # cid2.draw()

    def test_d_reduction(self):
        cid = get_trim_example_cid()
        self.assertTrue(nonrequisite(cid, 'D2', 'D1'))
        self.assertFalse(nonrequisite(cid, 'D2', 'Y2'))
        self.assertCountEqual(cid.get_parents('D2'), ['Y1', 'Y2', 'D1', 'Z1', 'Z2'])
        self.assertEqual(len(cid.edges), 12)
        reduced_cid = d_reduction(cid)
        self.assertEqual(len(reduced_cid.edges), 7)
        self.assertCountEqual(reduced_cid.get_parents('D2'), ['Y2'])


    def test_value_of_control(self):
        cid = get_content_recommender()
        self.assertCountEqual(admits_voc_list(cid, 'P'), ['O', 'I', 'M', 'C'])
        cid2 = get_modified_content_recommender()
        self.assertCountEqual(admits_voc_list(cid2, 'P'), ['O', 'M', 'C'])
        self.assertTrue(admits_voc(cid2, 'P', 'M'))
        self.assertFalse(admits_voc(cid2, 'P', 'I'))
        


    def test_instrumental_control_incentive(self):
        cid = get_content_recommender()
        self.assertTrue(admits_ici(cid, 'P', 'I'))
        self.assertFalse(admits_ici(cid, 'P', 'O'))
        self.assertCountEqual(admits_ici_list(cid, 'P'), ['I', 'P', 'C'])
        

    def test_response_incentive(self):
        cid = get_grade_predictor()
        self.assertCountEqual(admits_ri_list(cid, 'P'), ['R', 'HS'])
        self.assertFalse(admits_ri(cid, 'P', 'E'))
        self.assertTrue(admits_ri(cid, 'P', 'R'))
        cid.remove_edge('HS', 'P')
        self.assertEqual(admits_ri_list(cid, 'P'), [])
        

        

        
        # cid.draw()
        # print(len(cid.edges))

        # # print(admits_voi_list(cid, 'P'))

        # temp = trim(cid)
        # print(len(temp.edges))
        # print(temp.get_parents('D2'))
        # temp.draw()



if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestAnalyze)
    unittest.TextTestRunner().run(suite)
