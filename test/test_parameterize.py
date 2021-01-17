import sys, os
sys.path.insert(0, os.path.abspath('.'))

from parameterize.generate import random_cids
from parameterize.get_systems import choose_systems, get_first_c_index
import unittest
from examples import get_3node_cid, get_5node_cid, get_2dec_cid, get_nested_cid
from parameterize.parameterize import parameterize_systems, merge_all_nodes
from verify_incentive import verify_incentive


# @unittest.skip("until Ryan/James fix")
class TestParameterize(unittest.TestCase):

    def test_random_cid(self):
        # TODO This test is flaky: sometimes gives e.g. "ValueError: D3 is an ancestor of D1"
        cid = random_cids(n_cids=1)[0]

    def test_parameterization(self):
        cid = get_3node_cid()
        D = 'D'
        X = 'S'
        systems = choose_systems(cid, D, X)
        systems[0]['i_C'] = get_first_c_index(cid, systems[0]['info'])
        all_cpds = parameterize_systems(cid, systems)
        all_cpds = [c.copy() for a in all_cpds for b in a.values() for c in b.values()]
        merged_cpds = merge_all_nodes(cid, all_cpds)
        cid.add_cpds(*merged_cpds.values())
        ev1, ev2 = verify_incentive(cid, D, X)
        self.assertEqual(ev1, 1)
        self.assertEqual(ev2, .5)

    def test_param3(self):
        cid = get_3node_cid()
        D = 'D'
        X = 'S'
        systems = choose_systems(cid, D, X)
        systems[0]['i_C'] = get_first_c_index(cid, systems[0]['info'])
        all_cpds = parameterize_systems(cid, systems)
        all_cpds = [c.copy() for a in all_cpds for b in a.values() for c in b.values()]
        merged_cpds = merge_all_nodes(cid, all_cpds)
        cid.add_cpds(*merged_cpds.values())
        ev1, ev2 = verify_incentive(cid, D, X)
        self.assertEqual(ev1, 1)
        self.assertEqual(ev2, .5)

    def test_param5(self):
        cid = get_5node_cid()
        D = 'D'
        X = 'S1'
        systems = choose_systems(cid, D, X)
        systems[0]['i_C'] = get_first_c_index(cid, systems[0]['info'])
        all_cpds = parameterize_systems(cid, systems)
        all_cpds = [c.copy() for a in all_cpds for b in a.values() for c in b.values()]
        merged_cpds = merge_all_nodes(cid, all_cpds)
        cid.add_cpds(*merged_cpds.values())
        ev1, ev2 = verify_incentive(cid, D, X)
        self.assertEqual(ev1, 1)
        self.assertEqual(ev2, .5)

    def test_2dec_cid(self):
        cid = get_2dec_cid()
        D = 'D1'
        X = 'S1'
        systems = choose_systems(cid, D, X)
        systems[0]['i_C'] = get_first_c_index(cid, systems[0]['info'])
        all_cpds = parameterize_systems(cid, systems)
        all_cpds = [c.copy() for a in all_cpds for b in a.values() for c in b.values()]
        merged_cpds = merge_all_nodes(cid, all_cpds)
        cid.add_cpds(*merged_cpds.values())
        ev1, ev2 = verify_incentive(cid, D, X)
        self.assertEqual(ev1, 1)
        self.assertEqual(ev2, .5)

    def test_param_nested(self):
        cid = get_nested_cid()
        D = 'D1'
        X = 'S1'
        systems = choose_systems(cid, D, X)
        systems[0]['i_C'] = get_first_c_index(cid, systems[0]['info'])
        all_cpds = parameterize_systems(cid, systems)
        all_cpds = [c.copy() for a in all_cpds for b in a.values() for c in b.values()]
        merged_cpds = merge_all_nodes(cid, all_cpds)
        cid.add_cpds(*merged_cpds.values())
        ev1, ev2 = verify_incentive(cid, D, X)
        self.assertEqual(ev1, 3)
        self.assertEqual(ev2, 2.75)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestParameterize)
    unittest.TextTestRunner().run(suite)
