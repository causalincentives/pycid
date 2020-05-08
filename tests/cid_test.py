#Licensed to the Apache Software Foundation (ASF) under one or more contributor license
#agreements; and to You under the Apache License, Version 2.0.

import sys, os
sys.path.insert(0, os.path.abspath('.'))
import unittest
import numpy as np

#from models.two_decisions import TwoDecisions

from examples import get_3node_cid, get_5node_cid, get_2dec_cid, get_nested_cid
from pgmpy.factors.discrete import TabularCPD
from get_systems import choose_systems, get_first_c_index
from parameterize import parameterize_systems, merge_all_nodes
from verify_incentive import verify_incentive

class TestCIDClass(unittest.TestCase):

    def test_assign_cpd(self):
        three_node = get_3node_cid()
        three_node.add_cpds(TabularCPD('D', 2, np.eye(2), evidence=['S'], evidence_card=[2]))
        three_node.check_model()
        cpd = three_node.get_cpds('D').values
        self.assertTrue(np.array_equal(cpd, np.array([[1, 0], [0, 1]])))

    def test_expected_utility(self):
        three_node = get_3node_cid()
        five_node = get_5node_cid()
        eu00 = three_node.expected_utility({'D': 0, 'S': 0})
        self.assertEqual(eu00, 1)
        eu10 = three_node.expected_utility({'D': 1, 'S': 0})
        self.assertEqual(eu10, 0)
        eu000 = five_node.expected_utility({'D': 0, 'S1': 0, 'S2': 0})
        self.assertEqual(eu000, 2)
        eu001 = five_node.expected_utility({'D': 0, 'S1': 0, 'S2': 1})
        self.assertEqual(eu001, 1)

    def test_optimal_decision(self):
        five_node = get_5node_cid()
        opt00 = five_node._optimal_decisions('D', {'S1': 0, 'S2': 0})
        self.assertEqual(opt00.tolist(), [0])
        opt01 = five_node._optimal_decisions('D', {'S1': 0, 'S2': 1})
        self.assertEqual(opt01.tolist(), [0, 1])

    def test_possible_decision_contexts(self):
        three_node = get_3node_cid()
        five_node = get_5node_cid()
        pdc3 = three_node._possible_contexts('D')
        self.assertEqual(pdc3, [{'S': 0}, {'S': 1}])
        pdc5 = five_node._possible_contexts('D')
        self.assertEqual(pdc5, [{'S1': 0, 'S2': 0}, {'S1': 0, 'S2': 1}, {'S1': 1, 'S2': 0}, {'S1': 1, 'S2': 1}])

    def test_sufficient_recall(self):
        two_decisions = get_2dec_cid()
        self.assertEqual(two_decisions.check_sufficient_recall(), True)
        two_decisions.remove_edge('S2', 'D2')
        self.assertEqual(two_decisions.check_sufficient_recall(), False)

    def test_solve(self):
        three_node = get_3node_cid()
        soln2 = three_node.solve()
        soln2 = three_node.solve() #check that it can be solved repeatedly
        cpd2 = soln2['D']
        self.assertTrue(np.array_equal(cpd2.values, np.array([[1, 0], [0, 1]])))
        three_node.add_cpds(cpd2)
        self.assertEqual(three_node.expected_utility({}), 1)

        two_decisions = get_2dec_cid()
        soln = two_decisions.solve()
        cpd = soln['D2']
        self.assertTrue(np.array_equal(cpd.values, np.array([[1, 0], [0, 1]])))
        two_decisions.add_cpds(cpd)
        self.assertEqual(two_decisions.expected_utility({}), 1)

class TestParameterize(unittest.TestCase):
    def test_parameterization(self):
        cid = get_3node_cid()
        D = 'D'
        X = 'S'
        systems = choose_systems(cid, D, X)
        systems[0]['i_C']=get_first_c_index(cid, systems[0]['info'])
        all_cpds = parameterize_systems(cid, systems)
        all_cpds = [c.copy() for a in all_cpds for b in a.values() for c in b.values()]
        merged_cpds = merge_all_nodes(cid, all_cpds)
        cid.add_cpds(*merged_cpds.values())
        ev1, ev2 = verify_incentive(cid, D, X)
        self.assertEqual(ev1, 1)
        self.assertEqual(ev2, .5)
        
class TestParameterize(unittest.TestCase):
    def test_param3(self):
        cid = get_3node_cid()
        D = 'D'
        X = 'S'
        systems = choose_systems(cid, D, X)
        systems[0]['i_C']=get_first_c_index(cid, systems[0]['info'])
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
        systems[0]['i_C']=get_first_c_index(cid, systems[0]['info'])
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
        systems[0]['i_C']=get_first_c_index(cid, systems[0]['info'])
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
        systems[0]['i_C']=get_first_c_index(cid, systems[0]['info'])
        all_cpds = parameterize_systems(cid, systems)
        all_cpds = [c.copy() for a in all_cpds for b in a.values() for c in b.values()]
        merged_cpds = merge_all_nodes(cid, all_cpds)
        cid.add_cpds(*merged_cpds.values())
        ev1, ev2 = verify_incentive(cid, D, X)
        self.assertEqual(ev1, 3)
        self.assertEqual(ev2, 2.75)


if __name__ == '__main__':
    unittest.main()
