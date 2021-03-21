import sys
import unittest

import numpy as np
import pytest
from pgmpy.factors.discrete import TabularCPD  # type: ignore

from pycid.examples.simple_cbns import get_3node_cbn
from pycid.examples.simple_cids import get_3node_cid, get_minimal_cid
from pycid.examples.story_macids import taxi_competition


class TestCBN(unittest.TestCase):
    # @unittest.skip("")
    def test_remove_add_edge(self) -> None:
        cid = get_3node_cid()
        cid.remove_edge("S", "D")
        self.assertTrue(cid.check_model())
        cid.add_edge("S", "D")
        self.assertTrue(cid.check_model())

    # @unittest.skip("")
    def test_assign_cpd(self) -> None:
        three_node = get_3node_cbn()
        three_node.add_cpds(TabularCPD("D", 2, np.eye(2), evidence=["S"], evidence_card=[2]))
        three_node.check_model()
        cpd = three_node.get_cpds("D").values
        self.assertTrue(np.array_equal(cpd, np.array([[1, 0], [0, 1]])))

    # @unittest.skip("")
    def test_query(self) -> None:
        three_node = get_3node_cbn()
        self.assertTrue(three_node.query(["U"], {"D": 2}).values[2] == float(1.0))
        # contexts need be within the domain of the variable
        with self.assertRaises(ValueError):
            three_node.query(["U"], {"S": 0})

    # @unittest.skip("")
    def test_intervention(self) -> None:
        cid = get_minimal_cid()
        cid.impute_random_policy()
        self.assertEqual(cid.expected_value(["B"], {})[0], 0.5)
        for a in [0, 1]:
            cid.intervene({"A": a})
            self.assertEqual(cid.expected_value(["B"], {})[0], a)
        self.assertEqual(cid.expected_value(["B"], {}, intervention={"A": 1})[0], 1)
        macid = taxi_competition()
        macid.impute_fully_mixed_policy_profile()
        self.assertEqual(macid.expected_value(["U1"], {}, intervention={"D1": "c", "D2": "e"})[0], 3)
        self.assertEqual(macid.expected_value(["U2"], {}, intervention={"D1": "c", "D2": "e"})[0], 5)

    # @unittest.skip("")
    def test_copy_without_cpds(self) -> None:
        cbn = get_3node_cbn()
        cbn_no_cpds = cbn.copy_without_cpds()
        self.assertTrue(len(cbn_no_cpds.cpds) == 0)


if __name__ == "__main__":
    pytest.main(sys.argv)
