import sys
import unittest

import pytest

from pycid.analyze.reasoning_patterns import (
    direct_effect,
    get_reasoning_patterns,
    manipulation,
    revealing_or_denying,
    signaling,
)
from pycid.core.macid import MACID


class TestReasoning(unittest.TestCase):
    def test_direct_effect(self) -> None:
        macid = MACID(
            [("D1", "U"), ("D2", "D1")],
            agent_decisions={1: ["D1", "D2"]},
            agent_utilities={1: ["U"]},
        )
        self.assertTrue(direct_effect(macid, "D1"))
        self.assertFalse(direct_effect(macid, "D2"))
        with self.assertRaises(KeyError):
            direct_effect(macid, "D3")

    def test_manipulation(self) -> None:
        macid = MACID(
            [("D1", "U2"), ("D1", "D2"), ("D2", "U1"), ("D2", "U2")],
            agent_decisions={1: ["D1"], 2: ["D2"]},
            agent_utilities={1: ["U1"], 2: ["U2"]},
        )
        effective_set = {"D2"}  # by direct effect
        self.assertTrue(manipulation(macid, "D1", effective_set))
        self.assertFalse(manipulation(macid, "D2", effective_set))
        with self.assertRaises(KeyError):
            manipulation(macid, "D3", effective_set)
        effective_set2 = {"A"}
        with self.assertRaises(KeyError):
            manipulation(macid, "D1", effective_set2)

    def test_signaling(self) -> None:
        macid = MACID(
            [("X", "U1"), ("X", "U2"), ("X", "D1"), ("D1", "D2"), ("D2", "U1"), ("D2", "U2")],
            agent_decisions={1: ["D1"], 2: ["D2"]},
            agent_utilities={1: ["U1"], 2: ["U2"]},
        )
        effective_set = {"D2"}  # by direct effect
        self.assertTrue(signaling(macid, "D1", effective_set))
        self.assertFalse(signaling(macid, "D2", effective_set))
        with self.assertRaises(KeyError):
            signaling(macid, "D3", effective_set)
        effective_set2 = {"A"}
        with self.assertRaises(KeyError):
            signaling(macid, "D1", effective_set2)

    def test_revealing_or_denying(self) -> None:
        macid = MACID(
            [("D1", "X2"), ("X1", "X2"), ("X2", "D2"), ("D2", "U1"), ("D2", "U2"), ("X1", "U2")],
            agent_decisions={1: ["D1"], 2: ["D2"]},
            agent_utilities={1: ["U1"], 2: ["U2"]},
        )
        effective_set = {"D2"}  # by direct effect
        self.assertTrue(revealing_or_denying(macid, "D1", effective_set))
        self.assertFalse(revealing_or_denying(macid, "D2", effective_set))
        with self.assertRaises(KeyError):
            revealing_or_denying(macid, "D3", effective_set)
        effective_set2 = {"A"}
        with self.assertRaises(KeyError):
            revealing_or_denying(macid, "D1", effective_set2)

    def test_get_reasoning_patterns(self) -> None:
        macid = MACID(
            [("D1", "U"), ("D2", "D1")],
            agent_decisions={1: ["D1", "D2"]},
            agent_utilities={1: ["U"]},
        )
        self.assertEqual(get_reasoning_patterns(macid)["dir_effect"], ["D1"])

        macid2 = MACID(
            [("D1", "U2"), ("D1", "D2"), ("D2", "U1"), ("D2", "U2")],
            agent_decisions={1: ["D1"], 2: ["D2"]},
            agent_utilities={1: ["U1"], 2: ["U2"]},
        )
        self.assertEqual(get_reasoning_patterns(macid2)["dir_effect"], ["D2"])
        self.assertEqual(get_reasoning_patterns(macid2)["manip"], ["D1"])

        macid3 = MACID(
            [("X", "U1"), ("X", "U2"), ("X", "D1"), ("D1", "D2"), ("D2", "U1"), ("D2", "U2")],
            agent_decisions={1: ["D1"], 2: ["D2"]},
            agent_utilities={1: ["U1"], 2: ["U2"]},
        )
        self.assertEqual(get_reasoning_patterns(macid3)["dir_effect"], ["D2"])
        self.assertEqual(get_reasoning_patterns(macid3)["sig"], ["D1"])

        macid4 = MACID(
            [("D1", "X2"), ("X1", "X2"), ("X2", "D2"), ("D2", "U1"), ("D2", "U2"), ("X1", "U2")],
            agent_decisions={1: ["D1"], 2: ["D2"]},
            agent_utilities={1: ["U1"], 2: ["U2"]},
        )
        self.assertEqual(get_reasoning_patterns(macid4)["dir_effect"], ["D2"])
        self.assertEqual(get_reasoning_patterns(macid4)["rev_den"], ["D1"])


if __name__ == "__main__":
    pytest.main(sys.argv)
