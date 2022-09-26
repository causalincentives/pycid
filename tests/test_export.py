import sys
import unittest

import pytest

from pycid.examples.story_macids import taxi_competition
from pycid.export.gambit import behavior_to_cpd, macid_to_efg, macid_to_gambit_file


class TestExport(unittest.TestCase):
    # @unittest.skip("")
    def test_macid_to_efg(self) -> None:
        macid = taxi_competition()
        game, _ = macid_to_efg(macid)
        self.assertEqual(len(game.players), 2)
        self.assertEqual(len(game.players[0].strategies), 2)
        self.assertEqual(len(game.players[1].strategies), 4)
        self.assertEqual(len(game.actions), 6)
        self.assertEqual(len(game.outcomes), 4)
        self.assertEqual(len(game.infosets), 3)

    # @unittest.skip("")
    def test_macid_to_gambit_file(self) -> None:
        macid = taxi_competition()
        self.assertTrue(macid_to_gambit_file(macid, "taxi_competition.efg"))

    def test_behavior_to_cpd(self) -> None:
        # pygambit NE solver
        macid = taxi_competition()
        game, parents_to_infoset = macid_to_efg(macid)
        ne_behavior_strategies = macid._pygambit_ne_solver(game)
        ne = [behavior_to_cpd(macid, parents_to_infoset, strat) for strat in ne_behavior_strategies]
        self.assertEqual(len(ne), 4)
        self.assertEqual(ne[0][0].domain, ["e", "c"])
        self.assertEqual(ne[0][0].domain, ["e", "c"])


if __name__ == "__main__":
    pytest.main(sys.argv)
