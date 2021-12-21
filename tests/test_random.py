import sys
from typing import Tuple

import networkx as nx
import pytest

from pycid import CausalBayesianNetwork, RandomCPD
from pycid.random.random_cid import random_cid, random_cids
from pycid.random.random_dag import random_dag
from pycid.random.random_macid import random_macid, random_macids
from pycid.random.random_macidbase import random_macidbase


class TestRandomDag:
    @staticmethod
    def test_random_dag_create_one() -> None:
        dag = random_dag(number_of_nodes=5, edge_density=0.4, max_in_degree=4)
        assert nx.is_directed_acyclic_graph(dag)


class TestRandomCpd:
    @staticmethod
    def test_random_cpd_copy() -> None:
        """check that a copy of a random cpd yields the same distribution"""
        cbn = CausalBayesianNetwork([("A", "B")])
        cbn.add_cpds(A=RandomCPD(), B=lambda A: A)
        cbn2 = cbn.copy()
        assert cbn.expected_value(["B"], {}) == cbn2.expected_value(["B"], {})


class TestRandomMacidbase:
    @staticmethod
    @pytest.mark.parametrize(
        "number_of_nodes,agent_decisions_num,agent_utilities_num",
        [(10, (1, 2), (2, 1)), (16, (2, 1, 1), (1, 2, 2)), (15, (2, 1), (3, 2))],
    )
    def test_random_macidbase(
        number_of_nodes: int,
        agent_decisions_num: Tuple[int],
        agent_utilities_num: Tuple[int],
    ) -> None:
        macidbase = random_macidbase(number_of_nodes, agent_decisions_num, agent_utilities_num, add_cpds=True)
        macidbase.check_model()

    @staticmethod
    def test_unmatching_num_of_agents_specified() -> None:
        with pytest.raises(ValueError):
            random_macidbase(number_of_nodes=10, agent_decisions_num=(1, 2), agent_utilities_num=(1, 1, 1))


class TestRandomCid:
    @staticmethod
    @pytest.mark.parametrize(
        "number_of_nodes,number_of_decisions,number_of_utilities", [(4, 1, 1), (8, 2, 2), (12, 3, 3)]
    )
    def test_random_cid_create(number_of_nodes: int, number_of_decisions: int, number_of_utilities: int) -> None:
        random_cid(number_of_nodes, number_of_decisions, number_of_utilities).check_model()

    @staticmethod
    def test_cid_sufficient_recall() -> None:
        cid = random_cid(sufficient_recall=True)
        assert cid.sufficient_recall()

    @staticmethod
    def test_random_cids_create_one() -> None:
        for cid in random_cids(n_cids=1):
            cid.check_model()


class TestRandomMacid:
    @staticmethod
    @pytest.mark.parametrize(
        "number_of_nodes,agent_decisions_num,agent_utilities_num",
        [(10, (1, 3), (2, 2)), (18, (1, 1, 2), (2, 3, 2)), (14, (2, 2), (1, 4))],
    )
    def test_random_macid(
        number_of_nodes: int,
        agent_decisions_num: Tuple[int],
        agent_utilities_num: Tuple[int],
    ) -> None:
        macid = random_macid(number_of_nodes, agent_decisions_num, agent_utilities_num, add_cpds=True)
        macid.check_model()

    @staticmethod
    def test_random_macids_create_one() -> None:
        for macid in random_macids(n_macids=1):
            macid.check_model()


if __name__ == "__main__":
    pytest.main(sys.argv)
