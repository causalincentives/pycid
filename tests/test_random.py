import sys

import networkx as nx
import pytest

from pycid import CausalBayesianNetwork, FunctionCPD, RandomCPD
from pycid.random.random_cid import random_cid, random_cids, random_macidbase
from pycid.random.random_dag import random_dag


@pytest.mark.parametrize("number_of_nodes,number_of_decisions,number_of_utilities", [(4, 1, 1), (8, 2, 2), (12, 3, 3)])
def test_random_cid_create(number_of_nodes: int, number_of_decisions: int, number_of_utilities: int) -> None:
    random_cid(number_of_nodes, number_of_decisions, number_of_utilities).check_model()


def test_cid_sufficient_recall() -> None:
    cid = random_cid(sufficient_recall=True)
    assert cid.sufficient_recall()


@pytest.mark.parametrize(
    "number_of_nodes,number_of_agents,max_decisions_for_agent,max_utilities_for_agent",
    [(10, 2, 1, 1), (14, 3, 1, 2), (18, 3, 2, 2)],
)
def test_random_macidbase(
    number_of_nodes: int, number_of_agents: int, max_decisions_for_agent: int, max_utilities_for_agent: int
) -> None:
    macid = random_macidbase(number_of_nodes, number_of_agents, max_decisions_for_agent, max_utilities_for_agent)
    macid.check_model()


def test_random_cids_create_one() -> None:
    for cid in random_cids(n_cids=1):
        cid.check_model()


def test_random_dag_create_one() -> None:
    dag = random_dag(number_of_nodes=5, edge_density=0.4, max_in_degree=4)
    assert nx.is_directed_acyclic_graph(dag)


def test_random_cpd_copy() -> None:
    """check that a copy of a random cpd yields the same distribution"""
    cbn = CausalBayesianNetwork([("A", "B")])
    cbn.add_cpds(
        RandomCPD("A"),
        FunctionCPD("B", lambda a: a),
    )
    cbn2 = cbn.copy()
    assert cbn.expected_value(["B"], {}) == cbn2.expected_value(["B"], {})


if __name__ == "__main__":
    pytest.main(sys.argv)
