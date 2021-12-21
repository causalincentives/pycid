import numpy as np
from pgmpy.factors.discrete import TabularCPD  # type: ignore

from pycid.core.macid import MACID


def get_basic_subgames() -> MACID:
    macid = MACID(
        [
            ("D11", "U11"),
            ("D11", "U2"),
            ("D11", "D12"),
            ("X1", "U11"),
            ("X1", "D11"),
            ("X1", "D2"),
            ("X1", "U3"),
            ("D2", "U2"),
            ("D2", "U3"),
            ("D2", "D3"),
            ("D3", "U2"),
            ("D3", "U3"),
            ("D12", "U3"),
            ("D12", "U22"),
            ("X2", "U22"),
            ("X2", "D12"),
        ],
        agent_decisions={
            0: ["D11", "D12"],
            1: ["D2"],
            2: ["D3"],
        },
        agent_utilities={
            0: ["U11"],
            1: ["U2", "U22"],
            2: ["U3"],
        },
    )

    return macid


def get_basic_subgames2() -> MACID:
    macid = MACID(
        [
            ("X2", "U3"),
            ("X2", "D1"),
            ("D3", "U3"),
            ("D3", "U2"),
            ("D1", "U1"),
            ("D1", "U2"),
            ("D2", "U1"),
            ("D2", "U2"),
            ("D4", "U1"),
            ("D4", "D2"),
            ("D4", "U4"),
            ("X1", "D4"),
            ("X1", "U4"),
        ],
        agent_decisions={
            1: ["D1"],
            2: ["D2"],
            3: ["D3"],
            4: ["D4"],
        },
        agent_utilities={
            1: ["U1"],
            2: ["U2"],
            3: ["U3"],
            4: ["U4"],
        },
    )

    return macid


def get_basic_subgames3() -> MACID:
    macid = MACID(
        [
            ("D4", "U4"),
            ("D2", "U4"),
            ("D3", "U4"),
            ("D2", "U2"),
            ("D3", "U3"),
            ("D1", "U2"),
            ("D1", "U3"),
            ("D1", "U1"),
        ],
        agent_decisions={
            1: ["D1"],
            2: ["D2"],
            3: ["D3"],
            4: ["D4"],
        },
        agent_utilities={
            1: ["U1"],
            2: ["U2"],
            3: ["U3"],
            4: ["U4"],
        },
    )

    return macid


def get_path_example() -> MACID:
    macid = MACID(
        [("X1", "X3"), ("X1", "D"), ("X2", "D"), ("X2", "U"), ("D", "U")],
        agent_decisions={1: ["D"]},
        agent_utilities={1: ["U"]},
    )
    return macid


def basic2agent_tie_break() -> MACID:
    macid = MACID(
        [("D1", "D2"), ("D1", "U1"), ("D1", "U2"), ("D2", "U2"), ("D2", "U1")],
        agent_decisions={0: ["D1"], 1: ["D2"]},
        agent_utilities={0: ["U1"], 1: ["U2"]},
    )

    cpd_d1 = [0, 1]
    cpd_d2 = [0, 1]
    cpd_u1 = TabularCPD(
        "U1",
        6,
        np.array([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        evidence=["D1", "D2"],
        evidence_card=[2, 2],
    )
    cpd_u2 = TabularCPD(
        "U2",
        6,
        np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]]),
        evidence=["D1", "D2"],
        evidence_card=[2, 2],
    )

    macid.add_cpds(D1=cpd_d1, D2=cpd_d2, U1=cpd_u1, U2=cpd_u2)

    return macid


def two_agent_one_pne() -> MACID:
    """This macim is a simultaneous two player game
    and has a parameterisation that
    corresponds to the following normal
    form game - where the row player is agent 1, and the
    column player is agent 2
        +----------+----------+----------+
        |          | Act(0)   | Act(1)   |
        +----------+----------+----------+
        | Act(0)   | 1, 2     | 3, 0     |
        +----------+----------+----------+
        | Act(1)   | 0, 3     | 2, 2     |
        +----------+----------+----------+
    """
    macid = MACID(
        [("D1", "U1"), ("D1", "U2"), ("D2", "U2"), ("D2", "U1")],
        agent_decisions={1: ["D1"], 2: ["D2"]},
        agent_utilities={1: ["U1"], 2: ["U2"]},
    )

    agent1_payoff = np.array([[1, 3], [0, 2]])
    agent2_payoff = np.array([[2, 0], [3, 2]])

    macid.add_cpds(
        D1=[0, 1], D2=[0, 1], U1=lambda D1, D2: agent1_payoff[D1, D2], U2=lambda D1, D2: agent2_payoff[D1, D2]
    )
    return macid


def two_agent_two_pne() -> MACID:
    """This macim is a simultaneous two player game
    and has a parameterisation that
    corresponds to the following normal
    form game - where the row player is agent 0, and the
    column player is agent 1
        +----------+----------+----------+
        |          | Act(0)   | Act(1)   |
        +----------+----------+----------+
        | Act(0)   | 1, 1     | 4, 2     |
        +----------+----------+----------+
        | Act(1)   | 2, 4     | 3, 3     |
        +----------+----------+----------+
    """
    macid = MACID(
        [("D1", "U1"), ("D1", "U2"), ("D2", "U2"), ("D2", "U1")],
        agent_decisions={0: ["D1"], 1: ["D2"]},
        agent_utilities={0: ["U1"], 1: ["U2"]},
    )

    cpd_u1 = TabularCPD(
        "U1",
        5,
        np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]]),
        evidence=["D1", "D2"],
        evidence_card=[2, 2],
    )
    cpd_u2 = TabularCPD(
        "U2",
        5,
        np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
        evidence=["D1", "D2"],
        evidence_card=[2, 2],
    )

    macid.add_cpds(cpd_u1, cpd_u2, D1=[0, 1], D2=[0, 1])
    return macid


def two_agent_no_pne() -> MACID:
    """This macim is a simultaneous two player game
    and has a parameterisation that
    corresponds to the following normal
    form game - where the row player is agent 0, and the
    column player is agent 1
        +----------+----------+----------+
        |          | Act(0)   | Act(1)   |
        +----------+----------+----------+
        | Act(0)   | 1, 0     | 0, 1     |
        +----------+----------+----------+
        | Act(1)   | 0, 1     | 1, 0     |
        +----------+----------+----------+
    """
    macid = MACID(
        [("D1", "U1"), ("D1", "U2"), ("D2", "U2"), ("D2", "U1")],
        agent_decisions={0: ["D1"], 1: ["D2"]},
        agent_utilities={0: ["U1"], 1: ["U2"]},
    )

    cpd_u1 = TabularCPD("U1", 2, np.array([[0, 1, 1, 0], [1, 0, 0, 1]]), evidence=["D1", "D2"], evidence_card=[2, 2])
    cpd_u2 = TabularCPD("U2", 2, np.array([[1, 0, 0, 1], [0, 1, 1, 0]]), evidence=["D1", "D2"], evidence_card=[2, 2])

    macid.add_cpds(cpd_u1, cpd_u2, D1=[0, 1], D2=[0, 1])
    return macid


def two_agents_three_actions() -> MACID:
    """This macim is a representation of a
    game where two players must decide between
    threee different actions simultaneously
    - the row player is agent 1 and the
    column player is agent 2 - the normal form
    representation of the payoffs is as follows:
        +----------+----------+----------+----------+
        |          |  L       |     C    |     R    |
        +----------+----------+----------+----------+
        | T        | 4, 3     | 5, 1     | 6, 2     |
        +----------+----------+----------+----------+
        | M        | 2, 1     | 8, 4     |  3, 6    |
        +----------+----------+----------+----------+
        | B        | 3, 0     | 9, 6     |  2, 8    |
        +----------+----------+----------+----------+
    - The game has one pure NE (T,L)
    """
    macid = MACID(
        [("D1", "U1"), ("D1", "U2"), ("D2", "U2"), ("D2", "U1")],
        agent_decisions={1: ["D1"], 2: ["D2"]},
        agent_utilities={1: ["U1"], 2: ["U2"]},
    )

    d1_domain = ["T", "M", "B"]
    d2_domain = ["L", "C", "R"]

    agent1_payoff = np.array([[4, 5, 6], [2, 8, 3], [3, 9, 2]])
    agent2_payoff = np.array([[3, 1, 2], [1, 4, 6], [0, 6, 8]])

    macid.add_cpds(
        D1=d1_domain,
        D2=d2_domain,
        U1=lambda D1, D2: agent1_payoff[d1_domain.index(D1), d2_domain.index(D2)],
        U2=lambda D1, D2: agent2_payoff[d1_domain.index(D1), d2_domain.index(D2)],
    )
    return macid


def basic_different_dec_cardinality() -> MACID:
    """A basic MACIM where the cardinality of each agent's decision node
    is different. It has one subgame perfect NE.
    """
    macid = MACID(
        [("D1", "D2"), ("D1", "U1"), ("D1", "U2"), ("D2", "U2"), ("D2", "U1")],
        agent_decisions={0: ["D1"], 1: ["D2"]},
        agent_utilities={0: ["U1"], 1: ["U2"]},
    )

    agent1_payoff = np.array([[3, 1, 0], [1, 2, 3]])
    agent2_payoff = np.array([[1, 2, 1], [1, 0, 3]])

    macid.add_cpds(
        D1=[0, 1], D2=[0, 1, 2], U1=lambda D1, D2: agent1_payoff[D1, D2], U2=lambda D1, D2: agent2_payoff[D1, D2]
    )

    return macid
