import numpy as np
from pgmpy.factors.discrete import TabularCPD  # type: ignore
from core.macid import MACID
from core.cpd import DecisionDomain, FunctionCPD


def get_basic2agent_acyclic() -> MACID:
    macid = MACID([
        ('D1', 'D2'),
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1')],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']}})
    return macid


def get_basic2agent_cyclic() -> MACID:
    macid = MACID([
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1')],

        {0: {'D': ['D1'], 'U': ['U1']},
         1: {'D': ['D2'], 'U': ['U2']}})

    return macid


def get_basic_subgames() -> MACID:
    macid = MACID([
        ('D11', 'U11'),
        ('D11', 'U2'),
        ('D11', 'D12'),
        ('X1', 'U11'),
        ('X1', 'D11'),
        ('X1', 'D2'),
        ('X1', 'U3'),
        ('D2', 'U2'),
        ('D2', 'U3'),
        ('D2', 'D3'),
        ('D3', 'U2'),
        ('D3', 'U3'),
        ('D12', 'U3'),
        ('D12', 'U22'),
        ('X2', 'U22'),
        ('X2', 'D12')],
        {0: {'D': ['D11', 'D12'], 'U': ['U11']},
         1: {'D': ['D2'], 'U': ['U2', 'U22']},
         2: {'D': ['D3'], 'U': ['U3']}})

    return macid


def get_basic_subgames2() -> MACID:
    macid = MACID([
        ('X2', 'U3'),
        ('X2', 'D1'),
        ('D3', 'U3'),
        ('D3', 'U2'),
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U1'),
        ('D2', 'U2'),
        ('D4', 'U1'),
        ('D4', 'D2'),
        ('D4', 'U4'),
        ('X1', 'D4'),
        ('X1', 'U4')],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']},
         3: {'D': ['D3'], 'U': ['U3']},
         4: {'D': ['D4'], 'U': ['U4']}})

    return macid


def get_basic_subgames3() -> MACID:
    macid = MACID([
        ('D4', 'U4'),
        ('D2', 'U4'),
        ('D3', 'U4'),
        ('D2', 'U2'),
        ('D3', 'U3'),
        ('D1', 'U2'),
        ('D1', 'U3'),
        ('D1', 'U1')],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']},
         3: {'D': ['D3'], 'U': ['U3']},
         4: {'D': ['D4'], 'U': ['U4']},

         })

    return macid


def get_path_example() -> MACID:
    macid = MACID([
        ('X1', 'X3'),
        ('X1', 'D'),
        ('X2', 'D'),
        ('X2', 'U'),
        ('D', 'U')],
        {1: {'D': ['D'], 'U': ['U']}})
    return macid


def basic2agent_tie_break() -> MACID:
    macid = MACID([
        ('D1', 'D2'),
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1')],
        {0: {'D': ['D1'], 'U': ['U1']},
         1: {'D': ['D2'], 'U': ['U2']}})

    cpd_d1 = DecisionDomain('D1', [0, 1])
    cpd_d2 = DecisionDomain('D2', [0, 1])
    cpd_u1 = TabularCPD('U1', 6, np.array([[0, 1, 0, 0],
                                           [0, 0, 0, 1],
                                           [0, 0, 0, 0],
                                           [1, 0, 1, 0],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0]]),
                        evidence=['D1', 'D2'], evidence_card=[2, 2])
    cpd_u2 = TabularCPD('U2', 6, np.array([[0, 0, 0, 0],
                                           [1, 0, 0, 0],
                                           [0, 0, 1, 1],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0],
                                           [0, 1, 0, 0]]),
                        evidence=['D1', 'D2'], evidence_card=[2, 2])

    macid.add_cpds(cpd_d1, cpd_d2, cpd_u1, cpd_u2)

    return macid


def basic2agent() -> MACID:
    macid = MACID([
        ('D1', 'D2'),
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1')],
        {0: {'D': ['D1'], 'U': ['U1']},
         1: {'D': ['D2'], 'U': ['U2']}})

    cpd_d1 = DecisionDomain('D1', [0, 1])
    cpd_d2 = DecisionDomain('D2', [0, 1])
    cpd_u1 = TabularCPD('U1', 6, np.array([[0, 1, 0, 0],
                                           [0, 0, 0, 1],
                                           [0, 0, 0, 0],
                                           [1, 0, 1, 0],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0]]),
                        evidence=['D1', 'D2'], evidence_card=[2, 2])
    cpd_u2 = TabularCPD('U2', 6, np.array([[0, 0, 0, 1],
                                           [1, 0, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0],
                                           [0, 1, 0, 0]]),
                        evidence=['D1', 'D2'], evidence_card=[2, 2])

    macid.add_cpds(cpd_d1, cpd_d2, cpd_u1, cpd_u2)

    return macid


def basic2agent_3() -> MACID:
    macid = MACID([
        ('D1', 'D2'),                   # KM_NE should = {'D1': 1, 'D2': 0, 'D3': 1}
        ('D1', 'D3'),
        ('D2', 'D3'),
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D1', 'U3'),
        ('D2', 'U1'),
        ('D2', 'U2'),
        ('D2', 'U3'),
        ('D3', 'U1'),
        ('D3', 'U2'),
        ('D3', 'U3')],
        {0: {'D': ['D1'], 'U': ['U1']},
         1: {'D': ['D2'], 'U': ['U2']},
         2: {'D': ['D3'], 'U': ['U3']}})

    cpd_d1 = DecisionDomain('D1', [0, 1])
    cpd_d2 = DecisionDomain('D2', [0, 1])
    cpd_d3 = DecisionDomain('D3', [0, 1])

    cpd_u1 = TabularCPD('U1', 7, np.array([[0, 0, 0, 1, 0, 0, 0, 1],
                                           [1, 1, 0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 1, 0, 0],
                                           [0, 0, 1, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 1, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0]]),
                        evidence=['D1', 'D2', 'D3'], evidence_card=[2, 2, 2])
    cpd_u2 = TabularCPD('U2', 7, np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 1, 0, 1, 0],
                                           [0, 1, 1, 1, 0, 0, 0, 0],
                                           [1, 0, 0, 0, 0, 0, 0, 1],
                                           [0, 0, 0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0]]),
                        evidence=['D1', 'D2', 'D3'], evidence_card=[2, 2, 2])
    cpd_u3 = TabularCPD('U3', 7, np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                                           [0, 0, 1, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 1, 0, 0, 0],
                                           [0, 1, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 1, 0, 0, 0, 0],
                                           [1, 0, 0, 0, 0, 1, 0, 0]]),
                        evidence=['D1', 'D2', 'D3'], evidence_card=[2, 2, 2])

    macid.add_cpds(cpd_d1, cpd_d2, cpd_d3, cpd_u1, cpd_u2, cpd_u3)

    return macid


def two_agent_one_pne() -> MACID:
    """ This macim is a simultaneous two player game
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
    macid = MACID([
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1')],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']}})

    d1_domain = ['a', 'b']
    d2_domain = ['a', 'b']
    cpd_d1 = DecisionDomain('D1', d1_domain)
    cpd_d2 = DecisionDomain('D2', d2_domain)

    agent1_payoff = np.array([[1, 3],
                             [0, 2]])
    agent2_payoff = np.array([[2, 0],
                             [3, 2]])

    cpd_u1 = FunctionCPD('U1', lambda d1, d2: agent1_payoff[d1_domain.index(d1), d2_domain.index(d2)], evidence=['D1', 'D2'])
    cpd_u2 = FunctionCPD('U2', lambda d1, d2: agent2_payoff[d1_domain.index(d1), d2_domain.index(d2)], evidence=['D1', 'D2'])

    macid.add_cpds(cpd_d1, cpd_d2, cpd_u1, cpd_u2)
    return macid


def two_agent_two_pne() -> MACID:
    """ This macim is a simultaneous two player game
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
    macid = MACID([
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1')],
        {0: {'D': ['D1'], 'U': ['U1']},
         1: {'D': ['D2'], 'U': ['U2']}})

    cpd_d1 = DecisionDomain('D1', [0, 1])
    cpd_d2 = DecisionDomain('D2', [0, 1])

    cpd_u1 = TabularCPD('U1', 5, np.array([[0, 0, 0, 0],
                                          [1, 0, 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1],
                                          [0, 1, 0, 0]]),
                        evidence=['D1', 'D2'], evidence_card=[2, 2])
    cpd_u2 = TabularCPD('U2', 5, np.array([[0, 0, 0, 0],
                                          [1, 0, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 0, 0, 1],
                                          [0, 0, 1, 0]]),
                        evidence=['D1', 'D2'], evidence_card=[2, 2])

    macid.add_cpds(cpd_d1, cpd_d2, cpd_u1, cpd_u2)
    return macid


def two_agent_no_pne() -> MACID:
    """ This macim is a simultaneous two player game
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
    macid = MACID([
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1')],
        {0: {'D': ['D1'], 'U': ['U1']},
         1: {'D': ['D2'], 'U': ['U2']}})

    cpd_d1 = DecisionDomain('D1', [0, 1])
    cpd_d2 = DecisionDomain('D2', [0, 1])

    cpd_u1 = TabularCPD('U1', 2, np.array([[0, 1, 1, 0],
                                          [1, 0, 0, 1]]),
                        evidence=['D1', 'D2'], evidence_card=[2, 2])
    cpd_u2 = TabularCPD('U2', 2, np.array([[1, 0, 0, 1],
                                          [0, 1, 1, 0]]),
                        evidence=['D1', 'D2'], evidence_card=[2, 2])

    macid.add_cpds(cpd_d1, cpd_d2, cpd_u1, cpd_u2)
    return macid


def prisoners_dilemma() -> MACID:
    """ This macim is a representation of the canonical
    prisoner's dilemma. It is a simultaneous
    symmetric two-player game with payoffs
    corresponding to the following normal
    form game - the row player is agent 1 and the
    column player is agent 2:
        +----------+----------+----------+
        |          |Cooperate | Defect   |
        +----------+----------+----------+
        |Cooperate | -1, -1   | -3, 0    |
        +----------+----------+----------+
        |  Defect  | 0, -3    | -2, -2   |
        +----------+----------+----------+
    - This game has one pure NE: (defect, defect)
    """
    macid = MACID([
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1')],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']}})

    d1_domain = ['c', 'd']
    d2_domain = ['c', 'd']
    cpd_d1 = DecisionDomain('D1', d1_domain)
    cpd_d2 = DecisionDomain('D2', d2_domain)

    agent1_payoff = np.array([[-1, -3],
                             [0, -2]])
    agent2_payoff = np.transpose(agent1_payoff)

    cpd_u1 = FunctionCPD('U1', lambda d1, d2: agent1_payoff[d1_domain.index(d1), d2_domain.index(d2)], evidence=['D1', 'D2'])
    cpd_u2 = FunctionCPD('U2', lambda d1, d2: agent2_payoff[d1_domain.index(d1), d2_domain.index(d2)], evidence=['D1', 'D2'])

    macid.add_cpds(cpd_d1, cpd_d2, cpd_u1, cpd_u2)
    return macid


def battle_of_the_sexes() -> MACID:
    """ This macim is a representation of the
    battle of the sexes game (also known as Bach or Stravinsky). 
    It is a simultaneous symmetric two-player game with payoffs
    corresponding to the following normal
    form game - the row player is Female and the
    column player is Male:
        +----------+----------+----------+
        |          |Opera     | Football |
        +----------+----------+----------+
        |  Opera   | 3, 2     |   0, 0   |
        +----------+----------+----------+
        | Football | 0, 0     | 2, 3     |
        +----------+----------+----------+
    - This game has two pure NE: (Opera, Football) and (Football, Opera)
    """
    macid = MACID([
        ('D_F', 'U_F'),
        ('D_F', 'U_M'),
        ('D_M', 'U_M'),
        ('D_M', 'U_F')],
        {'M': {'D': ['D_F'], 'U': ['U_F']},
         'F': {'D': ['D_M'], 'U': ['U_M']}})

    d_f_domain = ['O', 'F']
    d_m_domain = ['O', 'F']
    cpd_d_f = DecisionDomain('D_F', d_f_domain)
    cpd_d_m = DecisionDomain('D_M', d_m_domain)

    agent_f_payoff = np.array([[3, 0],
                              [0, 2]])
    agent_m_payoff = np.array([[2, 0],
                              [0, 3]])

    cpd_u_f = FunctionCPD('U_F', lambda d_f, d_m: agent_f_payoff[d_f_domain.index(d_f), d_m_domain.index(d_m)], evidence=['D_F', 'D_M'])
    cpd_u_m = FunctionCPD('U_M', lambda d_f, d_m: agent_m_payoff[d_f_domain.index(d_f), d_m_domain.index(d_m)], evidence=['D_F', 'D_M'])

    macid.add_cpds(cpd_d_f, cpd_d_m, cpd_u_f, cpd_u_m)
    return macid


def matching_pennies() -> MACID:
    """ This macim is a represetnation of the
    matching pennies game.
    It is symmetric two-player game with payoffs
    corresponding to the following normal
    form game - the row player is agent 1 and the
    column player is agent 2:
        +----------+----------+----------+
        |          |Heads     | Tails    |
        +----------+----------+----------+
        |  Heads   | +1, -1   | -1, +1   |
        +----------+----------+----------+
        | Tails    | -1, +1   | +1, -1   |
        +----------+----------+----------+
    - This game has no pure NE, but has a mixed NE where
    each player chooses Heads or Tails with equal probability.
    """
    macid = MACID([
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1')],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']}})

    d1_domain = ['H', 'T']
    d2_domain = ['H', 'T']
    cpd_d1 = DecisionDomain('D1', d1_domain)
    cpd_d2 = DecisionDomain('D2', d2_domain)

    agent1_payoff = np.array([[1, -1],
                             [-1, 1]])
    agent2_payoff = np.array([[-1, 1],
                             [1, -1]])

    cpd_u1 = FunctionCPD('U1', lambda d1, d2: agent1_payoff[d1_domain.index(d1), d2_domain.index(d2)], evidence=['D1', 'D2'])
    cpd_u2 = FunctionCPD('U2', lambda d1, d2: agent2_payoff[d1_domain.index(d1), d2_domain.index(d2)], evidence=['D1', 'D2'])

    macid.add_cpds(cpd_d1, cpd_d2, cpd_u1, cpd_u2)
    return macid


def two_agents_three_actions() -> MACID:
    """ This macim is a represetnation of a
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
    macid = MACID([
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1')],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']}})

    d1_domain = ['T', 'M', 'B']
    d2_domain = ['L', 'C', 'R']
    cpd_d1 = DecisionDomain('D1', d1_domain)
    cpd_d2 = DecisionDomain('D2', d2_domain)

    agent1_payoff = np.array([[4, 5, 6],
                             [2, 8, 3],
                             [3, 9, 2]])
    agent2_payoff = np.array([[3, 1, 2],
                             [1, 4, 6],
                             [0, 6, 8]])

    cpd_u1 = FunctionCPD('U1', lambda d1, d2: agent1_payoff[d1_domain.index(d1), d2_domain.index(d2)], evidence=['D1', 'D2'])
    cpd_u2 = FunctionCPD('U2', lambda d1, d2: agent2_payoff[d1_domain.index(d1), d2_domain.index(d2)], evidence=['D1', 'D2'])

    macid.add_cpds(cpd_d1, cpd_d2, cpd_u1, cpd_u2)
    return macid













def c2d() -> MACID:
    macid = MACID([
        ('C1', 'U1'),
        ('C1', 'U2'),
        ('C1', 'D1'),
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D1', 'D2'),
        ('D2', 'U1'),
        ('D2', 'U2'),
        ('C1', 'D2')],
        {0: {'D': ['D1'], 'U': ['U1']},
         1: {'D': ['D2'], 'U': ['U2']}})

    cpd_c1 = TabularCPD('C1', 2, np.array([[.5], [.5]]))
    cpd_d1 = DecisionDomain('D1', [0, 1])
    cpd_d2 = DecisionDomain('D2', [0, 1])
    cpd_u1 = TabularCPD('U1', 4, np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                                           [1, 0, 1, 0, 0, 1, 0, 0],
                                           [0, 1, 0, 1, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 1]]),
                        evidence=['C1', 'D1', 'D2'], evidence_card=[2, 2, 2])
    cpd_u2 = TabularCPD('U2', 6, np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                           [1, 0, 0, 0, 0, 0, 1, 0],
                                           [0, 1, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 1, 0, 0, 1, 0, 1],
                                           [0, 0, 0, 1, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 1, 0, 0, 0]]),
                        evidence=['C1', 'D1', 'D2'], evidence_card=[2, 2, 2])
    macid.add_cpds(cpd_c1, cpd_d1, cpd_d2, cpd_u1, cpd_u2)
    return macid


def basic_different_dec_cardinality() -> MACID:
    macid = MACID([
        ('D1', 'D2'),
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1')],
        {0: {'D': ['D1'], 'U': ['U1']},
         1: {'D': ['D2'], 'U': ['U2']}})

    cpd_d1 = DecisionDomain('D1', [0, 1])
    cpd_d2 = DecisionDomain('D2', [0, 1, 2])

    cpd_u1 = TabularCPD('U1', 4, np.array([[0, 0, 1, 0, 0, 0],
                                           [0, 1, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 1, 0],
                                           [1, 0, 0, 0, 0, 1]]),
                        evidence=['D1', 'D2'], evidence_card=[2, 3])
    cpd_u2 = TabularCPD('U2', 4, np.array([[0, 0, 0, 0, 1, 0],
                                           [1, 0, 1, 1, 0, 0],
                                           [0, 1, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 1]]),
                        evidence=['D1', 'D2'], evidence_card=[2, 3])

    macid.add_cpds(cpd_d1, cpd_d2, cpd_u1, cpd_u2)

    return macid
