import numpy as np
from pgmpy.factors.discrete import TabularCPD
from core.cpd import DecisionDomain, FunctionCPD
from core.macid import MACID


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

    cpd_u1 = FunctionCPD('U1', lambda d1, d2: agent1_payoff[d1_domain.index(d1), d2_domain.index(d2)],
                         evidence=['D1', 'D2'])
    cpd_u2 = FunctionCPD('U2', lambda d1, d2: agent2_payoff[d1_domain.index(d1), d2_domain.index(d2)],
                         evidence=['D1', 'D2'])

    macid.add_cpds(cpd_d1, cpd_d2, cpd_u1, cpd_u2)
    return macid


def prisoners_dilemma2() -> MACID:
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
    # TODO: Repeat of the prisoner's dilemma above except for interger
    valued decision domains becuase of the pgmpy query problem.
    """
    macid = MACID([
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1')],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']}})

    d1_domain = [0, 1]
    d2_domain = [0, 1]
    cpd_d1 = DecisionDomain('D1', d1_domain)
    cpd_d2 = DecisionDomain('D2', d2_domain)

    agent1_payoff = np.array([[-1, -3],
                             [0, -2]])
    agent2_payoff = np.transpose(agent1_payoff)

    cpd_u1 = FunctionCPD('U1', lambda d1, d2: agent1_payoff[d1_domain.index(d1), d2_domain.index(d2)],
                         evidence=['D1', 'D2'])
    cpd_u2 = FunctionCPD('U2', lambda d1, d2: agent2_payoff[d1_domain.index(d1), d2_domain.index(d2)],
                         evidence=['D1', 'D2'])

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

    cpd_u_f = FunctionCPD('U_F', lambda d_f, d_m: agent_f_payoff[d_f_domain.index(d_f), d_m_domain.index(d_m)],
                          evidence=['D_F', 'D_M'])
    cpd_u_m = FunctionCPD('U_M', lambda d_f, d_m: agent_m_payoff[d_f_domain.index(d_f), d_m_domain.index(d_m)],
                          evidence=['D_F', 'D_M'])

    macid.add_cpds(cpd_d_f, cpd_d_m, cpd_u_f, cpd_u_m)
    return macid


def matching_pennies() -> MACID:
    """ This macim is a representation of the
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

    cpd_u1 = FunctionCPD('U1', lambda d1, d2: agent1_payoff[d1_domain.index(d1), d2_domain.index(d2)],
                         evidence=['D1', 'D2'])
    cpd_u2 = FunctionCPD('U2', lambda d1, d2: agent2_payoff[d1_domain.index(d1), d2_domain.index(d2)],
                         evidence=['D1', 'D2'])

    macid.add_cpds(cpd_d1, cpd_d2, cpd_u1, cpd_u2)
    return macid


def taxi_competition() -> MACID:
    """ A MACIM for the "Taxi Competition" example
    introduced in "Equilibrium Refinements for Multi-Agent
    Influence Diagrams: Theory and Practice" by Hammond, Fox,
    Everitt, Abate & Wooldridge, 2021:

                              D1
        +----------+----------+----------+
        |  taxi 1  | expensive|  cheap   |
        +----------+----------+----------+
        |expensive |     2    |   3      |
    D2  +----------+----------+----------+
        | cheap    |     5    |   1      |
        +----------+----------+----------+

                              D1
        +----------+----------+----------+
        |  taxi 2  | expensive|  cheap   |
        +----------+----------+----------+
        |expensive |     2    |   5      |
    D2  +----------+----------+----------+
        | cheap    |     3    |   1      |
        +----------+----------+----------+

    - There are 3 pure startegy NE and 1 pure SPE.
    """
    macid = MACID([
        ('D1', 'D2'),
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1')],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']}})

    d1_domain = ['e', 'c']
    d2_domain = ['e', 'c']
    cpd_d1 = DecisionDomain('D1', d1_domain)
    cpd_d2 = DecisionDomain('D2', d2_domain)

    agent1_payoff = np.array([[2, 3],
                             [5, 1]])
    agent2_payoff = np.array([[2, 5],
                             [3, 1]])

    cpd_u1 = FunctionCPD('U1', lambda d1, d2: agent1_payoff[d2_domain.index(d2), d1_domain.index(d1)],
                         evidence=['D1', 'D2'])
    cpd_u2 = FunctionCPD('U2', lambda d1, d2: agent2_payoff[d2_domain.index(d2), d1_domain.index(d1)],
                         evidence=['D1', 'D2'])

    macid.add_cpds(cpd_d1, cpd_d2, cpd_u1, cpd_u2)
    return macid


def modified_taxi_competition() -> MACID:
    """ Modifying the payoffs in the taxi competition example
    so that there is a tie break (if taxi 1 chooses to stop
    in front of the expensive hotel, taxi 2 is indifferent
    between their choices.)

    - There are now two SPNE

                              D1
        +----------+----------+----------+
        |  taxi 1  | expensive|  cheap   |
        +----------+----------+----------+
        |expensive |     2    |   3      |
    D2  +----------+----------+----------+
        | cheap    |     5    |   1      |
        +----------+----------+----------+

                              D1
        +----------+----------+----------+
        |  taxi 2  | expensive|  cheap   |
        +----------+----------+----------+
        |expensive |     2    |   5      |
    D2  +----------+----------+----------+
        | cheap    |     3    |   5      |
        +----------+----------+----------+

    """
    macid = MACID([
        ('D1', 'D2'),
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1')],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']}})

    d1_domain = ['e', 'c']
    d2_domain = ['e', 'c']
    cpd_d1 = DecisionDomain('D1', d1_domain)
    cpd_d2 = DecisionDomain('D2', d2_domain)

    agent1_payoff = np.array([[2, 3],
                             [5, 1]])
    agent2_payoff = np.array([[2, 5],
                             [3, 5]])

    cpd_u1 = FunctionCPD('U1', lambda d1, d2: agent1_payoff[d2_domain.index(d2), d1_domain.index(d1)],
                         evidence=['D1', 'D2'])
    cpd_u2 = FunctionCPD('U2', lambda d1, d2: agent2_payoff[d2_domain.index(d2), d1_domain.index(d1)],
                         evidence=['D1', 'D2'])

    macid.add_cpds(cpd_d1, cpd_d2, cpd_u1, cpd_u2)
    return macid



def tree_doctor() -> MACID:
    macid = MACID([
        ('PT', 'E'),
        ('PT', 'TS'),
        ('PT', 'BP'),
        ('TS', 'TDoc'),
        ('TS', 'TDead'),
        ('TDead', 'V'),
        ('TDead', 'Tree'),
        ('TDoc', 'TDead'),
        ('TDoc', 'Cost'),
        ('TDoc', 'BP'),
        ('BP', 'V')],
        {0: {'D': ['PT', 'BP'], 'U': ['E', 'V']},
         1: {'D': ['TDoc'], 'U': ['Tree', 'Cost']}})

    return macid


def forgetful_movie_star() -> MACID:
    macid = MACID([
        ('S', 'D11'),
        ('S', 'D12'),
        ('D2', 'U2'),
        ('D2', 'U11'),
        ('D11', 'U2'),
        ('D11', 'U11'),
        ('D11', 'U12'),
        ('D12', 'U12')],
        {1: {'D': ['D11', 'D12'], 'U': ['U11', 'U12']},
         2: {'D': ['D2'], 'U': ['U2']}})
    return macid


def subgame_difference() -> MACID:
    macid = MACID([
        ('N', 'D1'),
        ('N', 'U1_A'),
        ('N', 'U2_A'),
        ('D1', 'U1_A'),
        ('D1', 'U2_A'),
        ('D1', 'U1_B'),
        ('D1', 'U2_B'),
        ('D1', 'D2'),
        ('D2', 'U1_B'),
        ('D2', 'U2_B')],
        {1: {'D': ['D1'], 'U': ['U1_A', 'U1_B']},
         2: {'D': ['D2'], 'U': ['U2_A', 'U2_B']}})
    return macid


def road_example() -> MACID:
    macid = MACID([
        ('S1W', 'B1W'),
        ('S1W', 'U1W'),
        ('S1E', 'B1E'),
        ('S1E', 'U1E'),

        ('B1W', 'U1W'),
        ('B1W', 'U1E'),
        ('B1W', 'B2E'),
        ('B1W', 'U2W'),
        ('B1W', 'B2W'),

        ('B1E', 'U1E'),
        ('B1E', 'U1W'),
        ('B1E', 'B2E'),
        ('B1E', 'U2E'),
        ('B1E', 'B2W'),

        ('S2W', 'B2W'),
        ('S2W', 'U2W'),
        ('S2E', 'B2E'),
        ('S2E', 'U2E'),

        ('B2W', 'U1W'),
        ('B2W', 'U2W'),
        ('B2W', 'U2E'),
        ('B2W', 'B3E'),
        ('B2W', 'U3W'),
        ('B2W', 'B3W'),

        ('B2E', 'U1E'),
        ('B2E', 'U2E'),
        ('B2E', 'U2W'),
        ('B2E', 'B3E'),
        ('B2E', 'U3E'),
        ('B2E', 'B3W'),

        ('S3W', 'B3W'),
        ('S3W', 'U3W'),
        ('S3E', 'B3E'),
        ('S3E', 'U3E'),

        ('B3W', 'U3W'),
        ('B3W', 'U3E'),
        ('B3W', 'U2W'),

        ('B3E', 'U3E'),
        ('B3E', 'U3W'),
        ('B3E', 'U2E')],
        {'1W': {'D': ['B1W'], 'U': ['U1W']},
         '1E': {'D': ['B1E'], 'U': ['U1E']},
         '2W': {'D': ['B2W'], 'U': ['U2W']},
         '2E': {'D': ['B2E'], 'U': ['U2E']},
         '3W': {'D': ['B3W'], 'U': ['U3W']},
         '3E': {'D': ['B3E'], 'U': ['U3E']}})

    return macid


def politician() -> MACID:
    macid = MACID([
        ('D1', 'I'),
        ('T', 'I'),
        ('T', 'U2'),
        ('I', 'D2'),
        ('R', 'D2'),
        ('D2', 'U1'),
        ('D2', 'U2')],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']}})
    return macid


def umbrella() -> MACID:
    macid = MACID([
        ('W', 'F'),
        ('W', 'A'),
        ('F', 'UM'),
        ('UM', 'A')],
        {1: {'D': ['UM'], 'U': ['A']}})

    cpd_w = TabularCPD('W', 2, np.array([[.6], [.4]]))
    cpd_f = TabularCPD('F', 2, np.array([[.8, .3], [.2, .7]]),
                       evidence=['W'], evidence_card=[2])
    cpd_um = DecisionDomain('UM', [0, 1])
    cpd_a = TabularCPD('A', 3, np.array([[0, 1, 1, 0],
                                         [1, 0, 0, 0],
                                         [0, 0, 0, 1]]),
                       evidence=['W', 'UM'], evidence_card=[2, 2])
    macid.add_cpds(cpd_w, cpd_f, cpd_um, cpd_a)
    return macid


def sequential() -> MACID:
    macid = MACID([
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D1', 'D2'),
        ('D2', 'U1'),
        ('D2', 'U2')],
        {0: {'D': ['D1'], 'U': ['U1']},
         1: {'D': ['D2'], 'U': ['U2']}})
    return macid


def signal() -> MACID:
    macid = MACID([
        ('X', 'D1'),
        ('X', 'U2'),
        ('X', 'U1'),
        ('D1', 'U2'),
        ('D1', 'U1'),
        ('D1', 'D2'),
        ('D2', 'U1'),
        ('D2', 'U2')],
        {0: {'D': ['D1'], 'U': ['U1']},
         1: {'D': ['D2'], 'U': ['U2']}})
    cpd_x = TabularCPD('X', 2, np.array([[.5], [.5]]))
    cpd_d1 = DecisionDomain('D1', [0, 1])
    cpd_d2 = DecisionDomain('D1', [0, 1])

    u1_cpd_array = np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 1, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0]])

    u2_cpd_array = np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 1, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0]])

    cpd_u1 = TabularCPD('U1', 6, u1_cpd_array, evidence=['X', 'D1', 'D2'], evidence_card=[2, 2, 2])
    cpd_u2 = TabularCPD('U2', 6, u2_cpd_array, evidence=['X', 'D1', 'D2'], evidence_card=[2, 2, 2])

    macid.add_cpds(cpd_x, cpd_d1, cpd_d2, cpd_u1, cpd_u2)

    return macid


def triage() -> MACID:
    macid = MACID([

        ('H1', 'D1'),
        ('H1', 'U1'),

        ('H2', 'D2'),
        ('H2', 'U2'),

        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D1', 'D3'),
        ('D1', 'D4'),
        ('D1', 'U3'),
        ('D1', 'U4'),

        ('D2', 'U1'),
        ('D2', 'U2'),
        ('D2', 'D4'),
        ('D2', 'D3'),
        ('D2', 'U3'),
        ('D2', 'U4'),

        ('H3', 'D3'),
        ('H3', 'U3'),

        ('H4', 'D4'),
        ('H4', 'U4'),

        ('D3', 'U3'),
        ('D3', 'U4'),
        ('D3', 'U1'),
        ('D3', 'U2'),
        ('D4', 'U3'),
        ('D4', 'U4'),
        ('D4', 'U1'),
        ('D4', 'U2'),


        ('D3', 'U5'),
        ('D3', 'U6'),
        ('D4', 'U5'),
        ('D4', 'U6'),

        ('D1', 'U5'),
        ('D1', 'U6'),
        ('D2', 'U5'),
        ('D2', 'U6'),

        ('H5', 'D5'),
        ('H5', 'U5'),

        ('H6', 'D6'),
        ('H6', 'U6'),

        ('D1', 'D5'),
        ('D1', 'D6'),
        ('D2', 'D5'),
        ('D2', 'D6'),

        ('D3', 'D5'),
        ('D3', 'D6'),
        ('D4', 'D5'),
        ('D4', 'D6'),


        ('D5', 'U3'),
        ('D5', 'U4'),
        ('D5', 'U1'),
        ('D5', 'U2'),
        ('D5', 'U5'),
        ('D5', 'U6'),
        ('D6', 'U3'),
        ('D6', 'U4'),
        ('D6', 'U1'),
        ('D6', 'U2'),
        ('D6', 'U5'),
        ('D6', 'U6')],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']},
         3: {'D': ['D3'], 'U': ['U3']},
         4: {'D': ['D4'], 'U': ['U4']},
         5: {'D': ['D5'], 'U': ['U5']},
         6: {'D': ['D6'], 'U': ['U6']}})

    return macid
