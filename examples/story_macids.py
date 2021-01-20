import numpy as np
from pgmpy.factors.discrete import TabularCPD

from macid import MACID
from cpd import DecisionDomain


def tree_doctor():
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
        ('BP', 'V'),
        ],
        {0: {'D': ['PT', 'BP'], 'U': ['E', 'V']}, 1: {'D': ['TDoc'], 'U': ['Tree', 'Cost']}, 'C': ['TS', 'TDead']})

    return macid


def road_example():
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
        ('B3E', 'U2E'),

        ],
        {'1W': {'D': ['B1W'], 'U': ['U1W']}, '1E': {'D': ['B1E'], 'U': ['U1E']},
         '2W': {'D': ['B2W'], 'U': ['U2W']}, '2E': {'D': ['B2E'], 'U': ['U2E']},
         '3W': {'D': ['B3W'], 'U': ['U3W']}, '3E': {'D': ['B3E'], 'U': ['U3E']},
         'C': ['S1W', 'S1E', 'S2W', 'S2E', 'S3W', 'S3E']})

    return macid


def politician():
    macid = MACID([
        ('D1', 'I'),
        ('T', 'I'),
        ('T', 'U2'),
        ('I', 'D2'),
        ('R', 'D2'),
        ('D2', 'U1'),
        ('D2', 'U2'),
        ],
        {1: {'D': ['D1'], 'U': ['U1']}, 2: {'D': ['D2'], 'U': ['U2']}, 'C': ['R', 'I', 'T']})
    return macid


def umbrella():
    from pgmpy.factors.discrete.CPD import TabularCPD
    macid = MACID([
        ('W', 'F'),
        ('W', 'A'),
        ('F', 'UM'),
        ('UM', 'A'),
        ],
        {1: {'D': ['UM'], 'U': ['A']}, 'C': ['W', 'F']})

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


def sequential():
    macid = MACID([
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D1', 'D2'),
        ('D2', 'U1'),
        ('D2', 'U2'),
        ],
        {0: {'D': ['D1'], 'U': ['U1']}, 1: {'D': ['D2'], 'U': ['U2']}, 'C': []})
    return macid


def signal():
    macid = MACID([
        ('X', 'D1'),
        ('X', 'U2'),
        ('X', 'U1'),
        ('D1', 'U2'),
        ('D1', 'U1'),
        ('D1', 'D2'),
        ('D2', 'U1'),
        ('D2', 'U2'),
        ],
        {0: {'D': ['D1'], 'U': ['U1']}, 1: {'D': ['D2'], 'U': ['U2']}, 'C': ['X']})
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


def triage():
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
        ('D6', 'U6'),


        ],
        {1: {'D': ['D1'], 'U': ['U1']}, 2: {'D': ['D2'], 'U': ['U2']}, 3: {'D': ['D3'], 'U': ['U3']},
         4: {'D': ['D4'], 'U': ['U4']}, 5: {'D': ['D5'], 'U': ['U5']}, 6: {'D': ['D6'], 'U': ['U6']},
         'C': ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']}
        )

    return macid
