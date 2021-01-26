import numpy as np
from pgmpy.factors.discrete import TabularCPD
from core.macid import MACID
from core.cpd import DecisionDomain


def get_basic2agent_acyclic():
    macid = MACID([
        ('D1', 'D2'),
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1'),
        ],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']}})
    return macid


def get_basic2agent_cyclic():
    macid = MACID([
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1'),
        ],
        {0: {'D': ['D1'], 'U': ['U1']},
         1: {'D': ['D2'], 'U': ['U2']}})

    return macid


def get_basic_subgames():
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
        ('X2', 'D12'),
        ],
        {0: {'D': ['D11', 'D12'], 'U': ['U11']},
        1: {'D': ['D2'], 'U': ['U2', 'U22']},
        2: {'D': ['D3'], 'U': ['U3']},
         })

    return macid


def get_basic_subgames2():
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
        ('X1', 'U4'),
        ],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']},
         3: {'D': ['D3'], 'U': ['U3']},
         4: {'D': ['D4'], 'U': ['U4']},

         })

    return macid


def get_path_example():
    macid = MACID([
        ('X1', 'X3'),
        ('X1', 'D'),
        ('X2', 'D'),
        ('X2', 'U'),
        ('D', 'U')],
        {1: {'D': ['D'], 'U': ['U']}})
    return macid


def example_temp():
    macid = MACID([
        ('D1mec', 'D1'),
        ('D1', 'U1'),
        ('X1', 'U1'),
        ('X1', 'D1')],
        {1: {'D': ['D1'], 'U': ['U1']}})
    return macid


def basic2agent_2():
    macid = MACID([
        ('D1', 'D2'),
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1'),
        ],
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


def basic_rel_agent():
    macid = MACID([
        ('D1', 'D2'),
        ('D1', 'U2'),
        ('D1', 'U1'),
        ('D2', 'U2'),
        ('D2', 'U1'),
        ],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']}})

    return macid


def basic_rel_agent2():
    macid = MACID([
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1')
        ],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']}})

    return macid

def basic_rel_agent3():
    macid = MACID([
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1'),
        ('Ch', 'D1'),
        ('Ch', 'U1'),
        ('Ch', 'U2')
        ],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']}})

    return macid

def basic_rel_agent4():
    macid = MACID([
        ('D1', 'Ch'),
        ('Ch', 'D2'),
        ('Ch', 'U1'),
        ('Ch', 'U2'),
        ('D2', 'U1'),
        ('D2', 'U2')
        ],
        {1: {'D': ['D1'], 'U': ['U1']},
         2: {'D': ['D2'], 'U': ['U2']}},
        )
    return macid


def basic2agent_3():
    from pgmpy.factors.discrete.CPD import TabularCPD
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
        ('D3', 'U3'),
        ],
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


def c2d():
    macid = MACID([
        ('C1', 'U1'),
        ('C1', 'U2'),
        ('C1', 'D1'),
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D1', 'D2'),
        ('D2', 'U1'),
        ('D2', 'U2'),
        ('C1', 'D2'),
        ],
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
