import numpy as np
from pgmpy.factors.discrete import TabularCPD
from core.macid import MACID
from core.cpd import DecisionDomain


def get_basic2agent():
    macid = MACID([
        ('D1', 'D2'),
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1'),
        ],
        {'A': {'D': ['D1'], 'U': ['U1']},
         'B': {'D': ['D2'], 'U': ['U2']}})
    return macid



def get_basic2agent2():
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
        },     #defines the decisions, chance nodes and utility nodes for each agent
         
         )     

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
