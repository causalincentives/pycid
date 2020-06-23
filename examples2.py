#%%

from pgmpy.factors.discrete import TabularCPD
import numpy as np
from macid import MACID
from cpd import NullCPD
import itertools
import operator
import functools


#%%

# Example from understanding agent incentives paper - to test the incentives classes

def fitness_tracker2():
    from pgmpy.factors.discrete.CPD import TabularCPD
    macid = MACID([
        ('TD', 'TF'),
        ('TF', 'SC'),
        ('TF', 'C'),
        ('EF', 'EWD'),
        ('EWD', 'C'),
        ('C', 'F'),
        ('P', 'D'),
        ('P', 'SC'),
        ('P', 'F'),
        ('SC', 'C'),
        ],
        {1: {'D': ['C'], 'U': ['F']}, 'C': ['TD', 'EF', 'EWD', 'D', 'P', 'TF', 'SC']})

    return macid


def car_accident_predictor():
    from pgmpy.factors.discrete.CPD import TabularCPD
    macid = MACID([
        ('B', 'N'),
        ('N', 'AP'),
        ('N', 'P'),
        ('P', 'Racc'),
        ('Age', 'Adt'),
        ('Adt', 'Racc'),
        ('Racc', 'Accu'),
        ('M', 'AP'),
        ('AP', 'Accu'),
        ],
        {1: {'D': ['AP'],'U': ['Accu']}, 'C': ['B', 'N', 'P', 'Racc', 'Adt', 'Age', 'M']},     #defines the decisions, chance nodes and utility nodes for each agent
        )     

    return macid

def content_reccomender():
    from pgmpy.factors.discrete.CPD import TabularCPD
    macid = MACID([
        ('O', 'I'),
        ('O', 'M'),
        ('M', 'P'),
        ('P', 'I'),
        ('I', 'C'),
        ('P', 'C'),
        ],
        {1: {'D': ['P'],'U': ['C']}, 'C': ['O', 'M', 'I']},     #defines the decisions, chance nodes and utility nodes for each agent
        )     

    return macid


def modified_content_reccomender():
    from pgmpy.factors.discrete.CPD import TabularCPD
    macid = MACID([
        ('O', 'I'),
        ('O', 'M'),
        ('M', 'P'),
        ('P', 'I'),
        ('P', 'C'),
        ('M', 'C')
        ],
        {1: {'D': ['P'],'U': ['C']}, 'C': ['O', 'M', 'I']},     #defines the decisions, chance nodes and utility nodes for each agent
        )     

    return macid





#%%
def basic2agent():
    from pgmpy.factors.discrete.CPD import TabularCPD
    macid = MACID([
        ('D1', 'D2'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1'),
        ],
        {'A': {'D': ['D1'], 'U': ['U1']}, 'B': {'D': ['D2'], 'U': ['U2']}, 'C': []})

    return macid


#%%
def basic2agent_2():
    from pgmpy.factors.discrete.CPD import TabularCPD
    macid = MACID([
        ('D1', 'D2'),
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1'),
        ],
        {1: {'D': ['D1'], 'U': ['U1']}, 2: {'D': ['D2'], 'U': ['U2']}, 'C': []},     #defines the decisions, chance nodes and utility nodes for each agent
        {'U1' :list(range(5)), 'U2': list(range(5))} 
        )
        
        
        # {1:[2,3,1,3], 2:[4,1,2,3]})     #defines utilities 

    
    cpd_D1 = NullCPD('D1', 2)
    cpd_D2 = NullCPD('D2', 2)

    cpd_U1 = TabularCPD('U1', 5, np.array([[0, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0]]), evidence=['D1', 'D2'], evidence_card=[2,2])
    cpd_U2 = TabularCPD('U2', 5, np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]), evidence=['D1', 'D2'], evidence_card=[2,2])

    macid.add_cpds(cpd_D1, cpd_D2, cpd_U1, cpd_U2)




    for dec in macid.all_decision_nodes:
        cpd_d = NullCPD(dec, 2)
        macid.add_cpds(cpd_d)

    for util in macid.all_utility_nodes:
        #num_unique_utilities = functools.reduce(operator.mul, macid.decision_cardinalities.values())   # I think need to include chance nodes in this too!
        parents = macid.get_parents(util)
        parents_card = [macid.get_cardinality(par) for par in parents]
        num_unique_utilities = functools.reduce(operator.mul, parents_card)   # think about this when defining utility dictionary to feed in.
        cpd_u = TabularCPD(variable=util, variable_card=num_unique_utilities,
                            values=np.eye(num_unique_utilities),
                            evidence=parents,
                            evidence_card=parents_card
                            )
        macid.add_cpds(cpd_u)
    # cpd_u1 = TabularCPD(variable='U1', variable_card=4,
    #                     values=np.eye(4),
    #                     evidence=['D1', 'D2'], evidence_card=[2, 2])
    # cpd_u2 = TabularCPD(variable='U2', variable_card=4,
    #                     values=np.eye(4),
    #                     evidence=['D1', 'D2'], evidence_card=[2, 2])
    
    # macid.add_cpds(cpd_u1, cpd_u2)
    print("added cpds")

    return macid

# %%

def basic_rel_agent():
    from pgmpy.factors.discrete.CPD import TabularCPD
    macid = MACID([
        ('D1', 'D2'),
        ('D1', 'U2'),
        ('D1', 'U1'),
        ('D2', 'U2'),
        ('D2', 'U1'),
        ],
        {1: {'D': ['D1'],'U': ['U1']}, 2: {'D': ['D2'], 'U': ['U2']}, 'C': []},     #defines the decisions, chance nodes and utility nodes for each agent
       )     #defines utilities 

    return macid


# %%

def basic_rel_agent2():
    from pgmpy.factors.discrete.CPD import TabularCPD
    macid = MACID([
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1')
        ],
        {1: {'D': ['D1'],'U': ['U1']}, 2: {'D': ['D2'], 'U': ['U2']}, 'C': []},     #defines the decisions, chance nodes and utility nodes for each agent
        )     #defines utilities 

    return macid


# %%

def basic_rel_agent3():
    from pgmpy.factors.discrete.CPD import TabularCPD
    macid = MACID([
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U2'),
        ('D2', 'U1'),
        ('Ch', 'D1'),
        ('Ch', 'U1'),
        ('Ch', 'U2')
        ],
        {1: {'D': ['D1'],'U': ['U1']}, 2: {'D': ['D2'], 'U': ['U2']}, 'C': ['Ch']},     #defines the decisions, chance nodes and utility nodes for each agent
        )     #defines utilities 

    return macid

# %%

def basic_rel_agent4():
    from pgmpy.factors.discrete.CPD import TabularCPD
    macid = MACID([
        ('D1', 'Ch'),
        ('Ch', 'D2'),
        ('Ch', 'U1'),
        ('Ch', 'U2'),
        ('D2', 'U1'),
        ('D2', 'U2')
        ],
        {1: {'D': ['D1'],'U': ['U1']}, 2: {'D': ['D2'], 'U': ['U2']}, 'C': ['Ch']},     #defines the decisions, chance nodes and utility nodes for each agent
        )     #defines utilities 

    return macid





# %%

def tree_doctor():
    from pgmpy.factors.discrete.CPD import TabularCPD
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
        {1: {'D': ['PT', 'BP'], 'U': ['E', 'V']}, 2: {'D': ['TDoc'], 'U': ['Tree', 'Cost']}, 'C': ['TS', 'TDead']},     #defines the decisions, chance nodes and utility nodes for each agent
        )     #defines utilities 

    return macid

# %%


def road_example():
    from pgmpy.factors.discrete.CPD import TabularCPD
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
        'C': ['S1W', 'S1E', 'S2W', 'S2E', 'S3W', 'S3E']},     #defines the decisions, chance nodes and utility nodes for each agent
         
         )     

    return macid





#%%
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
        {0: {'D': ['D1'], 'U': ['U1']}, 1: {'D': ['D2'], 'C': [], 'U': ['U2']}, 2: {'D': ['D3'], 'U': ['U3']}, 'C': []},     #defines the decisions, chance nodes and utility nodes for each agent
        {'U1':np.arange(7), 'U2':np.arange(7), 'U3':np.arange(7)} 
        )    
    
    cpd_D1 = NullCPD('D1', 2)
    cpd_D2 = NullCPD('D2', 2)
    cpd_D3 = NullCPD('D3', 2)

    cpd_U1 = TabularCPD('U1', 7, np.array([[0, 0, 0, 1, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]), evidence=['D1', 'D2', 'D3'], evidence_card=[2,2,2])
    cpd_U2 = TabularCPD('U2', 7, np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]), evidence=['D1', 'D2', 'D3'], evidence_card=[2,2,2])
    cpd_U3 = TabularCPD('U3', 7, np.array([[0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0]]), evidence=['D1', 'D2', 'D3'], evidence_card=[2,2,2])

    macid.add_cpds(cpd_D1, cpd_D2, cpd_D3, cpd_U1, cpd_U2, cpd_U3)



    
    # cpd_d1 = NullCPD('D1', 2)
    # cpd_d2 = NullCPD('D2', 2)
    # cpd_d3 = NullCPD('D3', 2)
    # cpd_u1 = TabularCPD(variable='U1', variable_card=8,
    #                     values=np.eye(8),
    #                     evidence=['D1', 'D2', 'D3'], evidence_card=[2, 2, 2])
    # cpd_u2 = TabularCPD(variable='U2', variable_card=8,
    #                     values= np.eye(8),
    #                     evidence=['D1', 'D2', 'D3'], evidence_card=[2, 2, 2])
    # cpd_u3 = TabularCPD(variable='U3', variable_card=8,
    #                     values= np.eye(8),
    #                     evidence=['D1', 'D2', 'D3'], evidence_card=[2, 2, 2])
    
    # macid.add_cpds(cpd_d1, cpd_d2, cpd_d3, cpd_u1, cpd_u2, cpd_u3)
    
    # for dec in macid.all_decision_nodes:
    #     cpd_d = NullCPD(dec, 2)
    #     macid.add_cpds(cpd_d)

    # for util in macid.all_utility_nodes:
    #     #num_unique_utilities = functools.reduce(operator.mul, macid.decision_cardinalities.values())   # I think need to include chance nodes in this too!
    #     parents = macid.get_parents(util)
    #     parents_card = [macid.get_cardinality(par) for par in parents]
    #     num_unique_utilities = functools.reduce(operator.mul, parents_card)   # think about this when defining utility dictionary to feed in.
    #     cpd_u = TabularCPD(variable=util, variable_card=num_unique_utilities,
    #                         values=np.eye(num_unique_utilities),
    #                         evidence=parents,
    #                         evidence_card=parents_card
    #                         )
    #     macid.add_cpds(cpd_u)
    
    # print("added cpds")

    return macid





# %%

def prisoners_dilemma():
    from pgmpy.factors.discrete.CPD import TabularCPD
    macid = MACID([
        ('D1', 'U1'),
        ('D1', 'U2'),
        ('D2', 'U1'),
        ('D2', 'U2'),
        ],
        {1: {'D': ['D1'], 'U': ['U1']}, 2: {'D': ['D2'], 'U': ['U2']}, 'C': []},     #defines the decisions, chance nodes and utility nodes for each agent
        )     #defines utilities 

    return macid


def politician():
    from pgmpy.factors.discrete.CPD import TabularCPD
    macid = MACID([
        ('D1', 'I'),
        ('T', 'I'),
        ('T', 'U2'),
        ('I', 'D2'),
        ('R', 'D2'),
        ('D2', 'U1'),
        ('D2', 'U2'),
        ],
        {1: {'D': ['D1'], 'U': ['U1']}, 2: {'D': ['D2'], 'U': ['U2']}, 'C': ['R', 'I', 'T']},     #defines the decisions, chance nodes and utility nodes for each agent
        )     #defines utilities 

    return macid




def umbrella():
    from pgmpy.factors.discrete.CPD import TabularCPD
    macid = MACID([
        ('W', 'F'),
        ('W', 'A'),
        ('F', 'UM'),
        ('UM', 'A'),
        ],
        {1: {'D': ['UM'], 'U': ['A']}, 'C': ['W', 'F']},     #defines the decisions, chance nodes and utility nodes for each agent
        {'UM': 2}, #defines the decision cardinalities
        )     #defines utilities 


    cpd_W = TabularCPD('W',2,np.array([[.6],[.4]]))
    cpd_F = TabularCPD('F',2,np.array([[.8, .3],[.2,.7]]), evidence=['W'], evidence_card=[2])
    cpd_UM = NullCPD('UM', 2)
    cpd_A = TabularCPD('A', 3, np.array([[0, 1, 1, 0], [1, 0, 0, 0], [0,0,0,1]]), evidence=['W', 'UM'], evidence_card=[2,2])


  
    macid.add_cpds(cpd_W, cpd_F, cpd_UM, cpd_A)



    





    return macid



def c2d():
    from pgmpy.factors.discrete.CPD import TabularCPD
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
        {0: {'D': ['D1'], 'U': ['U1']}, 1: {'D': ['D2'], 'U': ['U2']},'C': ['C1']},     #defines the decisions, chance nodes and utility nodes for each agent
         
        {0:np.arange(6), 1:-np.arange(6)})     #defines utility ranges 


    cpd_C1 = TabularCPD('C1',2,np.array([[.5],[.5]]))
    cpd_D1 = NullCPD('D1', 2)
    cpd_D2 = NullCPD('D2', 2)
    cpd_U1 = TabularCPD('U1', 4, np.array([[0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]]), evidence=['C1', 'D1', 'D2'], evidence_card=[2,2,2])
    cpd_U2 = TabularCPD('U2', 6, np.array([[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]]), evidence=['C1', 'D1', 'D2'], evidence_card=[2,2,2])

    macid.add_cpds(cpd_C1, cpd_D1, cpd_D2, cpd_U1, cpd_U2)




    return macid




print("loaded examples")


# %%


def signal():
    from pgmpy.factors.discrete.CPD import TabularCPD
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
        {0: {'D': ['D1'], 'U': ['U1']}, 1: {'D': ['D2'], 'U': ['U2']},'C': ['X']},     #defines the decisions, chance nodes and utility nodes for each agent
         #defines the decision cardinalities
        )     #defines utilities 


    cpd_X = TabularCPD('X',2,np.array([[.5],[.5]]))
    cpd_D1 = NullCPD('D1', 2)
    cpd_D2 = NullCPD('D1', 2)
    cpd_U1 = TabularCPD('U1', 6, np.array([[0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0]]), evidence=['X', 'D1', 'D2'], evidence_card=[2,2,2])
    cpd_U2 = TabularCPD('U2', 6, np.array([[0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0]]), evidence=['X', 'D1', 'D2'], evidence_card=[2,2,2])


    macid.add_cpds(cpd_X, cpd_D1, cpd_D2, cpd_U1, cpd_U2)




    return macid


# %%



