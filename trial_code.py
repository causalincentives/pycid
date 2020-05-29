#%%
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
from cpd import NullCPD
from examples2 import road_example
import matplotlib.pyplot as plt
from incentives import Information, Response, Control, Influence
from pgmpy.inference import BeliefPropagation

import itertools
from collections import defaultdict
import operator



#%%
def main():

  


    # m = fitness_tracker2()
    # h = Information(m, m.decision_nodes[1], m.utility_nodes[1])
    # print(h.all_inf_inc_nodes())
    

    # m = basic2agent_2()
    # m.draw()
    # plt.figure(1)
    # plt.show()

    # #m.strategically_acyclic()
    # m.strategic_rel_graph()
    # print(m.get_acyclic_topological_ordering())
    # m.get_KM_NE()

    
    
    m = road_example()
    m.draw()
    plt.figure(1)
    plt.show()

    m.draw_strategic_rel_graph()

    #print(len(m.edges()))

    m._is_s_reachable(['B2E', 'B3E'])



    print("finished running")




#%%

if __name__ == '__main__':
    main()




