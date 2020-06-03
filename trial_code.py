#%%
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
from cpd import NullCPD
from examples2 import basic2agent_3
import matplotlib.pyplot as plt
from incentives import Information, Response, Control, Influence
from pgmpy.inference import BeliefPropagation


import itertools
from collections import defaultdict
import operator
import matplotlib.cm as cm



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

    
    
    m = basic2agent_3()
    m.draw()
    plt.figure(1)
    plt.show()

    m.get_all_PSNE()
    m.get_acyclic_topological_ordering()
    m.ordering
    #m.draw_SCCs()
    #m.component_graph()
    #m.get_KM_NE()


    #m.get_all_PSNE()
    #m.get_KM_NE2()

    #m.get_KM_NE3()

    #m.draw_strategic_rel_graph()
    #m.draw_SCCs()
    #m.find_SCCs()
    #m._set_color_SCC()
    #print(len(m.edges()))

    #m._is_s_reachable(['B2E', 'B3E'])


 





    print("finished running")




#%%

if __name__ == '__main__':
    main()





