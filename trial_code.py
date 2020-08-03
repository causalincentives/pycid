#%%
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import networkx as nx

from cpd import NullCPD
from examples2 import umbrella, sequential, politician, c2d, signal, road_example, fitness_tracker2, car_accident_predictor, content_reccomender, modified_content_reccomender, basic2agent_2, triage
import matplotlib.pyplot as plt
from incentives import Information, Response, Control, Influence
from pgmpy.inference import BeliefPropagation

import itertools

from collections import defaultdict
import operator



#import gambit

import subprocess


from collections import deque




#%%
def main():

  











    #m2 = modified_content_reccomender()
    m2 = content_reccomender()
    m2.draw()
    
    print(m2.all_inf_inc_nodes(1))
    print(f"con_nodes {m2.all_con_inc_nodes(1)}")
    print(m2.all_feasible_con_inc_nodes(1))

    # m2 = road_example()
    # m2.draw()
    # m2.draw_strategic_rel_graph()
    # m2.draw_SCCs()

    # m2.random_instantiation_dec_nodes()
    # m2.MACID_to_Gambit_file()


    m = basic2agent_2()
    # m.draw()
    m.get_all_PSNE()

   
    # m3 = signal()
    # #m3.draw()
    # m3.draw_strategic_rel_graph()
    # m3.draw_SCCs()
    

#%%

if __name__ == '__main__':
    main()





