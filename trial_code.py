#%%
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import networkx as nx


from cpd import NullCPD
from examples2 import umbrella, politician, c2d, signal, road_example, fitness_tracker2, car_accident_predictor, content_reccomender, modified_content_reccomender, basic2agent_3
import matplotlib.pyplot as plt
from incentives import Information, Response, Control, Influence
from pgmpy.inference import BeliefPropagation

import itertools
from collections import defaultdict
import operator


import gambit
import subprocess
from collections import deque






#%%
def main():

  











    # m2 = content_reccomender()
    # m2.draw()
    
    # print(m2.all_inf_inc_nodes(1))
    # print(f"con_nodes {m2.all_con_inc_nodes(1)}")
    # print(m2.all_feasible_con_inc_nodes(1))

    # m2 = c2d()
    # m2.random_instantiation_dec_nodes()
    # m2.MACID_to_Gambit_file()


   

    

#%%

if __name__ == '__main__':
    main()





