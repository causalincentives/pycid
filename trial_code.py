#%%
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.base import DAG
from pgmpy.factors.discrete import TabularCPD
import networkx as nx

from cpd import NullCPD
<<<<<<< HEAD
from examples2 import umbrella, sequential, politician, c2d, signal, road_example, fitness_tracker2, car_accident_predictor, content_reccomender, modified_content_reccomender, basic2agent_2, triage
=======
from examples2 import basic_reason_agent1, basic_reason_agent2, basic_reason_agent3, basic_reason_agent4, basic_reason_agent5, two_stage_PA
>>>>>>> 8b15a79a53a4b16dc9184fbfd8e8ee6c2ebb6ac9
import matplotlib.pyplot as plt
from incentives import Information, Response, Control, Influence

from pgmpy.inference import BeliefPropagation
from pgmpy.inference.CausalInference import CausalInference

import itertools

from collections import defaultdict
import operator
import matplotlib.cm as cm

from functools import lru_cache



#from reasoning import Reasoning



from typing import List
from collections import Iterable

import copy




#import gambit

import subprocess


from collections import deque




#%%
def main():

  



<<<<<<< HEAD
=======
    # m = basic2agent_2()
    # m.draw()
    # plt.figure(1)
    # plt.show()
>>>>>>> 8b15a79a53a4b16dc9184fbfd8e8ee6c2ebb6ac9



 
    model = basic_reason_agent3()
    model.draw()
    plt.figure(1)
    plt.show()
    

   

    

#%%

<<<<<<< HEAD
=======
if __name__ == '__main__':
    main()
>>>>>>> 8b15a79a53a4b16dc9184fbfd8e8ee6c2ebb6ac9

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


<<<<<<< HEAD
    m = basic2agent_2()
    # m.draw()
    m.get_all_PSNE()

   
    # m3 = signal()
    # #m3.draw()
    # m3.draw_strategic_rel_graph()
    # m3.draw_SCCs()
    

#%%
=======
>>>>>>> 8b15a79a53a4b16dc9184fbfd8e8ee6c2ebb6ac9






