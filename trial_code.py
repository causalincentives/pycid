#%%
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.base import DAG
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
from cpd import NullCPD
from examples2 import basic_reason_agent1, basic_reason_agent2, basic_reason_agent3, basic_reason_agent4, basic_reason_agent5, two_stage_PA
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




#%%
def main():

  


    # m = fitness_tracker2()
    # h = Information(m, m.decision_nodes[1], m.utility_nodes[1])
    # print(h.all_inf_inc_nodes())
    

    # m = basic2agent_2()
    # m.draw()
    # plt.figure(1)
    # plt.show()



 
    model = basic_reason_agent3()
    model.draw()
    plt.figure(1)
    plt.show()
    

   

    

#%%

if __name__ == '__main__':
    main()









