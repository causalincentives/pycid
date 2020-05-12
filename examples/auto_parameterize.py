#!/usr/bin/env python
# coding: utf-8

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('pdb', '')
import sys
sys.path.append("..") #TODO: make /cid a package, then delete this
from examples import get_3node_cid, get_5node_cid, get_2dec_cid, get_nested_cid
from parameterize import parameterize_systems, merge_all_nodes
from generate import random_cids
from get_systems import get_motifs, choose_systems, check_systems, get_first_c_index
import numpy as np
from verify_incentive import verify_incentive
import networkx as nx
from pgmpy.inference.ExactInference import BeliefPropagation
import matplotlib.pyplot as plt
import time
#TODO: allow multiple utility nodes

def draw_path(cid, path): 
    out = path[0] 
    for j in range(1, len(path)): 
        pred, node = path[j-1:j+1] 
        if pred in cid.get_parents(node): 
            out = out + "->{}".format(node) 
        elif node in cid.get_parents(pred): 
            out = out + "<-{}".format(node) 
        else: 
            raise ValueError('no edge {}--{}'.format(pred, node)) 
    return out 

def main():
    rand_cids = random_cids(n_cids=1000, ns_range = (12, 12), nu_range=(2, 2), nd_range=(2, 2), edge_density=.3)
    cids = [r.trimmed() for r in rand_cids if nx.is_connected(r.trimmed().to_undirected())]

    start_time = time.time()
    tested = 0
    for j, cid in enumerate(cids):
        D = 'D0'
        chance_parents = set(cid.get_parents(D)) - set(cid._get_decisions())
        if chance_parents:
            tested += 1
            X = list(chance_parents)[0]
            systems = choose_systems(cid, D, X)
            systems[0]['i_C']=get_first_c_index(cid, systems[0]['info'])
            all_cpds = parameterize_systems(cid, systems)
            all_cpds = [c.copy() for a in all_cpds for b in a.values() for c in b.values()]
            merged_cpds = merge_all_nodes(cid, all_cpds)
            cid.add_cpds(*merged_cpds.values())
            ev1, ev2 = verify_incentive(cid, D, X)
            print('tested: {}: {}s'.format(tested, time.time() - start_time))
            if ev1<=ev2:
                print('incentive not demonstrated')
                cid.draw()
                break


    print(cid.edges)
    print(cid.utilities)
    print(cid._get_decisions())

    import ipdb; ipdb.set_trace()

if __name__=='__main__':
    main()
