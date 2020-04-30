#big todos: 
# 1) finish parameterize_system. should take systems as input and do extension case and utilities.

#Licensed to the Apache Software Foundation (ASF) under one or more contributor license 
#agreements; and to You under the Apache License, Version 2.0.

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from cid import NullCPD
from get_systems import is_directed
from get_cpd import get_identity_cpd, merge_cpds, get_equality_cpd, get_xor_cpd

def get_motifs(cid, path):
    shapes = []
    for i in range(len(path)):
        if i==0:
            shapes.append('start')
        elif i==len(path)-1:
            shapes.append('end')
        elif path[i] in cid.get_parents(path[i-1]) and path[i] in cid.get_parents(path[i+1]):
            shapes.append('f')
        elif path[i-1] in cid.get_parents(path[i]) and path[i+1] in cid.get_parents(path[i]):
            shapes.append('c')
        elif path[i-1] in cid.get_parents(path[i]) and path[i] in cid.get_parents(path[i+1]):
            shapes.append('r')
        elif path[i] in cid.get_parents(path[i-1]) and path[i+1] in cid.get_parents(path[i]):
            shapes.append('l')
    return shapes

def parameterize_system(cid, systems, system_idx, H_cpd):
    if H_cpd:
        h_state_names = H_cpd.state_names
        H = H_cpd.variable
        variable_card = 2**H_cpd.variable_card
    else: #if this is the first system (the infolink in question)
        h_state_names = None
        variable_card = 2

    system = systems[system_idx]
    control = system['control']
    info = system['info']
    i_C = system['i_C']
    C = info[i_C]
    X = info[0]
    D = control[0]

    info_cpds = {} #maps from node name to TabularCPD or NullCPDs
    control_cpds = {}
    #ordered_decisions = cid._get_valid_order(cid._get_decisions())
    #later_decisions = ordered_decisions[np.where(np.array(ordered_decisions)==D)[0][0]+1:]
    #later_Cs = [system['info'][system['i_C']] for system in systems
    #        if system['control'][0] in later_decisions]

    #create CPDs for each node, 
    if is_directed(cid, info[i_C:]):
        #parameterize C
        if H_cpd:
            info_cpds[C] = get_identity_cpd(cid, {H:H_cpd, **info_cpds}, C, H_cpd.variable)#, state_names=None)
        else:
            info_cpds[C] =  TabularCPD(variable=C,
                        variable_card=variable_card,
                         evidence=[],
                         evidence_card=[],
                         values=np.array(np.tile(1/variable_card,(1,variable_card)))
                         #state_names = None,
                        )
        #parameterize nodes before C to equal their parents
        for j in range(i_C-1,-1, -1):
            parent, W = info[j+1:j-1]
            info_cpds[W] = get_identity_cpd(cid, info_cpds, W, parent)#, state_names=None)
        
        #parameterize nodes after C to equal their parents
        for j in range(i_C+1,len(info)-1):
            parent, W = info[j-1:j+1]
            info_cpds[W] = get_identity_cpd(cid, info_cpds, W, parent)#, state_names=None)

        #parameterize decision
        control_cpds[D] = get_identity_cpd(cid, info_cpds, D, info[0])
        #parameterize control path
        for j, W in enumerate(control[1:-1]):
            parent = control[j-1]
            control_cpds[W] = get_identity_cpd(cid, control_cpds, W, parent)#, state_names=None)
        #parameterize utility node
        U = control[-1]
        info_cpd = control_cpds[control[-2]]
        control_cpd = info_cpds[info[-2]]
        control_cpds[U] = get_equality_cpd(U, info_cpd, control_cpd)

    else:
        #if path from C is not directed, then sample a bistring 2^H_cpd.variable_card at S, F. 
        #XOR it at colliders, and slice it.
        #uniformly sample an integer, which will be interpreted as a function 2^H_cpd.variable_card->2
        motifs = get_motifs(cid, info)

        for j in range(i_C+1, len(info)):
            W = info[j]
            motif = motifs[j]
            #parameterize forks
            if motif is 'f':
                info_cpds[W] = TabularCPD(variable=W,
                            variable_card=variable_card,
                             evidence=[],
                             evidence_card=[],
                             values=np.array(np.tile(1/variable_card,(1,variable_card)))
                             #state_names = None,
                            )
            #parameterize right-chains
            if motif is 'r':
                parent = info[j-1]
                info_cpds[W] = get_identity_cpd(cid, info_cpds, W, parent)#, state_names=None)

        for j in range(len(info)-2, i_C, -1):
            X, W, Y = info[j-1:j+2]
            motif = motifs[j]
            #parameterize left-chains
            if motif is 'l':
                parent = Y
                info_cpds[W] = get_identity_cpd(cid, info_cpds, W, Y)#, state_names=None)

            #parameterize C as  F_1[H]
            if W==C:
                assert X in cid.get_parents(W) and Y in cid.get_parents(W)
                info_cpds[W] = get_func_cpd(W, info_cpds[X], info_cpds[Y])
            #parameterize other colliders as XOR
            elif motif is 'c':
                info_cpds[W] = get_xor_cpd(W, info_cpds[X], info_cpds[Y])

        #parameterize decision with extra bit
        control_cpds[D] = NullCPD(D, 2)
        #parameterize control path to transmit extra bit
        for j, W in enumerate(control[1:-1]):
            parent = control[j-1]
            control_cpds[W] = get_identity_cpd(cid, control_cpds, W, parent)#, state_names=None)

        #parameterize utility
        U = control[-1]
        control_cpds[U] = get_equality_cpd(U, info_cpds[info[-2]], control_cpds[control[-2]])

    cpds = {'info':info_cpds, 'control':control_cpds}
    return cpds



def parameterize_graph(cid, systems, infolink):
    #systems is a list of dicts {'history':(1, 'control'),'control':..,'info':..}
    sys_cpds = [] #list of lists of cpds
    for system in systems:
        if system['h_pointer']:
            H_cpd = get_H_cpd(sys_cpds, systems, h_pointer)
        else:
            H_cpd = None
        sys_cpds.append(parameterize_system(cid, system, H_cpd))

    node_cpds = merge_cpds(cid, sys_cpds, node)
    return node_cpds


