#Licensed to the Apache Software Foundation (ASF) under one or more contributor license 
#agreements; and to You under the Apache License, Version 2.0.

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from cid import NullCPD
from get_systems import is_directed
from get_cpd import get_identity_cpd, merge_node, get_equality_cpd, get_xor_cpd, get_func_cpd
from get_cpd import get_equals_func_cpd

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

def _parameterize_system(cid, systems, system_idx, H_cpd, Hprime_cpd):
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
            info_cpds[C] = get_identity_cpd(cid, C, H_cpd, (system_idx, 'info'))
        else:
            info_cpds[C] =  TabularCPD(variable=(system_idx, 'info', C),
                        variable_card=variable_card,
                         evidence=[],
                         evidence_card=[],
                         values=np.array(np.tile(1/variable_card,(1,variable_card)))
                         #state_names = None,
                        )
        #parameterize nodes before C to equal their parents
        for j in range(i_C-1,-1, -1):
            parent, W = info[j+1:j-1]
            info_cpds[W] = get_identity_cpd(cid, W, info_cpds[parent], (system_idx, 'info'))
        
        #parameterize nodes after C to equal their parents
        for j in range(i_C+1,len(info)-1):
            parent, W = info[j-1:j+1]
            info_cpds[W] = get_identity_cpd(cid, W, info_cpds[parent], (system_idx, 'info'))

        #parameterize decision
        control_cpds[D] = get_identity_cpd(cid, D, info_cpds[info[0]], (system_idx, 'control'))
        #parameterize control path
        for j in range(1, len(control)-1):
            parent, W = control[j-1:j+1]
            control_cpds[W] = get_identity_cpd(cid, W, control_cpds[parent], (system_idx, 'control'))
        #parameterize utility node
        U = control[-1]
        info_cpd = control_cpds[control[-2]]
        control_cpd = info_cpds[info[-2]]
        control_cpds[U] = get_equality_cpd(U, info_cpd, control_cpd, (system_idx, 'control'))

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
                info_cpds[W] = TabularCPD(variable=(system_idx, 'info', W),
                            variable_card=variable_card,
                             evidence=[],
                             evidence_card=[],
                             values=np.array(np.tile(1/variable_card,(1,variable_card)))
                             #state_names = None,
                            )
            #parameterize right-chains
            if motif is 'r':
                parent = info[j-1]
                info_cpds[W] = get_identity_cpd(cid, W, info_cpds[parent], (system_idx, 'info'))

        #parameterize C as  F_1[H]
        assert not (H_cpd is None and i_C==0), 'in first system, if observation is i_C, the infopath should be directed'
        if H_cpd:
            info_cpds[C] = get_func_cpd(C, info_cpds[info[i_C+1]], H_cpd, (system_idx, 'info'))
        else:
            info_cpds[C] = get_func_cpd(C, info_cpds[info[i_C+1]], info_cpds[info[i_C-1]], (system_idx, 'info')) #TODO: is this correct?

        for j in range(len(info)-2, i_C, -1):
            X, W, Y = info[j-1:j+2]
            motif = motifs[j]
            #parameterize left-chains
            if motif is 'l':
                parent = Y
                info_cpds[W] = get_identity_cpd(cid, W, info_cpds[Y], (system_idx, 'info'))

            #parameterize other colliders as XOR
            elif motif is 'c':
                info_cpds[W] = get_xor_cpd(W, info_cpds[X], info_cpds[Y])


        #parameterize decision with extra bit
        control_cpds[D] = NullCPD((system_idx, 'control', D), 2)
        #parameterize control path to transmit extra bit
        for j in range(1, len(control)-1):
            parent, W = control[j-1:j+1]
            control_cpds[W] = get_identity_cpd(cid, W, control_cpds[parent], (system_idx, 'control'))

        #parameterize utility
        U = control[-1]
        control_cpds[U] = get_equals_func_cpd(U, Hprime_cpd, info_cpds[info[-2]], control_cpds[control[-2]], (system_idx, 'control'))
        #TODO: change to apply function of control_cpds[control[-2]] to the history then check equality with info[-2]


    cpds = {'info':info_cpds, 'control':control_cpds}
    return cpds


def get_Hcpd_idx(systems, idx):
    system = systems[idx]
    h_pointer = system['h_pointer']
    C = system['info'][system['i_C']]
    path = systems[h_pointer[0]][h_pointer[1]]
    loc = np.where(np.array(path)==C)[0][0]
    Hcpd_idx = h_pointer + tuple([loc-1])
    return Hcpd_idx

def get_Hcpd(systems, systems_cpds, system_idx):
    cpd_idx, path_type, node_idx = get_Hcpd_idx(systems, system_idx)
    H = systems[cpd_idx][path_type][node_idx]
    Hcpd = systems_cpds[cpd_idx][path_type][H]
    return Hcpd

def get_Hprimecpd(systems, systems_cpds, system_idx):
    cpd_idx, path_type, _ = get_Hcpd_idx(systems, system_idx)
    Hprime = systems[cpd_idx][path_type][-2]
    Hprimecpd = systems_cpds[cpd_idx][path_type][Hprime]
    return Hprimecpd

def parameterize_systems(cid, systems):
    all_cpds = []
    all_cpds.append(_parameterize_system(cid, systems, 0, None, None))
    for i in range(1,len(systems)):
        Hcpd = get_Hcpd(systems, all_cpds, i)
        Hprimecpd = get_Hprimecpd(systems, all_cpds, i)
        print(Hcpd.variable, Hprimecpd.variable)
        all_cpds.append(_parameterize_system(cid, systems, i, Hcpd, Hprimecpd))
    return all_cpds


def merge_all_nodes(cid, all_cpds):
    all_cpds = [i.copy() for i in all_cpds]
    all_cpds = all_cpds.copy()
    merged_cpds = {}
    for node in cid._get_valid_order(cid.nodes):
        merged_cpds, all_cpds, _, _ = merge_node(cid, merged_cpds, all_cpds, node)
    return merged_cpds




