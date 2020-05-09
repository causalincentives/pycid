#Licensed to the Apache Software Foundation (ASF) under one or more contributor license 
#agreements; and to You under the Apache License, Version 2.0.

import numpy as np
#from pgmpy.factors.discrete import TabularCPD
from cid import NullCPD
from get_systems import is_directed, get_motifs
from get_cpd import get_identity_cpd, merge_node, get_equality_cpd, get_xor_cpd, get_func_cpd
from get_cpd import get_equals_func_cpd, get_random_cpd
import warnings


def _parameterize_system(cid, systems, system_idx, H_cpd, Hprime_cpd):
    assert (H_cpd and Hprime_cpd) or (not H_cpd and not Hprime_cpd), 'should either provide Hcpd and Hprime cpd or neither'
    system = systems[system_idx]
    control = system['control']
    info = system['info']
    i_C = system['i_C']
    C = info[i_C]
    X = info[0]
    D = control[0] #TODO: just parameterize D in the control path
    if H_cpd:
        variable_card = 2**H_cpd.variable_card
    else: #if this is the first system (the infolink in question)
        if is_directed(cid, info[i_C:]):
            variable_card = 2
        else:
            variable_card = 4


    info_cpds = {} #maps from node name to TabularCPD or NullCPD
    control_cpds = {}

    #create CPDs for each node, 
    if is_directed(cid, info[i_C:]):
        #parameterize C
        if not H_cpd:
            info_cpds[C] = get_random_cpd((system_idx, 'info', C), variable_card)
            #if C!=H_cpd.variable[2]: #TODO: is this correct?
                #info_cpds[C] = get_identity_cpd(cid, C, H_cpd, (system_idx, 'info'))
        #else:


        #parameterize nodes before C to equal their parents
        for j in range(i_C-1,-1, -1):
            W, parent = info[j:j+2]
            info_cpds[W] = get_identity_cpd(cid, W, info_cpds[parent], (system_idx, 'info'))
        
        #parameterize nodes after C to equal their parents
        #if H_cpd:
        #    W = info[i_C+1]
        #    info_cpds[W] = get_identity_cpd(cid, W, H_cpd, (system_idx, 'info'))
        #else:
        #    parent, W = info[i_C+1:i_C+3]
        #    info_cpds[W] = get_identity_cpd(cid, W, info_cpds[parent], (system_idx, 'info'))
        for j in range(i_C+1,len(info)-1):
            if j==i_C+1 and H_cpd:
                W = info[j]
                info_cpds[W] = get_identity_cpd(cid, W, H_cpd, (system_idx, 'info'))
            else:
                parent, W = info[j-1:j+1]
                info_cpds[W] = get_identity_cpd(cid, W, info_cpds[parent], (system_idx, 'info'))

        #parameterize decision
        if not H_cpd:
            control_cpds[D] = get_identity_cpd(cid, D, info_cpds[info[0]], (system_idx, 'control'))
            #parameterize control path
            for j in range(1, len(control)-1):
                parent, W = control[j-1:j+1]
                control_cpds[W] = get_identity_cpd(cid, W, control_cpds[parent], (system_idx, 'control'))

        #parameterize utility node
        U = control[-1]
        if H_cpd:
            if info[-2]==H_cpd.variable[2]:
                info_cpd = H_cpd
            else:
                info_cpd = info_cpds[info[-2]]
            if control[-2]==Hprime_cpd.variable[2]:
                control_cpd = Hprime_cpd
            else:
                control_cpd = control_cpds[control[-2]]
        else:
            info_cpd = info_cpds[info[-2]]
            control_cpd = control_cpds[control[-2]]
        control_cpds[U] = get_equality_cpd(U, info_cpd, control_cpd, (system_idx, 'control'))

    else:
        #if path from C is not directed, then sample a bistring 2^H_cpd.variable_card at S, F. 
        #XOR it at colliders, and slice it.
        #uniformly sample an integer, which will be interpreted as a function 2^H_cpd.variable_card->2

        motifs = get_motifs(cid, info)
        info_ind = {node:ind for ind, node in enumerate(info)}
        #parameterize infopath except utility node
        for W in cid._get_valid_order(info[:-1]):
            j = info_ind[W]
            motif = motifs[j]
            Y = info[j+1]
            if j==i_C:
                #parameterize C as  F_1[H]
                assert not (H_cpd is None and j==0), 'in first system, if observation is i_C, the infopath should be directed'
                if H_cpd:
                    info_cpds[W] = get_func_cpd(W, info_cpds[Y], H_cpd, (system_idx, 'info'))
                else:
                    X = info[j-1]
                    info_cpds[W] = get_func_cpd(W, info_cpds[Y], info_cpds[X], (system_idx, 'info')) #TODO: is this correct?
            elif j!=i_C:
                motif = motifs[j]
                #parameterize forks
                if motif is 'f' and j<i_C:
                    info_cpds[W] = get_random_cpd((system_idx, 'info', W), variable_card//2)
                elif motif is 'f':
                    info_cpds[W] = get_random_cpd((system_idx, 'info', W), variable_card)
                #parameterize right-chains
                elif motif in ['l']:
                    info_cpds[W] = get_identity_cpd(cid, W, info_cpds[Y], (system_idx, 'info'))
                elif motif is 'r':
                    X = info[j-1]
                    info_cpds[W] = get_identity_cpd(cid, W, info_cpds[X], (system_idx, 'info'))
                #parameterize left-chains
                #parameterize other colliders as XOR
                elif motif is 'c':
                    X = info[j-1]
                    info_cpds[W] = get_xor_cpd(W, info_cpds[X], info_cpds[Y], (system_idx, 'info'))


        #parameterize decision with extra bit
        if H_cpd:
            control_cpds[D] = NullCPD((system_idx, 'control', D), 2)
        else:
            control_cpds[D] = NullCPD((system_idx, 'control', D), variable_card)
        #parameterize control path to transmit extra bit
        for j in range(1, len(control)-1):
            parent, W = control[j-1:j+1]
            control_cpds[W] = get_identity_cpd(cid, W, control_cpds[parent], (system_idx, 'control'))

        #parameterize obs paths
        if 'obs_paths' in system: #TODO: can this be deleted?
            for path in system['obs_paths']:
                #if path[0]!=C: #TODO: is it correct to leave this out?
                for j,W in enumerate(path[:-1]):
                    if j!=0:
                        X = path[j-1]
                        info_cpds[W] = get_identity_cpd(cid, W, info_cpds[X], (system_idx, 'info'))

        #parameterize utility
        U = control[-1]
        control_cpds[U] = get_equals_func_cpd(U, Hprime_cpd, info_cpds[info[-2]], control_cpds[control[-2]], (system_idx, 'control'))


    cpds = {'info':info_cpds, 'control':control_cpds}
    return cpds


def get_Hcpd_idx(systems, idx, directed_path):
    system = systems[idx]
    h_pointer = system['h_pointer']
    C = system['info'][system['i_C']]
    path = systems[h_pointer[0]][h_pointer[1]]
    loc = np.where(np.array(path)==C)[0][0]
    if not directed_path:
        loc -= 1
    Hcpd_idx = h_pointer + tuple([loc])
    return Hcpd_idx

def get_Hcpd(systems, systems_cpds, system_idx, directed_path):
    cpd_idx, path_type, node_idx = get_Hcpd_idx(systems, system_idx, directed_path)
    H = systems[cpd_idx][path_type][node_idx]
    Hcpd = systems_cpds[cpd_idx][path_type][H]
    #if Hcpd.variable[2]=='S2':
    #    import ipdb; ipdb.set_trace()
    return Hcpd

def get_Hprimecpd(systems, systems_cpds, system_idx, directed_path):
    cpd_idx, path_type, _ = get_Hcpd_idx(systems, system_idx, directed_path)
    Hprime = systems[cpd_idx][path_type][-2]
    Hprimecpd = systems_cpds[cpd_idx][path_type][Hprime]
    return Hprimecpd

def parameterize_systems(cid, systems):
    all_cpds = []
    all_cpds.append(_parameterize_system(cid, systems, 0, None, None))
    for i in range(1,len(systems)):
        info = systems[i]['info']
        i_C = systems[i]['i_C']
        directed_path = is_directed(cid, info[i_C:])
        Hcpd = get_Hcpd(systems, all_cpds, i, directed_path)
        Hprimecpd = get_Hprimecpd(systems, all_cpds, i, directed_path)
        #print(Hcpd.variable, Hprimecpd.variable)
        all_cpds.append(_parameterize_system(cid, systems, i, Hcpd, Hprimecpd))
    return all_cpds


def merge_all_nodes(cid, all_cpds):
    all_cpds = [i.copy() for i in all_cpds]
    all_cpds = all_cpds.copy()
    merged_cpds = {}
    for node in cid._get_valid_order(cid.nodes):
        merged_cpds, all_cpds, _, _ = merge_node(cid, merged_cpds, all_cpds, node)
    return merged_cpds




