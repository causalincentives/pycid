#Licensed to the Apache Software Foundation (ASF) under one or more contributor license 
#agreements; and to You under the Apache License, Version 2.0.

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from cid import NullCPD
from get_right_paths import is_directed


#TODO: add state names everywhere
#TODO: make nested label system
#TODO: how to we turn variables from dictionaries to bitstrings and back?

#def parameterize_paths_first(cid, paths, D, X):
#    cpds = {} #maps from node name to TabularCPD or NullCPDs
#    #compute i_C and modified infopath
#    
#    #if C is upstream of X, then it's a special case of history extending where h=''
#    
#    #if C is downstream of X, then parameterize as a variant of history-extending where C does not contain X
#    return funcs
#
#
#def parameterize_recursive():
#    pass


def __get_identity_values(dims, axis):
    A = np.ones((1,np.product(dims[:axis])))
    B = np.eye(dims[axis])
    C = np.ones((1, np.product(dims[axis+1:])))
    print(A, B, C)
    values = np.kron(np.kron(A, B), C)
    return values

def get_identity_cpd(cid, cpds, node, parent, state_names):
    if node in cid._get_decisions():
        if parent:
            if state_names is None:
                state_names = cpds[parent].state_names
            parent_card = cpds[parent].variable_card
        else:
            parent_card = 2
            state_names = StateName('{}-{}hatH'.format(parent,node))
        cpd = NullCPD(node, parent_card, state_names)
    else:
        evidence = cid.get_parents(node)
        evidence_card = [cpds[e].variable_card for e in evidence]
        parent_card = cpds[parent].variable_card
        if parent:
            if state_names is None:
                state_names = cpds[parent].state_names
            parent_idx = np.where(np.array(evidence)==parent)[0][0]
            values = __get_identity_values(evidence_card,parent_idx)
        else:
            parent_card = []
            values = np.array([[0.5, 0.5]])
            state_names = StateName('{}-{}H.format(parent,node)')

        cpd = TabularCPD(variable=node,
                        variable_card=parent_card,
                         evidence=evidence,
                         evidence_card=evidence_card,
                         values=values,
                         state_names = state_names
                        )
    return cpd


def get_shapes(cid, info, i_C):
    shapes = []
    for i in range(len(info)):
        if i==0:
            shapes.append('start')
        elif i==len(info):
            shapes.append('end')
        elif info[i] in cid.get_parents(info[i-1]) and info[i] in cid.get_parents(info[i+1]):
            return 'f'
        elif info[i-1] in cid.get_parents(info[i]) and info[i+1] in cid.get_parents(info[i]):
            return 'c'
        elif info[i-1] in cid.get_parents(info[i]) and info[i] in cid.get_parents(info[i+1]):
            return 'r'
        elif info[i] in cid.get_parents(info[i-1]) and info[i+1] in cid.get_parents(info[i]):
            return 'l'
    return



def parameterize_paths_subsequent(cid, all_paths, infolink, H_cpd):
    #paths is a dict {'info':..,'control':..,'i_C':..} for infolink
    #H is the parent of C along the previous path


    #TODO: decide H input in a way that works for all cases
    if H_cpd: #TODO: tidy this up, when we decide what to do with H_cpd
        h_state_names = H_cpd.state_names
    else:
        h_state_names = None
    variable_card = 2**H_cpd.variable_card

    info_cpds = {} #maps from node name to TabularCPD or NullCPDs
    control_cpds = {}
    paths = all_paths[infolink]
    control_cpds = {}
    control = paths['control']
    info = paths['info']
    i_C = paths['i_C']
    C = info[i_C]
    ordered_decisions = cid._get_valid_order(cid._get_decisions())
    later_decisions = ordered_decisions[np.where(np.array(ordered_decisions)==infolink[1])[0][0]+1:]
    later_Cs = [paths['info'][paths['i_C']] for XD, paths in all_paths.items() 
            if XD[1] in later_decisions]

    
    #create CPDs for each node, 
    if is_directed(cid, info[i_C:]):
        #parameterize C
        info_cpds[C] = H_cpd #TODO: does a modification need to be performed?

        #parameterize nodes before C to equal their parents
        for j in range(i_C-1,-1, -1):
            parent, W = paths['info'][j+1:j-1]
            info_cpds[W] = get_identity_cpd(cid, info_cpds, W, parent, state_names=None)
        
        #parameterize nodes after C to equal their parents
        for j in range(i_C+1,len(paths['info'])-1):
            parent, W = paths['info'][j-1:j+1]
            info_cpds[W] = get_identity_cpd(cid, info_cpds, W, parent, state_names=None)
            if W in later_Cs:
                break
        #parameterize decision/control path
        control_cpds[infolink[1]] = get_identity_cpd(cid, info_cpds, infolink[1], infolink[0], state_names=None)
        for j, W in enumerate(paths['control'][1:-1]):
            parent = paths['info'][j-1]
            control_cpds[W] = get_identity_cpd(cid, control_cpds, W, parent, state_names=None)
            if W in later_Cs:
                break

                
    #if path from C is directed, then C=H, and downstream paths copy it, 
    #when it reaches the next decision, then it calls parameterize_other()
    
    else:
        #if path from C is not directed, then sample a bistring 2^H_cpd.variable_card at S, F. 
        #XOR it at colliders, and slice it.
        #uniformly sample an integer, which will be interpreted as a function 2^H_cpd.variable_card->2
            shapes = get_shapes(cid, info)

        #each infopath fork is randomly sampled with length |H|

        #chains after i_C are equal to their parent in the infopath

        #C is equal to (H, F_1(H))
        info_cpds[C] = H_cpd #TODO: this should merge any inbound cpds into H, and then compute F_1[H]

        #nodes before C on infopath are equal to their parent

        #parameterize decision/control path
        control_cpds[infolink[1]] = get_identity_cpd(cid, info_cpds, infolink[1], infolink[0], state_names=None)
        for j, W in enumerate(paths['control'][1:-1]):
            parent = paths['info'][j-1]
            control_cpds[W] = get_identity_cpd(cid, control_cpds, W, parent, state_names=None)
            if W in later_Cs:
                break





        evidence = cid.get_parents('S1')
        width = 2**len(evidence)
        variable_card = 2**H_cpd.variable_card
        values = np.ones((variable_card, width))/variable_card 
        cpds[C] = TabularCPD(variable='S1',
                             variable_card=variable_card,
                             evidence=evidence,
                             evidence_card=len(evidence),
                             values=values)
    cpds = {'info':info_cpds, 'control':control_cpds}
    return cpds

def parameterize_graph(cid, all_paths, D, X):
    all_cpds = {}
    all_cpds[(X, D)] = parameterize_first(cid, all_paths, D, X, None)
    ordered_decisions = cid._get_valid_order(cid._get_decisions())
    later_decisions = ordered_decisions[np.where(np.array(ordered_decisions)==D)[0][0]+1:]
    for decision in later_decisions:
        relevant_infolinks = [(x, d) for x, d in all_paths.keys() if d==decision]
        earlier_decisions = ordered_decisions[:np.where(ordered_decisions==D)[0][0]]
        for infolink in relevant_infolinks:
            paths = all_paths[infolink]
            info = paths['info']
            i_C = [paths['i_C']]
            C = info[i_C]
            Hs = [all_paths[XD][name][-1] for XD in earlier_decisions if all_paths[XD][name][-1]==C]
            H_cpds = [al_cpds[XD][H] for H in Hs]
            H_cpd = merge_cpds(H_cpds)
            all_cpds[infolink] = parameterize_paths_subsequent(cid, all_paths, infolink, H_cpd)
        all_cpds[infolink] = parameterize_utilities(cid, all_paths, infolink, H_cpd)

    merged_cpds = {}
    for W in cid.nodes:
        merged_cpds[W] = merge_cpds(cid, all_paths, all_cpds, W)
        #TODO: write merge_cpds
        #merge_cpds should parameterize utilities

        

#def parameterize_utilities():
#    state_names = info_cpds[parent].state_names[parent]
#    h_idx = 
#    hat_idx = np.where(state_names==...
#    values = ...
#    info_cpds[W] = ...


def _combine_values(A, B):
    #take a product of two probability tables
    if len(A.shape)==1:
        A = A[:,np.newaxis]
    if len(B.shape)==1:
        B = B[:,np.newaxis]
    C = np.array([np.outer(A[:,i], B[:,i]).ravel() for i in range(A.shape[1])]).T
    return C

def merge_cpds(cpd1, cpd2):
    #merge two TabularCPDs that apply to the same node
    assert cpd1.variable==cpd2.variable
    variable = cpd1.variable
    variable_card = cpd1.variable_card * cpd2.variable_card
    state_names = {variable:list(zip(cpd1.state_names[variable],
                                    cpd2.state_names[variable]))}
    values = _combine_values(cpd1.get_values(), cpd2.get_values())
    evidence = cpd1.get_evidence()
    evidence_card = cpd1.cardinality[1:]
    cpd3 = TabularCPD(variable=variable,
                     variable_card=variable_card,
                     values=values,
                      evidence=evidence,
                      evidence_card=evidence_card,
                     state_names=state_names)
    return cpd3

def merge_parameterization(cpds1, cpds2):
    cpds3 = {}
    cpds3.update(cpds1)
    cpds3.update(cpds2)
    for key, value in cpds3:
        cpds3[key] = merge_cpds(cpds1, cpds2)
    return cpds3

def merge_parameterizations(cpds):
    #take CPDs for each infolink and combine them
    
    #take cross-product of CPDs
    pass
    



