#Licensed to the Apache Software Foundation (ASF) under one or more contributor license 
#agreements; and to You under the Apache License, Version 2.0.
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from cid import NullCPD
from typing import List

def get_H_cpd(sys_cpds, systems, h_pointer):
    H_cpd = sys_cpds[h_pointer[0]][h_pointer[1]]
    hist = systems[h_pointer[0]]
    H = hist[np.where(np.array(hist[h_pointer[1]])==paths[paths['i_C']])[0][0]-1]
    H_cpd = systems[h_pointer[0]][h_pointer[1]]
    return H_cpd

#def __project_values(dims, axis, values): #TODO: is this a special case of project_values?
#    A = np.ones((1,np.product(dims[:axis])))
#    C = np.ones((1, np.product(dims[axis+1:])))
#    values = np.kron(np.kron(A, values), C)
#    return values


def __get_identity_values(dims, axis):
    B = np.eye(dims[axis])
    #values = __project_values(dims, axis, B)
    return values

def get_identity_cpd(cid, cpds, node, parent):#, state_names):
    if node in cid._get_decisions():
        if parent:
            #if state_names is None:
            #    state_names = cpds[parent].state_names
            variable_card = cpds[parent].variable_card
        else:
            variable_card = 2
            #state_names = StateName('{}-{}hatH'.format(parent,node))
        cpd = NullCPD(node, variable_card)#, state_names)
    else:
        #evidence = cid.get_parents(node)
        evidence = [parent]
        evidence_card = [cpds[e].variable_card]
        variable_card = cpds[parent].variable_card
        if parent:
            #if state_names is None:
            #    state_names = cpds[parent].state_names
            parent_idx = np.where(np.array(evidence)==parent)[0][0]
            values = __get_identity_values(evidence_card,parent_idx)
        else:
            variable_card = []
            values = np.array([[0.5, 0.5]])
            #state_names = StateName('{}-{}H.format(parent,node)')

        cpd = TabularCPD(variable=node,
                        variable_card=variable_card,
                         evidence=evidence,
                         evidence_card=evidence_card,
                         values=values,
                         #state_names = state_names
                        )
    return cpd

def __recursive_project(dims, axes, values):
    print('dims: {}, axes: {}, values: {}'.format(dims, axes, values))
    if not dims:
        return values
    dims = dims.copy()
    dim = dims.pop(0)
    if axes and axes[0]==0:
        axes = axes.copy()
        axes.pop(0)
        axes = [a-1 for a in axes]
        return np.array([_rec(dims, axes, v) for v in values]).ravel()
    else:
        axes = axes.copy()
        axes = [a-1 for a in axes]
        return np.tile(_rec(dims, axes, values), dim)

def project_row(dims:List[int], axes:List[int], values:np.ndarray):
    #projects an integer-valued function over extra dimensions
    #dims is the evidence_card
    #axes indexes the evidence that is relevant for values
    #values is the integer-valued vector defined over axes
    assert len(values.shape)==1 or values.shape[1]==1
    relevant_dims = np.array(dims)[list(axes)]
    values = values.reshape(relevant_dims)
    return __recursive_project(dims, axes, values)

def project_values(dims, axes, values):
    #projects a TabularCPD values table `values` to `dims` dimensions
    #where `axes` denotes the original dimensions
    return np.array([project(dims, axes, v) for v in values])

def __get_xor_values(size):#(dims, axes)
    #size = dims[axes[0]]
    vec_of_bitstrings = [int(np.binary_repr(i)) for i in range(hsize)]
    one_hot_to_xor_table= np.bitwise_xor.outer(vec_of_bitstrings, vec_of_bitstrings)
    str_vec = one_hot_to_xor_table.ravel().astype(str)
    int_vec = [int(s, 2) for s in str_vec]
    values = np.eye(hsize)[int_vec].T
    #projected = project_values(dims, axes, values)
    return values
    #return projected

def get_xor_cpd(node, cpd1, cpd2): #TODO: do I need cid here? #TODO: can this be a decision?
    evidence = [cpd1.variable_name, cpd2.variable_name]
    evidence_card = [cpd1.variable_card, cpd2.variable_card]
    values = __get_xor_values(cpd1.variable_card)
    cpd = TabularCPD(variable=node,
            variable_card=evidence_card,
            evidence=evidence,
            evidence_card=evidence_card,
            values=values
            )
    return cpd

def __get__func_values(size):
    funcsize = 2**hsize
    bitstrings = [np.binary_repr(i, hsize+funcsize) for i in np.arange(2**(funcsize+hsize))]
    hists = [b[-hsize:] for b in bitstrings]
    funcs = [b[:-hsize] for b in bitstrings]
    indices = [int(h, 2) for h in hists]
    values = np.array([f[i] for f, i in zip(funcs, indices)]).astype(int)
    value_table = np.eye(2)[values].T
    return value_table

def get_func_cpd(node, f_cpd, h_cpd):
    evidence = [f_cpd.variable_name, h_cpd.variable_name]
    evidence_card = [f_cpd.variable_card, h_cpd.variable_card]
    values = __get_func_values(h_cpd.variable_card)
    cpd = TabularCPD(variable=node,
            variable_card=2,
            evidence=evidence,
            evidence_card=evidence_card,
            values=values
            )
    return cpd
    

def __get_equality_values(dim):
    return np.array((np.eye(dim).ravel(), 1-np.eye(dim).ravel()))

def __project_values2(dims, axes, values): #TODO: finish this
    for axis in axes:
        A = np.ones((1, np.product(dims[:axis])))
        X = np.kron(A, X)


    A = np.ones((1,np.product(dims[:axis])))
    C = np.ones((1, np.product(dims[axis+1:])))
    values = np.kron(np.kron(A, B), C)
    return values

def get_equality_cpd(U, info_cpd, control_cpd):
    #evidence = cid.get_parents(U)
    evidence = [info_cpd.variable, control_cpd.variable]
    #evidence_card = [cpds[e].variable_card for e in evidence]
    evidence_card = [info_cpd.variable_card, control_cpd.variable_card]
    parent_card = info_cpd.variable_card
    parent_names = [info_cpd.variable, control_cpd.variable]
    parent_idxs = np.where(np.array([name in parent_names for name in evidence]))
    values = __get_equality_values(parent_card)
    #values = __project_values2(evidence_card, parent_idxs)
    cpd = TabularCPD(variable=U,
            variable_card=2,
            evidence=evidence,
            evidence_card=evidence_card,
            values=values
            )
    return cpd





def _combine_values(A, B):
    #take a product of two probability tables
    if len(A.shape)==1:
        A = A[:,np.newaxis]
    if len(B.shape)==1:
        B = B[:,np.newaxis]
    C = np.array([np.outer(A[:,i], B[:,i]).ravel() for i in range(A.shape[1])]).T
    return C


def merge_cpds(node_cpds, child_cpds): #TODO: finish this
    #node_cpds are a list of cpds
    for cpd in node_cpds:
        assert cpds[0].variable==cpd.variable

    #variable = cpd1.variable
    #variable_card = cpd1.variable_card * cpd2.variable_card
    #state_names = {variable:list(zip(cpd1.state_names[variable],
    #                                cpd2.state_names[variable]))}
    #values = _combine_values(cpd1.get_values(), cpd2.get_values())
    #evidence = cpd1.get_evidence()
    #evidence_card = cpd1.cardinality[1:]
    #cpd3 = TabularCPD(variable=variable,
    #                 variable_card=variable_card,
    #                 values=values,
    #                  evidence=evidence,
    #                  evidence_card=evidence_card,
    #                 state_names=state_names)
    
    return cpds, child_cpds

