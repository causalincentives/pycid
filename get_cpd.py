#Licensed to the Apache Software Foundation (ASF) under one or more contributor license 
#agreements; and to You under the Apache License, Version 2.0.
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from cid import NullCPD
from typing import List
from functools import reduce

#def get_H_cpd(sys_cpds, systems, h_pointer):
#    H_cpd = sys_cpds[h_pointer[0]][h_pointer[1]]
#    hist = systems[h_pointer[0]]
#    H = hist[np.where(np.array(hist[h_pointer[1]])==paths[paths['i_C']])[0][0]-1]
#    H_cpd = systems[h_pointer[0]][h_pointer[1]]
#    return H_cpd


def __get_identity_values(size):
    values = np.eye(size)
    return values

def get_random_cpd(variable, variable_card):
        return TabularCPD(
                variable=variable,
                variable_card=variable_card,
                evidence=[],
                evidence_card=[],
                values=np.array(np.tile(1/variable_card,(1,variable_card)))
                )

def get_identity_cpd(cid, node, parent_cpd, path_name):
    if node in cid._get_decisions():
        if parent_cpd:
            variable_card = parent_cpd.variable_card
        else:
            variable_card = 2
        cpd = NullCPD(path_name+(node,), variable_card)
    else:
        evidence_card = [parent_cpd.variable_card]
        variable_card = parent_cpd.variable_card
        if parent_cpd:
            values = __get_identity_values(parent_cpd.variable_card)
        else:
            variable_card = []
            values = np.array([[0.5, 0.5]])

        cpd = TabularCPD(variable=path_name + (node,),
                        variable_card=variable_card,
                         evidence=[parent_cpd.variable],
                         evidence_card=evidence_card,
                         values=values,
                        )
    return cpd

def __recursive_project(dims, axes, values):
    if not dims:
        return values
    dims = dims.copy()
    dim = dims.pop(0)
    if axes and axes[0]==0:
        axes = axes.copy()
        axes.pop(0)
        axes = [a-1 for a in axes]
        return np.array([__recursive_project(dims, axes, v) for v in values]).ravel()
    else:
        axes = axes.copy()
        axes = [a-1 for a in axes]
        return np.tile(__recursive_project(dims, axes, values), dim)

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
    assert len(values.shape)==2
    #projects a TabularCPD values table `values` to `dims` dimensions
    #where `axes` denotes the original dimensions
    return np.array([project_row(dims, axes, v) for v in values])

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

def get_xor_cpd(node, cpd1, cpd2, path_name):
    evidence = [cpd1.variable_name, cpd2.variable_name]
    evidence_card = [cpd1.variable_card, cpd2.variable_card]
    values = __get_xor_values(cpd1.variable_card)
    cpd = TabularCPD(variable=path_name + (node,),
            variable_card=evidence_card,
            evidence=evidence,
            evidence_card=evidence_card,
            values=values
            )
    return cpd

def __get_func_values(h_card):
    hdim = np.log2(h_card).astype(int)
    funcsize = 2**hdim
    bitstrings = [np.binary_repr(i, funcsize+hdim) for i in np.arange(2**(funcsize+hdim))]
    funcs = [b[:-hdim] for b in bitstrings]
    hists = [b[-hdim:] for b in bitstrings]
    indices = [int(h, 2) for h in hists]
    values = np.array([f[i] for f, i in zip(funcs, indices)]).astype(int)
    value_table = np.eye(2)[values].T
    return value_table

def get_func_cpd(node, f_cpd, h_cpd, path_name):
    evidence = [f_cpd.variable, h_cpd.variable]
    evidence_card = [f_cpd.variable_card, h_cpd.variable_card]
    values = __get_func_values(h_cpd.variable_card)
    cpd = TabularCPD(variable=path_name +(node,),
            variable_card=2,
            evidence=evidence,
            evidence_card=evidence_card,
            values=values
            )
    return cpd

def __get_equals_func_values(h_card):
    hdim = np.log2(h_card).astype(int)
    funcsize = 2**hdim
    bitstrings = [np.binary_repr(i, hdim+funcsize+hdim) for i in np.arange(2**(hdim+funcsize+hdim))]
    funcs = [b[:-2*hdim] for b in bitstrings]
    args = [b[-2*hdim:-hdim] for b in bitstrings]
    outputs = [b[-hdim:] for b in bitstrings]
    indices = [int(a, 2) for a in args]
    values = np.array([f[i]==o for o, f, i in zip(outputs, funcs, indices)])
    value_table = np.array([1-values, values])
    return value_table


def get_equals_func_cpd(node, arg_cpd, f_cpd, val_cpd, path_name):
    if arg_cpd:
        evidence = [f_cpd.variable, arg_cpd.variable, val_cpd.variable]
        evidence_card = [f_cpd.variable_card, arg_cpd.variable_card, val_cpd.variable_card]
        values = __get_equals_func_values(val_cpd.variable_card)
    else:
        evidence = [f_cpd.variable, val_cpd.variable]
        evidence_card = [f_cpd.variable_card, val_cpd.variable_card]
        values = __get_equals_func_values(val_cpd.variable_card//2)
    cpd = TabularCPD(variable=path_name +(node,),
            variable_card=2,
            evidence=evidence,
            evidence_card=evidence_card,
            values=values
            )
    return cpd









    

def __get_equality_values(dim):
    return np.array((1-np.eye(dim).ravel(), np.eye(dim).ravel()))

#def __project_values2(dims, axes, values): #TODO: finish this
#    for axis in axes:
#        A = np.ones((1, np.product(dims[:axis])))
#        X = np.kron(A, X)
#
#
#    A = np.ones((1,np.product(dims[:axis])))
#    C = np.ones((1, np.product(dims[axis+1:])))
#    values = np.kron(np.kron(A, B), C)
#    return values

def get_equality_cpd(U, info_cpd, control_cpd, path_name):
    #evidence = cid.get_parents(U)
    evidence = [info_cpd.variable, control_cpd.variable]
    #evidence_card = [cpds[e].variable_card for e in evidence]
    evidence_card = [info_cpd.variable_card, control_cpd.variable_card]
    parent_card = info_cpd.variable_card
    parent_names = [info_cpd.variable, control_cpd.variable]
    parent_idxs = np.where(np.array([name in parent_names for name in evidence]))
    values = __get_equality_values(parent_card)
    #values = __project_values2(evidence_card, parent_idxs)
    cpd = TabularCPD(variable=path_name + (U,),
            variable_card=2,
            evidence=evidence,
            evidence_card=evidence_card,
            values=values
            )
    return cpd





def _merge_values(A, B):
    #take a product of two prob tables, giving a prob table of 
    #dims [A.shape[0]*B.shape[0], A.shape[1]]
    if len(A.shape)==1:
        A = A[:,np.newaxis]
    if len(B.shape)==1:
        B = B[:,np.newaxis]
    C = np.array([np.outer(A[:,i], B[:,i]).ravel() for i in range(A.shape[1])]).T
    return C


def get_cpds(flat_cpds, name):
    return [i for i in flat_cpds if i.variable[2]==name]

def merge_children(cpd):
    merging = [i for i in cpd.get_evidence() if i[2]==name]
    non_merging = [i for i in cpd.get_evidence() if i[2]!=name]
    cpd.reorder_parents(merging+non_merging, inplace=True)
    dims = node_cards + [cpd.cardinality[1+len(merging):]]
    values = project_values(dims, np.arange(1+len(merging),len(cpd.get_evidence())), 
                            cpd.get_values())
    
    return cpd
    
def get_names(names, allow_singles=True):
    if allow_singles:
        out = []
        for i in names:
            if isinstance(i, (tuple, list)):
                out.append(i[2])
            else:
                out.append(i)
        return out
    else:
        return [i[2] for i in names]

#def to_single_names(names):
#    return [i[2] if isinstance(i, list) else i for i in names]

def utility_values(values):
    assert len(values.shape)==2
    height = values.shape[0]
    val_of_row = np.array([np.binary_repr(i).count('1') for i in range(height)])
    val_idx = [np.where(val_of_row==i)[0] for i in np.arange(np.log2(height)+1)]
    new_values = np.array([values[i].sum(axis=0) for i in val_idx])
    return new_values

def merge_node(cid, merged_cpds, flat_cpds, name):
    #merges cpds to fit cid
    node_cpds = get_cpds(flat_cpds, name)
    graphical_children = [i for i in flat_cpds if i.variable[2] in cid.get_children(name)]
    child_cpds = [i for i in graphical_children if isinstance(i, NullCPD) or name in get_names(i.get_evidence())]
    
    #make merged cpd for`name`
    node_names = [W.variable for W in node_cpds]
    node_cards = [W.variable_card for W in node_cpds]
    variable_card = np.product(node_cards, dtype=int)
    evidence = cid.get_parents(name)
    evidence_card = [merged_cpds[k].variable_card if k in merged_cpds else 1 for k in evidence] 
    
    #project each cpd onto set of actual parents
    if name in cid._get_decisions():
        assert isinstance(variable_card, (int, np.int64))
        new_cpd = NullCPD(variable=name, variable_card = variable_card)
    else:
        all_values = []
        for node_cpd in node_cpds:
            assert np.all([isinstance(i, str) for i in node_cpd.get_evidence()])
            useful_idx = [i for i in range(len(evidence)) if evidence[i] in node_cpd.get_evidence()]
            useful_parents = [evidence[i] for i in useful_idx]
            if evidence_card:
                if useful_parents:
                    node_cpd.reorder_parents(useful_parents, inplace=True)
                projection = project_values(evidence_card, useful_idx, node_cpd.get_values())
                all_values.append(projection)
            else:
                #evidence_card = []
                all_values.append(node_cpd.get_values())
            #if node_cpd.variable[2]=='S0':
            #    import ipdb; ipdb.set_trace()
        
        #merge the values
        if all_values:
            values = reduce(_merge_values, all_values)
        else:
            values = np.ones([1, np.product(evidence_card, dtype=int)])

        if name in cid.utilities:
            variable_card = int(np.log2(variable_card)+1)
            values = utility_values(values)
        
        new_cpd = TabularCPD(
                        variable=name,
                        variable_card=variable_card,
                        evidence=evidence,
                        evidence_card=evidence_card,
                        values=values.reshape(1, -1)
                        )


    
    #change children
    new_children = []
    for child_cpd in child_cpds:
        if isinstance(child_cpd, TabularCPD):
            old_evidence = child_cpd.get_evidence()
            merging = [i for i in old_evidence if isinstance(i, tuple) and i[2]==name]
            non_merging = [i for i in old_evidence if i not in merging]
            child_cpd.reorder_parents(merging+non_merging, inplace=True)
            node_dims = sorted([np.where([i==W for i in node_names])[0][0] for W in merging]) #TODO: is sorted correct?
            dims = node_dims + list(range(len(node_cards),len(node_cards)+len(non_merging)))
            assert len(set(dims))==len(dims)
            evidence_card_unflat = node_cards + child_cpd.cardinality.tolist()[1+len(merging):]
            new_values = project_values(evidence_card_unflat, dims, child_cpd.get_values())
            if old_evidence:
                evidence_card = [np.product(node_cards)] + evidence_card_unflat[len(node_cards):]
                new_evidence = ([merging[0][2]] if merging else []) + non_merging
            else:
                evidence_card = None
                new_evidence = None
            
            new_children.append(TabularCPD(
                variable=child_cpd.variable,
                evidence=new_evidence,
                variable_card= child_cpd.variable_card,
                evidence_card=evidence_card,
                values = new_values            
            ))
        elif isinstance(child_cpd, NullCPD):
            new_children.append(child_cpd)
        
    #remove `name` from flat_cpds
    #update parents and values of child nodes
    merged_cpds[new_cpd.variable] = new_cpd
    flat_cpds = [i for i in flat_cpds if i.variable[2]!=name and i not in child_cpds] + new_children
    return merged_cpds, flat_cpds, new_cpd, new_children


