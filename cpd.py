#Licensed to the Apache Software Foundation (ASF) under one or more contributor license
#agreements; and to You under the Apache License, Version 2.0.
import itertools
from typing import List

from pgmpy.factors.base import BaseFactor
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from cid import CID


class NullCPD(BaseFactor):
    def __init__(self, variable, variable_card, state_names={}):
        self.variable = variable
        self.variable_card = variable_card #is this correct?
        self.cardinality = [variable_card] #TODO: possible problem because this usually includes cardinality of parents
        self.variables = [self.variable]
        self.state_names = state_names

    def scope(self):
        return [self.variable]

    def copy(self):
        return NullCPD(self.variable, self.variable_card)

    def marginalize(self, variables, inplace=True): #TODO (maybe): decrease cardinality by len(variables)
        if not inplace:
            return self

    def __repr__(self):
        return "<NullCPD {}:{}>".format(self.variable, self.variable_card)
    #def to_factor(self):
    #    return self


class FunctionCPD:

    def __init__(self, name: str, card: int, f, evidence: List[str], evidence_card: List[int]):
        self.name = name
        self.card = card
        self.f = f
        self.evidence = evidence
        self.evidence_card = evidence_card

    def possible_values(self, cid: BayesianModel):
        parents = cid.get_parents(self.name)
        parent_values = []
        for p in parents:
            if p.state_names:
                parent_values.append(p.state_names)
            elif isinstance(p, FunctionCPD):
                parent_values.append(p.possible_values())
            else:
                raise Exception("unknown parent values for {}".format(p))
        self.poss_values = [self.f(*x) for x in itertools.product(parents)]


    def convert2TabularCPD(self, cid: BayesianModel):
        parents = cid.get_parents(self.name)


        ranges = [range(i) for i in evidence_card]
        # values = [(f(*i) for i in itertools.product(*ranges)]

        # for i in itertools.product(*ranges):
        #     print(i, dom2val_list(evidence_dom2val, i), f(*dom2val_list(evidence_dom2val, i)),
        #           val2dom(f(*dom2val_list(evidence_dom2val, i))),
        #           val2dom(f(*dom2val_list(evidence_dom2val, i)))==3)

        matrix = np.array([[int(val2dom(f(*dom2val_list(evidence_dom2val, i))) == t)
                            for i in itertools.product(*ranges)]
                           for t in range(card)])

        super(FunctionCPD, self).__init__(name, card, matrix, evidence, evidence_card)
