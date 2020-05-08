#Licensed to the Apache Software Foundation (ASF) under one or more contributor license 
#agreements; and to You under the Apache License, Version 2.0.

from pgmpy.factors.base import BaseFactor

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
        return "<NullCPD {}:{}".format(self.variable, self.variable_card)
    #def to_factor(self):
    #    return self
