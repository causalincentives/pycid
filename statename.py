#Licensed to the Apache Software Foundation (ASF) under one or more contributor license
#agreements; and to You under the Apache License, Version 2.0.

from __future__ import annotations
from typing import Union
class StateName(tuple):
    def __new__(self, name:str, contents:Union[StateName,None]=None):
        if contents:
            sn = super().__new__(self, (name, contents))
            sn.length = sum([i.length for i in contents])
        else:
            sn = super().__new__(self, (name,))
            sn.length = 1
        sn.name = name
        return sn
    def __flatten(self, l):
        for i in l:
            if isinstance(i, (list,tuple)):
                for j in i:
                    yield j
            else:
                yield i
    def flatten(self):
        return self.__flatten(self)

    def _index_recurse(self, name, counter):
        if self.name==name:
            return counter, counter+self.length
        else:
            for i in self[:-1]:
                if i.name==name:
                    return counter, counter+self.length
                else:
                    counter += i.length
            return i._start_index_recurse(name, counter)

    def get_index(self, name):
        endpoints = self._index_recurse(name, 0)
        return slice(*endpoints)
