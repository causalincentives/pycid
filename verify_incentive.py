#Licensed to the Apache Software Foundation (ASF) under one or more contributor license
#agreements; and to You under the Apache License, Version 2.0.

def verify_incentive(cid, D, X):
    new = cid.copy()
    Dcpds = new.solve()
    new.add_cpds(*Dcpds.values())
    ev1 = new.expected_utility({})
    new = cid.copy()
    new.remove_edge(X, D)
    Dcpds = new.solve()
    new.add_cpds(*Dcpds.values())
    ev2 = new.expected_utility({})
    return ev1, ev2
