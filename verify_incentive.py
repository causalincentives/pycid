#Licensed to the Apache Software Foundation (ASF) under one or more contributor license 
#agreements; and to You under the Apache License, Version 2.0.

def verify_incentive(cid, D, X):
    new_cid = cid.copy()
    Dcpds = new_cid.solve()
    new_cid.add_cpds(*Dcpds.values())
    ev1 = new_cid.expected_utility({})
    new_cid = cid.copy()
    new_cid.remove_edge(X, D)
    Dcpds = new_cid.solve()
    new_cid.add_cpds(*Dcpds.values())
    ev2 = new_cid.expected_utility({})
    return ev1, ev2
