#!/usr/bin/env python
# coding: utf-8

# In[6]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#%pdb
import sys
sys.path.append("..") #TODO: make /cid a package, then delete this
from examples import get_3node_cid, get_5node_cid, get_2dec_cid, get_nested_cid
from parameterize import parameterize_systems, merge_all_nodes
from generate import random_cids
from get_systems import choose_systems, check_systems
#TODO: choose i_C for first infolink automatically
#TODO: add the collider->obs paths for colliders


# In[ ]:


#rand_cids = random_cids(n_cids=3000, ns_range = (11, 12), nu_range=(1, 1), nd_range=(2, 2), edge_density=.3)
#cids = [r.trimmed() for r in rand_cids]

#for cid in cids:
#    D = 'D0'
#    if cid.get_parents(D):
#        X = cid.get_parents(D)[0]
#        all_paths = choose_systems(cid, D, X)
#        cid.draw()


# In[2]:


from examples import get_nested_cid
cid = get_nested_cid()
cid.draw()
systems = choose_systems(cid, 'D1', 'S1')
systems[0]['i_C']=0
print(systems)


# In[3]:


all_cpds = parameterize_systems(cid, systems)
all_cpds = [c.copy() for a in all_cpds for b in a.values() for c in b.values()]


# In[4]:


merged_cpds = merge_all_nodes(cid, all_cpds)
merged_cpds


# In[5]:


cid.add_cpds(*merged_cpds.values())

new_cid = cid.copy()
Dcpds = new_cid.solve()
new_cid.add_cpds(*Dcpds.values())
ev1 = new_cid.expected_utility({})

new_cid = cid.copy()
new_cid.remove_edge('S1', 'D1')
Dcpds = new_cid.solve()
new_cid.add_cpds(*Dcpds.values())
ev2 = new_cid.expected_utility({})

print(ev1, ev2)


# In[9]:


#{(i,j):(k,new_cid._query(new_cid.utilities,{'S4':i,'S3':j}).values) for (i,j),k in zip(
#    [(a,b) for a in range(16) for b in range(4)], 
#    d2cpd.get_values().argmax(axis=0)) if new_cid._query(new_cid.utilities,{'S4':i,'S3':j}).values.sum()}

