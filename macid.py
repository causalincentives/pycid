#Licensed to the Apache Software Foundation (ASF) under one or more contributor license 
#agreements; and to You under the Apache License, Version 2.0.

import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.factors.discrete import TabularCPD
import logging
import typing
from typing import List, Tuple, Dict
import itertools
from pgmpy.inference import BeliefPropagation
import functools
import networkx as nx
from cpd import NullCPD
import matplotlib.pyplot as plt
import operator
from collections import defaultdict


class MACID(BayesianModel):
    def __init__(self, ebunch:List[Tuple[str, str]]=None, node_types:Dict=None, decision_cardinalities:Dict=None, utility_values:Dict=None ):
        super(MACID, self).__init__(ebunch=ebunch)
        self.node_types = node_types
        self.decision_cardinalities = decision_cardinalities
        self.utility_values = utility_values
        self.utility_nodes = dict((i, node_types[i]['U']) for i in node_types if i != 'C')     # this gives a dictionary matching each agent with their decision and utility nodes
        self.decision_nodes = dict((i, node_types[i]['D']) for i in node_types if i != 'C')     #  eg {'A': ['U1', 'U2'], 'B': ['U3', 'U4']}
        self.chance_nodes = node_types['C']     # list of chance nodes
        self.agents = [agent for agent in node_types if agent != 'C']   # gives a list of the MAID's agents
        self.all_utility_nodes = list(itertools.chain(*self.utility_nodes.values()))        
        self.all_decision_nodes = list(itertools.chain(*self.decision_nodes.values()))        
   
        #self.ordering = self.get_acyclic_topological_ordering()       # ordering in which you should consider the decisions
        #self.reversed_ordering = list(reversed(self.ordering))     ## clearly need to change these!
        #self.numDecisions = len(self.ordering)


        print(f"This example has {len(self.agents)} agents")
        print("loaded")
        

    def copy(self):
        model_copy = MACID(node_types=self.node_types)
        model_copy.add_nodes_from(self.nodes())
        model_copy.add_edges_from(self.edges())
        if self.cpds:
            model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        return model_copy


    def add_cpds(self, *cpds):
        # this method adds conditional probability distributions to the MACID.
        for cpd in cpds:
            if not isinstance(cpd, (TabularCPD, ContinuousFactor, NullCPD)):
                raise ValueError("Only TabularCPD, ContinuousFactor, or NullCPD can be added.")

            if set(cpd.scope()) - set(cpd.scope()).intersection(set(self.nodes())):
                raise ValueError("CPD defined on variable not in the model", cpd)

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    logging.warning(
                        "Replacing existing CPD for {var}".format(var=cpd.variable)
                    )
                    self.cpds[prev_cpd_index] = cpd
                    break
            else:
                self.cpds.append(cpd)


    def check_model(self, allow_null=True):
        """
        Check the model for various errors. This method checks for the following
        errors.

        * Checks if the sum of the probabilities for each state is equal to 1 (tol=0.01).
        * Checks if the CPDs associated with nodes are consistent with their parents.

        Returns
        -------
        check: boolean
            True if all the checks are passed
        """
        for node in self.nodes():
            cpd = self.get_cpds(node=node)

            if cpd is None:
                raise ValueError("No CPD associated with {}".format(node))
            elif isinstance(cpd, (TabularCPD, ContinuousFactor)):
                evidence = cpd.get_evidence()
                parents = self.get_parents(node)
                if set(evidence if evidence else []) != set(parents if parents else []):
                    raise ValueError(
                        "CPD associated with {node} doesn't have "
                        "proper parents associated with it.".format(node=node)
                    )
                if not cpd.is_valid_cpd():
                    raise ValueError(
                        "Sum or integral of conditional probabilites for node {node}"
                        " is not equal to 1.".format(node=node)
                    )
            elif isinstance(cpd, (NullCPD)):
                if not allow_null:
                    raise ValueError(
                        "CPD associated with {node} is nullcpd".format(node=node)
                    )
        return True



    def random_instantiation_dec_nodes(self):
        #imputes random uniform policy to all decision nodes (NullCPDs) - arbitrary fully mixed strategy profile for MACID   #perhaps add something checking whether it's "isinstance(cpd, NullCPD)" is true
        for dec in self.decision_cardinalities:
            dec_card = self.decision_cardinalities[dec]
            parents = self.get_parents(dec)
            parents_card = [self.get_cardinality(par) for par in parents]
            table = np.ones((dec_card, np.product(parents_card).astype(int))) / dec_card
            uniform_cpd = TabularCPD(variable=dec, variable_card=dec_card,
                            values=table, evidence=parents,
                            evidence_card=parents_card
                            )
            self.add_cpds(uniform_cpd)
    
        print("added decision probs --")
        

    def _get_dec_agent(self, dec):
        # finds which agent this decision node belongs to
        for agent, decisions in self.decision_nodes.items():
            if dec in decisions:
                return agent

    def _get_util_agent(self, util):
        # finds which agent this utility node belongs to
        for agent, utilities in self.utility_nodes.items():
            if util in utilities:
                return agent


    def _get_max_child(self, row, col, decArray, bp):
        # selects the child (ie the action at that dection) which will give maximum EV for the agent making that decision.
        # updates the decision-array with the list of optimal descendent decisions.
        l = []
        dec_num_act = self.decision_cardinalities[self.reversed_ordering[row]]  # number of possible actions for that decision
        for indx in range (col*dec_num_act, (col*dec_num_act)+dec_num_act):   # *dec_num_act and +dec_num_act to iterate over actions they're considering  #FIX
            l.append(self._get_ev(decArray[row+1][indx], row, bp))  
        maxl = max(l)
        max_index = l.index(maxl)
        return decArray[row+1][(col*dec_num_act)+max_index]   

    def _get_ev(self, dec_list, row, bp):
        # returns the expected value of that decision for the agent making the decision
        dec = self.reversed_ordering[row]   #gets the decision being made on this row
        agent = self._get_dec_agent(dec)      #gets the agent making that decision
        # self.random_instantiation_dec_nodes()
        # bp = BeliefPropagation(self)
        h = bp.query(variables=self.all_utility_nodes, evidence=dict(zip(self.reversed_ordering, dec_list)))  #FIX this needs to generalise for all utilities and decisions
        ev = 0
        for idx, prob in np.ndenumerate(h.values):
                if prob != 0:
                    ev += prob*self.utility_values[agent][idx[agent-1]]     #(need agent -1 because idx starts from 0, but agents starts from 1)
        return ev

    def get_KM_NE(self):
        #returns the solution to Koller and Milch (2001) algorithm 6.1 when the macid's strategic relevance graph is acyclic
        self.random_instantiation_dec_nodes()    #uniform random mixed strategy profile for M
        bp = BeliefPropagation(self)  #instantiates belief propogation class

        action_space_list = list(itertools.accumulate(self.decision_cardinalities.values(), operator.mul))  #gives number of pure strategies for each decision node (ie taking into account prev decisions)
        cols_in_decArray_row = [1] + action_space_list


        actions_for_dec_list = []
        for card in self.decision_cardinalities.values():
            actions_for_dec_list.append(list(range(card)))     # appending the range of actions each decion can make as a list
        final_row_actions = list(itertools.product(*actions_for_dec_list))     # creates entry for final row of decision array (cartesian product of actions_seq_list))

        decArray = defaultdict(dict)   # creates a nested dictionary
        for i in range(0, self.numDecisions+1):
            for j in range(cols_in_decArray_row[i]):     #initialises entire decision/action array with empty tuples.
                decArray[i][j] = ()

        for i in range(cols_in_decArray_row[-1]):
            decArray[self.numDecisions][i] = final_row_actions[i]

        print(f"starting decision array for this MACID is {decArray}")   

        decArray_rows = len(self.ordering) + 1
        for row in range(decArray_rows -2, -1,-1):
            for col in range (0, cols_in_decArray_row[row]):
                decArray[row][col] = self._get_max_child(row, col, decArray, bp)
                #print(f"after update for row {row} and col {col} numArray is {decArray}")

        print(f"\nKM_NE = {dict(zip(self.reversed_ordering, decArray[0][0]))}")
        print(f"final decision array for this MACID is {decArray}")
     

    def _is_s_reachable(self, dec_pair):
        """ - dec_pair is a list of two deicsion nodes ie ['D1', 'D2']
            - this method determines whether 'D2' is s-reachable from 'D1'
        
        A node D2 is s-reachable from a node D1 in a MACID M if there is some utility node U ∈ U_D
        such that if a new parent D2' were added to D2, there would be an active path in M from
        D2′ to U given Pa(D)∪{D}, where a path is active in a MAID if it is active in the same graph, viewed as a BN.


        """
        self.add_edge('temp_par', dec_pair[1])
        agent = self._get_dec_agent(dec_pair[0])
        agent_utilities = self.utility_nodes[agent]
        con_nodes = [dec_pair[0]] + self.get_parents(dec_pair[0]) 
        if any([self.is_active_trail('temp_par', u_node, con_nodes) for u_node in agent_utilities]):
            self.remove_node('temp_par')
            print("yes")
            return True
        else:
            self.remove_node('temp_par')
            print("no")
            return False

    def strategic_rel_graph(self):
        # finds the strategic relevance graph of the MAID
        # an edge D' -> D exists iff D' is s-reachable from D
        G = nx.DiGraph()
        dec_pair_perms = list(itertools.permutations(self.all_decision_nodes, 2))
        for dec_pair in dec_pair_perms:
            if self._is_s_reachable(dec_pair):
                G.add_edge(dec_pair[1], dec_pair[0])
        return G

    def draw_strategic_rel_graph(self):
        # draws a MACID's strategic relevance graph
        rg = self.strategic_rel_graph()
        nx.draw_networkx(rg, with_labels=True)
        plt.figure(0)
        plt.draw()
        

    def strategically_acyclic(self):
        #finds whether the MACID has an acyclic strategic relevance graph.
        rg = self.strategic_rel_graph()
        if nx.is_directed_acyclic_graph(rg):
            return True
        else:
            return False

    def get_acyclic_topological_ordering(self):
        # first checks whether the strategic relevance graph is acyclic
        # returns a topological ordering (which might not be unique) of the decision nodes
        rg = self.strategic_rel_graph()
        if self.strategically_acyclic():
            return list(nx.topological_sort(rg))
        else:
            return 'The strategic relevance graph for this MACID is not acyclic and so \
                        no topological ordering can be given'


    def _get_color(self, node):
    # This colour codes the decision, chance and utility nodes of each agent
        for i in self.node_types:
            if i == 'C':
                if node in self.node_types['C']:
                    return 'lightgray'
            else:
                if node in self.node_types[i]['D']:
                    return 'lightblue'
                if node in self.node_types[i]['U']:
                    return 'yellow'

    def draw(self):
    # This draws the DAG for the CID
        l = nx.kamada_kawai_layout(self)
        colors = [self._get_color(node) for node in self.nodes]
        nx.draw_networkx(self, pos=l, node_color=colors)






# def _get_color(self, node):
    # # This colour codes the decision, chance and utility nodes of each agent
    #     for i in self.node_types:
    #         if node in self.node_types[i]['D']:
    #             return 'lightblue'
    #         if node in self.node_types[i]['U']:   ## FIX!!!
    #             return 'yellow'
    #         if node in self.node_types[i]['C']:
    #             return 'lightgray'

