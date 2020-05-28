from pgmpy.models import BayesianModel
import networkx as nx
import copy
from itertools import compress
from typing import List


##
class Information:
    """Single agent Information incentive Class
    Criterion for Information incentive on X:
    (i) X is d-connected to U | Fa_D\{X}
    (ii) U∈Desc(D) (U must be a descendent of D)
    """

    def __init__(self, model, decision_nodes, utility_nodes):
    # pass in model, decision nodes and utility nodes of agent being considered    
        try:
            assert isinstance(model, BayesianModel)
        except AssertionError:
            raise AssertionError("Observation instantiation's first argument must be a BayesianModel")

        self.dag = model
        self.graph = self.dag.to_directed()
        self.decision_nodes = decision_nodes
        self.utility_nodes = utility_nodes
        self.all_nodes = self.dag.nodes

        try:
            assert set(self.decision_nodes).issubset(set(self.all_nodes))
            assert set(self.utility_nodes).issubset(set(self.all_nodes))
        except AssertionError:
            raise AssertionError("Some or all of the Decision and Utility nodes are not present on the graph")

    # def __repr__(self):  #maybe create this function
    #     return

    def inf_node(self, node):
        # returns True if a node faces an observation incentive
        try:
            assert self.utility_nodes[0] in nx.descendants(self.dag, self.decision_nodes[0])
        except AssertionError:
            raise AssertionError(
                f"utility node {self.utility_nodes[0]} is not a descendent of decision node {self.decision_nodes[0]}")

        if node not in self.utility_nodes + self.decision_nodes:
            con_nodes = self.decision_nodes + self.dag.get_parents(self.decision_nodes[0])  # nodes to be conditioned on
            if node in con_nodes:  # remove node from condition nodes
                con_nodes.remove(node)
            if self.dag.is_active_trail(node, self.utility_nodes[0], con_nodes):
                return True
            else:
                return False

    def all_inf_inc_nodes(self):
        # returns all nodes in the DAG that the the decision maker has an observation incentive for.
        inf_list = []
        for x in list(self.dag.nodes):
            if self.inf_node(x):
                inf_list.append(x)

        return f"The nodes which are facing an information incentive from this agent are {inf_list}"


##

class Response:
    """Response incentive Class
    Criterion for response incentive on X: 
    (i) there is a directed path from X to an observation W ∈ Pa_D 
    (ii) U∈Desc(D) (U must be a descendent of D)
    (i) W is d-connected to U | Fa_D \ {W}
    """

    def __init__(self, model, decision_nodes, utility_nodes):
        # pass in model, decision nodes and utility nodes of agent being considered
        try:
            assert isinstance(model, BayesianModel)
        except AssertionError:
            raise AssertionError("Response instantiation's first argument must be a BayesianModel")

        self.dag = model
        self.graph = self.dag.to_directed()
        self.decision_nodes = decision_nodes
        self.utility_nodes = utility_nodes
        self.all_nodes = self.dag.nodes

        try:
            assert set(self.decision_nodes).issubset(set(self.all_nodes))
            assert set(self.utility_nodes).issubset(set(self.all_nodes))
        except AssertionError:
            raise AssertionError("Some or all of the Decision and Utility nodes are not present on the graph")

    def all_res_inc_nodes(self):
        # returns all nodes in the DAG that the the decision maker has a response incentive for.
        res_list = []
        try:
            assert self.utility_nodes[0] in nx.descendants(self.dag, self.decision_nodes[0])
        except AssertionError:
            raise AssertionError(
                f"utility node {self.utility_nodes[0]} is not a descendent of decision node {self.decision_nodes[0]}")

        w_set = self.dag.get_parents(self.decision_nodes[0])
        for w in w_set:
            for x in list(self.dag.nodes):
                if w == x or w in nx.descendants(self.dag, x):  # condition (i)
                    w_set2 = copy.deepcopy(w_set)
                    w_set2.remove(w)
                    con_nodes = w_set2 + self.decision_nodes
                    if self.dag.is_active_trail(w, self.utility_nodes[0], con_nodes):
                        res_list.append(x)

        return res_list

    def res_node(self, node):
        # returns True if a node faces a response incentive
        if node in self.all_res_inc_nodes():
            return True
        else:
            return False


##
class Influence:
    """Influence incentive Class
    Criterion for Influence incentive on X: 
    (*) A directed path D ---> X ---> U exists
    """


    def __init__(self, model, decision_nodes, utility_nodes):
        # pass in model, decision nodes and utility nodes of agent being considered
        try:
            assert isinstance(model, BayesianModel)
        except AssertionError:
            raise AssertionError("Influence instantiation's first argument must be a BayesianModel")

        self.dag = model
        self.graph = self.dag.to_directed()
        self.decision_nodes = decision_nodes
        self.utility_nodes = utility_nodes
        self.all_nodes = self.dag.nodes

        try:
            assert set(self.decision_nodes).issubset(set(self.all_nodes))
            assert set(self.utility_nodes).issubset(set(self.all_nodes))
        except AssertionError:
            raise AssertionError("Some or all of the Decision and Utility nodes are not present on the graph")

    def all_infl_inc_nodes(self):
        con_list = []
        for x in list(self.dag.nodes):
            if self.con_node(x):
                con_list.append(x)
        return con_list

    def infl_node(self, node):
        # returns True if a node faces an influence incentive
        d_u_paths = Paths(self.dag).find_all_dir_path(self.decision_nodes[0], self.utility_nodes[0])
        if node not in self.decision_nodes:
            if any(node in paths for paths in d_u_paths):
                return True
            else:
                return False

##

class Control:
    """Control incentive Class
    Criterion for a control incentive on X: 
    (i) X is not a decision node
    (ii) iff there is a directed path X --> U in the reduced graph G*

    The reduced graph G* of a MACID G is the result of removing from G information links Y -> D from all non-requisite 
    observations Y ∈ Pa_D

    this control incentive is:
    A) Direct: if the directed path X --> U does not pass through D
    B) Indirect: if the directed path X --> U does pass through D and there is a backdoor path X -- U that begins 
    backwards from X(···←X) and is active when conditioning on Fa_D \ {X}
    """


    def __init__(self, model, decision_nodes, utility_nodes):
        # pass in model, decision nodes and utility nodes of agent being considered
        try:
            assert isinstance(model, BayesianModel)
        except AssertionError:
            raise AssertionError("Control instantiation's first argument must be a BayesianModel")

        self.dag = model
        self.graph = self.dag.to_directed()
        self.decision_nodes = decision_nodes
        self.utility_nodes = utility_nodes
        self.all_nodes = self.dag.nodes

        try:
            assert set(self.decision_nodes).issubset(set(self.all_nodes))
            assert set(self.utility_nodes).issubset(set(self.all_nodes))
        except AssertionError:
            raise AssertionError("Some or all of the Decision and Utility nodes are not present on the graph")

    def con_node(self, node):
        # returns True if a node faces a control incentive
        trimmed_dag = TrimGraph(self.dag, self.decision_nodes, self.utility_nodes).trim()
        if node != self.decision_nodes[0]:
            if node == self.utility_nodes[0] or self.utility_nodes[0] in nx.descendants(trimmed_dag, node):
                return True
            else:
                return False

    def indir_con_node(self, node): #!!! need to update to this to reflect new def in UndAgInc paper (backdoor)
        # returns True if a node faces an intervention incentive for control
        trimmed_dag = TrimGraph(self.dag, self.decision_nodes, self.utility_nodes).trim()
        if self.int_node(node):
            x_u_paths = Paths(trimmed_dag).find_all_dir_path(node, self.utility_nodes[0])
            if any(self.decision_nodes[0] in paths for paths in x_u_paths):
                return True
            else:
                return False

    def dir_con_node(self, node):
        # returns True if a node faces a direct control incentive
        trimmed_dag = TrimGraph(self.dag, self.decision_nodes, self.utility_nodes).trim()
        if self.int_node(node):
            x_u_paths = Paths(trimmed_dag).find_all_dir_path(node, self.utility_nodes[0])
            for path in x_u_paths:
                if self.decision_nodes[0] not in path:
                    return True
            return False


    def all_cont_inc_nodes(self):
        con_list = []
        for x in list(self.dag.nodes):
            if self.con_node(x):
                con_list.append(x)
        return con_list

    def all_dir_con_inc_nodes(self):
        dir_con_list = []
        for x in list(self.dag.nodes):
            if self.dir_con_node(x):
                dir_con_list.append(x)
        return dir_con_list

    def all_indir_con_inc_nodes(self):
        indir_con_list = []
        for x in list(self.dag.nodes):
            if self.indir_con_node(x):
                indir_con_list.append(x)
        return indir_con_list

##
class Paths:
    # class for finding paths that exist in the MAID
    def __init__(self, model):
        try:
            assert isinstance(model, BayesianModel)
        except AssertionError:
            raise AssertionError("Argument must be a BayesianModel")
        # print(model.nodes)
        self.dag = model

    def _find_dirpath_recurse(self, path: List[str], finish: str, all_paths):

        if path[-1] == finish:
            return path
        else:
            children = self.dag.get_children(path[-1])
            for child in children:
                ext = path + [child]
                ext = self._find_dirpath_recurse(ext, finish, all_paths)
                if ext and ext[-1] == finish:  # the "if ext" checks to see that it's a full directed path.
                    all_paths.append(ext)
                else:
                    continue
            return all_paths

    def find_all_dir_path(self, start, finish):
        # finds all direct paths from start node to end node that exist in the MAID
        all_paths = []
        return self._find_dirpath_recurse([start], finish, all_paths)


##

class TrimGraph:
    #this class trims away irrelevant infromation links.

    def __init__(self, model, decision_nodes, utility_nodes):
        # pass in model, decision nodes and utility nodes of agent being considered
        try:
            assert isinstance(model, BayesianModel)
        except AssertionError:
            raise AssertionError("Control instantiation's first argument must be a BayesianModel")
        self.dag = model
        self.graph = self.dag.to_directed()
        self.decision_nodes = decision_nodes
        self.utility_nodes = utility_nodes
        self.all_nodes = self.dag.nodes

    def trim(self):
        # returns the DAG which has been trimmed of all irrelevant information links.
        d_par = self.dag.get_parents(self.decision_nodes[0])
        no_info_bool = [not Information(self.dag, self.decision_nodes, self.utility_nodes).inf_node(n) for n in d_par]
        d_par_no_inf = list(compress(d_par, no_info_bool))

        for node in d_par_no_inf:
            self.dag.remove_edge(node, self.decision_nodes[0])
        return self.dag
