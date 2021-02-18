# ----------- Methods for converting MACID to EFG for Gambit to solve ---------



def _create_EFG_structure(self):
    """ Creates the EFG structure:
    1) Finds the MACID nodes needed for the EFG (this is the set {D ∪ Pa(D)})
    2) Creates an ordering of these nodes such that X_j precedes X_i in the ordering if and only if X_j
    is a descendant of X_i in the MACID.
    3) Labels each node X_i with a partial instantiation of the splits in the path to X_i in the EFG.
    """
    self.random_instantiation_dec_nodes()

    game_tree_nodes = self.all_decision_nodes + [parent for dec in self.all_decision_nodes for parent in self.get_parents(dec)]   # returns set {D ∪ Pa(D)}
    sorted_game_tree_nodes = [node for node in list(nx.topological_sort(self)) if node in game_tree_nodes] # sorting nodes in {D ∪ Pa(D)} into a topological ordering consistent with the graph
    numRows = len(sorted_game_tree_nodes) +1   # +1 means we have a row for the result after the final node has split

    cardinalities = map(self.get_cardinality, sorted_game_tree_nodes)
    node_cardinalities = dict(zip(sorted_game_tree_nodes, cardinalities))

    node_splits_list = list(itertools.accumulate(node_cardinalities.values(), operator.mul))  #gives number of unique paths to get to each row
    nodes_in_each_tree_row = [1] + node_splits_list   # 1 root node for the first row

    efg_structure = defaultdict(dict)   # creates a nested dictionary
    shell = defaultdict(dict)
    splits_at_each_node_list = []
    for card in node_cardinalities.values():
        splits_at_each_node_list.append(list(range(card)))

    efg_structure[0][0] = {}
    shell[0][0] = (0,0)

    for i in range(1, numRows):
        for j in range(nodes_in_each_tree_row[i]):
            splits = list(itertools.product(*splits_at_each_node_list[0:i]))[j]  # fills in the instantiation of splits in the path to get the j'th element along row i of the tree
            efg_structure[i][j] = dict(zip(sorted_game_tree_nodes, splits))
            shell[i][j] = (i,j)
    return efg_structure, sorted_game_tree_nodes, shell


def _add_utilities(self, tree, node_order):
    """
    adds the utilities as leaves in the EFG
    """
    bp = BeliefPropagation(self)
    for idx, leaf in enumerate(tree[len(tree)-1].values()):    # iterate over leaves of EFG structure
        tree['u'][idx] = self._get_leaf_utilities(leaf.values(), node_order, bp)
    return tree

def _get_leaf_utilities(self, node_selection, node_order, bp):
    # finds leaf utilities by querying (doing propabalistic inference) on the BN described by the MACID
    # utilities = {0:np.arange(6), 1:-np.arange(6)}    # this should come from the notebooks
    evidences = dict(zip(node_order, node_selection))
    leaf_utilities = []
    for agent in range(len(self.agents)):
        utils = self.all_utility_nodes[agent]       #gets the utility nodes for that agent
        h = bp.query(variables=utils, evidence=evidences)
        ev = 0
        for idx, prob in np.ndenumerate(h.values):
            for i in range(len(utils)):
                if prob != 0:
                    ev += prob*self.utility_domains[utils[i]][idx[i]]     #(need agent -1 because idx starts from 0, but agents starts from 1)
        leaf_utilities.append(ev)
    return leaf_utilities

def _get_leaf_utilities(self, node_selection, node_order, bp):
    # finds leaf utilities by querying (doing propabalistic inference) on the BN described by the MACID
    evidences = dict(zip(node_order, node_selection))
    leaf_utilities = []

    for agent in range(len(self.agents)):
        utils = self.all_utility_nodes[agent]       #gets the utility nodes for that agent

        h = bp.query(variables=utils, evidence=evidences)
        ev = 0
        for idx, prob in np.ndenumerate(h.values):
            for i in range(len(utils)):  #accounts for each agent potentially having multiple utility nodes
                if prob != 0:
                    ev += prob*self.utility_domains[utils[i]][idx[i]]     #(need agent -1 because idx starts from 0, but agents starts from 1)
        leaf_utilities.append(ev)
    return leaf_utilities

def _preorder_traversal(self, shell, node_order):
    """
    returns the EFG node location ordering consistent with a prefix-order traversal of the EFG tree
    """
    cardinalities = map(self.get_cardinality, node_order)
    node_cardinalities = dict(zip(node_order, cardinalities))

    stack = deque([])
    preorder = []
    preorder.append(shell[0][0])
    stack.append(shell[0][0])
    while len(stack)> 0:
        flag = 0   # checks wehther all child nodes have been visited
        if stack[len(stack)-1][0] == len(node_order):   # if top of stack is a leaf node, remove from stack
            stack.pop()
        else:    # consider case when top of stack is a parent with children
            par_row = stack[len(stack)-1][0]
            par_col = stack[len(stack)-1][1]
            num_child = node_cardinalities[node_order[par_row]]
        for child in range(par_col*num_child, (par_col*num_child)+num_child): #iterate through children
            if shell[par_row+1][child] not in preorder: #as soon as an unvisited child is found, push it to stack and add to preorder
                flag = 1
                stack.append(shell[par_row+1][child])
                preorder.append(shell[par_row+1][child])
                break    # start again in while loop to explore this new child
        if flag == 0:    # if all children of a parent have been visited, pop this node from the stack.
            stack.pop()
    return preorder

def _trim_efg(self, efg, node_order):
    """
    trims the EFG so at each efg node we only contain the instantiation of the splits made by the nodes' parents in the macid.
    This is necessary for determining the information sets.
    """
    for row in range(1,len(node_order)):
        for node_splits in efg[row].values():
            for node in list(node_splits.keys()):
                if node not in self.get_parents(node_order[row]):
                    del(node_splits[node])
    return efg


def _info_sets(self, trimmed_efg, node_order):
    """returns info sets for each row of the efg (we can consider rows individually because every EFG derived from a macid will be symmetric and have a row for each macid node)
    nodes in the same EFG information set are:
    1) labelled with the same partial instantiation of splits (accoding to their parent nodes in the MACID)
    2) The same actions available to them. """
    info_sets = {}
    for row in range(1,len(node_order)):
        node_splits = list(trimmed_efg[row].values())
        comparable_list = [str(split) for split in node_splits]
        _, info_set_nums = np.unique(comparable_list, return_inverse=True)
        info_sets[row] = list(info_set_nums+1)  # in gambit info sets must start at 1
    return info_sets

def _write_efg_file(self, macid_node_order, preorder, info_sets, efg):
    # writes EFG to a "macid_game.efg" file for GAMBIT. Can use GAMBIT's NE finders + GAMBIT's GUI for investigating game properties.
    cardinalities = map(self.get_cardinality, macid_node_order)
    node_cardinalities = dict(zip(macid_node_order, cardinalities))
    f = open("macid_game1.efg", "w")

    f.write("EFG 2 R \"Game\" { ")    #creates the necessary ".efg" header
    for i in range(len(self.agents)):
        f.write(f"\"player {i+1} \" ")
    f.write("}\n")

    terminal_inf_set = 0
    chance_inf_set = 0

    for node in preorder:     #creates the ".efg" body by iterating through nodes in a prefix-traversal ordering
        if node[0] < len(macid_node_order):
            macid_node = macid_node_order[node[0]]

            if self.get_node_type(macid_node)  == 'c':  #chance node
                bp = BeliefPropagation(self)
                probs = bp.query(variables=[macid_node], evidence=efg[node[0]][node[1]]).values
                chance_inf_set += 1
                f.write(f"c \"\" {chance_inf_set} \"\" ")
                f.write("{ ")
                for action in range(node_cardinalities[macid_node]):
                    f.write(f"\"{action}\" {probs[action]} ")
                f.write("} 0\n")

            elif self.get_node_type(macid_node)  == 'p':  #player node
                agent = self._get_dec_agent(macid_node)
                f.write(f"p \"\" {agent+1} ")
                f.write(f"{info_sets[node[0]][node[1]]+100*node[0]} \"\" ")   #messy way of handling the fact that a player could make multiple decisions => need unique info sets for each decision.
                f.write("{ ")
                for action in range(node_cardinalities[macid_node]):
                    f.write(f"\"{action}\" ")
                f.write("} 0\n")

        elif node[0] == len(macid_node_order):  #terminal nodes
            terminal_inf_set +=1
            f.write(f"t \"\" {terminal_inf_set} \"\" ")
            f.write("{ ")
            for util in efg['u'][node[1]]:
                f.write(f"{util}, ")
            f.write("}\n")

    f.close()


def MACID_to_Gambit_file(self):
    """
    Converts MACID to a ".efg" file for use with GAMBIT.
    """
    efg_structure, node_order, shell = self._create_EFG_structure()
    efg_with_utilities = self._add_utilities(efg_structure, node_order)
    preorder = self._preorder_traversal(shell, node_order)
    trimmed_efg = self._trim_efg(efg_with_utilities, node_order)
    info_sets = self._info_sets(trimmed_efg, node_order)
    self._write_efg_file(node_order, preorder, info_sets, efg_with_utilities)
    print("\nGambit .efg file has been created from the macid")
    return True
