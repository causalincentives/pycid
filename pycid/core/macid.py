from __future__ import annotations

import copy
import itertools
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import networkx as nx
import pygambit
from pgmpy.factors.discrete import TabularCPD

from pycid.core.cpd import DecisionDomain, StochasticFunctionCPD
from pycid.core.macid_base import MACIDBase
from pycid.core.relevance_graph import CondensedRelevanceGraph


class MACID(MACIDBase):
    """A Multi-Agent Causal Influence Diagram"""

    def get_all_ne(self, solver: Optional[str] = "enumpure") -> List[List[StochasticFunctionCPD]]:
        """
        Return a list of Nash equilbiria in the MACID.

        - solver can be any of the pygambit solvers.
        See pygambit docs for details https://gambitproject.readthedocs.io/en/latest/pyapi.html#module-pygambit.nash
            - "enumpure": enumerate all pure NEs in the MACID
            - "enummixed": enumerate all mixed NEs in the MACID (only for 2-player games)
            - "lcp": Compute NE using the Linear Complementarity Program (LCP) solver (only for 2-player games)
            - "lp": Compute (one) NE using the Linear Programming solver (only for 2-player games)
            - "simpdiv": Compute (one) NE using the Simplicial Subdivision
            - "ipa": Compute (one) NE using the Iterative Partial Assignment solver (only for 2-player games)

        - Each NE comes as a list of FunctionCPDs, one for each decision node in the MACID.
        """
        return self.get_all_ne_in_sg(solver=solver)

    def joint_pure_policies(self, decisions: Iterable[str]) -> List[Tuple[StochasticFunctionCPD, ...]]:
        """return a list of tuples of all joint pure policies in the MACID. A joint pure policy assigns a
        pure decision rule to every decision node in the MACID."""
        all_dec_decision_rules = list(map(self.pure_decision_rules, decisions))
        return list(itertools.product(*all_dec_decision_rules))

    def get_all_ne_in_sg(
        self,
        decisions_in_sg: Optional[Iterable[str]] = None,
        solver: Optional[str] = "enumpure",
    ) -> List[List[StochasticFunctionCPD]]:
        """
        Return a list of all Nash equilbiria in a MACID subgame.

        - Each NE comes as a list of FunctionCPDs, one for each decision node in the MAID subgame.
        - solver can be any of the pygambit solvers.
        See pygambit docs for details https://gambitproject.readthedocs.io/en/latest/pyapi.html#module-pygambit.nash
        - If decisions_in_sg is not specified, this method finds all NE in the full MACID.
        - If the MACID being operated on already has function CPDs for some decision nodes, it is
        assumed that these have already been optimised and so these are not changed.
        """
        # TODO: Check that the decisions in decisions_in_sg actually make up a MAID subgame
        if decisions_in_sg is None:
            decisions_in_sg = self.decisions
        else:
            decisions_in_sg = set(decisions_in_sg)  # For efficient membership checks
        agents_in_sg = list({self.decision_agent[dec] for dec in decisions_in_sg})

        # old NE finder
        # agent_decs_in_sg = {
        #     agent: [dec for dec in self.agent_decisions[agent] if dec in decisions_in_sg] for agent in agents_in_sg
        # }

        # all_ne_in_sg: List[List[StochasticFunctionCPD]] = []
        # for pp in self.joint_pure_policies(decisions_in_sg):
        #     macid.add_cpds(*pp)  # impute the policy profile

        #     for a in agents_in_sg:  # check that each agent is happy
        #         eu_pp_agent_a = macid.expected_utility({}, agent=a)
        #         macid.add_cpds(*macid.optimal_pure_policies(agent_decs_in_sg[a])[0])
        #         max_eu_agent_a = macid.expected_utility({}, agent=a)
        #         if max_eu_agent_a > eu_pp_agent_a:  # not an NE
        #             break
        #     else:
        #         # it's an NE
        #         all_ne_in_sg.append(list(pp))

        # impute random decisions to non-instantiated, irrelevant decision nodes
        macid = self.copy()
        for d in macid.decisions:
            if not macid.is_s_reachable(decisions_in_sg, d) and isinstance(macid.get_cpds(d), DecisionDomain):
                macid.impute_random_decision(d)

        # # pygambit NE solver
        efg, parents_to_infoset = self.macid_to_pygambit_efg(macid, decisions_in_sg, agents_in_sg)
        # print(efg.write())
        print(parents_to_infoset)
        ne_behaviour_strategies = self.pygambit_ne_solver(efg, solver=solver)
        print(ne_behaviour_strategies)
        all_ne_in_sg = [
            self.behavior_to_cpd(efg, macid, parents_to_infoset, strat, decisions_in_sg)
            for strat in ne_behaviour_strategies
        ]

        print(all_ne_in_sg)

        return all_ne_in_sg

    def macid_to_pygambit_efg(
        self,
        macid: Optional[MACID] = None,
        decisions_in_sg: Optional[Iterable[str]] = None,
        agents_in_sg: Optional[Iterable[str]] = None,
    ) -> Tuple[pygambit.Game, Mapping[Tuple[Union[int, str], ...], pygambit.Infoset]]:
        """TODO check typing
        Creates the EFG from a MACID:
        1) Finds the MACID nodes needed for the EFG (decision nodes and informational parents S = {D u Pa_D})
        2) Creats an ordering of these nodes such that X_j precedes X_i in the ordering if and only if X_j
        is a descendant of X_i in the MACID.
        3) Labels each node X_i with a partial instantiation of the splits in the path to X_i in the EFG.
        """

        def _get_cur_node(game: pygambit.Game, idx: Tuple[int, ...]):
            """Returns the current node in the game tree given the index of the node."""
            cur_node = game.root
            # first entry is the root node
            if len(idx) == 1:
                return cur_node
            # traverse the tree
            for i in idx[1:]:
                cur_node = cur_node.children[i]
            return cur_node

        # can use on a subgame if copying, else do the whole game
        if macid is None:
            macid = self
        if decisions_in_sg is None:
            decisions_in_sg = macid.decisions

        # instantiate the decisions TODO this might be why it doesn't work, we want to force a subgame setting?
        # macid.impute_fully_mixed_policy_profile()
        # choose only relevant nodes
        game_tree_nodes = set(
            list(decisions_in_sg) + [parent for dec in decisions_in_sg for parent in macid.get_parents(dec)]
        )
        # topologically order them
        sorted_game_tree_nodes = macid.get_valid_order(game_tree_nodes)
        # create the pygambit efg
        game = pygambit.Game.new_tree()
        # add the players
        agent_to_player = {}  # TODO typing
        for agent in agents_in_sg:
            player = game.players.add(agent)
            agent_to_player[agent] = player

        # key is instantiation of parents, value is pygambit infoset
        parents_to_infoset = defaultdict(dict)  # TODO typing
        # nodes referenced in the game tree. Root has node_idx (0,), rest are (0, n, m, ...)
        # state is a dict of node_idx:state of partial instantiations of nodes
        node_idx_to_state = defaultdict(dict)
        # get cardinality of each node
        num_children = [1] + [len(macid.model.domain[node]) for node in sorted_game_tree_nodes]
        range_num_children = [list(range(x)) for x in num_children]

        # grow the tree in topological order, breadth first
        for i, node in enumerate(sorted_game_tree_nodes, start=1):
            # iterate over all possible parents of the node
            for node_idx in itertools.product(*range_num_children[:i]):
                # get current node
                cur_node = _get_cur_node(game, node_idx)
                parents = macid.get_parents(node)
                parents_actions = {parent: node_idx_to_state[node_idx][parent] for parent in parents}
                parents_actions_tuple = tuple(parents_actions.items())  # hashable

                # if the node is a decision, consider infosets
                if node in decisions_in_sg:
                    # get agent and domain
                    agent = macid.decision_agent[node]
                    player = agent_to_player[agent]
                    actions = macid.model.domain[node]
                    # check if this matches an existing infoset
                    if parents_actions_tuple in parents_to_infoset:
                        cur_infoset = parents_to_infoset[parents_actions_tuple]
                        cur_node.append_move(cur_infoset)
                    # else create a new infoset
                    else:
                        cur_infoset = cur_node.append_move(player, len(actions))
                        # label with the node for easy reference
                        cur_infoset.label = node
                        # add to infosets
                        parents_to_infoset[parents_actions_tuple] = cur_infoset
                    # add state info
                    for action_idx, action in enumerate(actions):
                        cur_infoset.actions[action_idx].label = action
                        state_info = node_idx_to_state[node_idx].copy()
                        state_info.update({node: action})
                        node_idx_to_state[node_idx + (action_idx,)] = state_info
                else:
                    # otherwise is a chance node
                    factor = macid.query([node], context=parents_actions)
                    move = cur_node.append_move(game.players.chance, factor.cardinality)
                    # add state info
                    actions = macid.model.domain[node]
                    for action_idx, prob in enumerate(factor.values):
                        move.actions[action_idx].label = actions[action_idx]
                        move.actions[action_idx].prob = pygambit.Decimal(prob)
                        state_info = node_idx_to_state[node_idx].copy()
                        state_info.update({node: actions[action_idx]})
                        node_idx_to_state[node_idx + (action_idx,)] = state_info
        # add payoffs to the game as leave nodes
        for node_idx in itertools.product(*range_num_children):
            cur_node = _get_cur_node(game, node_idx)
            context = node_idx_to_state[node_idx]
            # name outcome as a string of the node_idx
            payoff_tuple = game.outcomes.add(str(node_idx))
            for i, agent in enumerate(agents_in_sg):
                payoff = macid.expected_utility(context=context, agent=agent)
                payoff_tuple[i] = pygambit.Decimal(payoff)
            cur_node.outcome = payoff_tuple

        return game, parents_to_infoset

    def pygambit_ne_solver(
        self, efg: pygambit.Game, solver: Optional[str] = "enummixed"
    ) -> List[pygambit.lib.libgambit.MixedStrategyProfile]:
        """Uses pygambit to find the Nash equilibria of the EFG.
        Pygambit will raise errors if solver not allowed for the game (e.g. more than 2 players)
        """
        if solver == "enummixed":
            mixed_strategies = pygambit.nash.enummixed_solve(efg, rational=False)
        elif solver == "enumpure":
            mixed_strategies = pygambit.nash.enumpure_solve(efg)
        elif solver == "lcp":
            mixed_strategies = pygambit.nash.lcp_solve(efg, rational=False)
        elif solver == "lp":
            mixed_strategies = pygambit.nash.lp_solve(efg, rational=False)
        elif solver == "simpdiv":
            mixed_strategies = pygambit.nash.simpdiv_solve(efg, rational=False)
        else:
            raise ValueError(f"Solver {solver} not recognised")
        # convert to behavior strategies
        behavior_strategies = [x.as_behavior() for x in mixed_strategies]

        return behavior_strategies

    def behavior_to_cpd(
        self,
        efg: pygambit.Game,
        macid: MACID,
        state_to_infoset: Mapping[Tuple[Union[int, str], ...], pygambit.Infoset],
        behavior: pygambit.lib.libgambit.MixedStrategyProfile,
        decisions_in_sg: List[str],
    ) -> List[StochasticFunctionCPD]:
        """Convert a behavior strategy to list of CPDs for each decision node"""

        def _decimal_from_fraction(item: Any):
            if isinstance(item, pygambit.Rational):
                return item.numerator / item.denominator
            else:
                return item

        def _action_prob_given_parents(**pv: Any):
            """Takes the parent instantiation and outputs the prob from the infoset"""
            pv_tuple = tuple(pv.items())
            # get the infoset for the node
            infoset = state_to_infoset[pv_tuple]
            # get the action probs for the infoset
            action_probs = {
                infoset.actions[i].label: _decimal_from_fraction(prob) for i, prob in enumerate(behavior[infoset])
            }
            # get the prob for the action
            return action_probs

        # require domain to get cpd.values in the same order as in macid
        cpds = [
            StochasticFunctionCPD(
                variable=node,
                stochastic_function=_action_prob_given_parents,
                cbn=macid,
                domain=macid.model.domain[node],
            )
            for node in decisions_in_sg
        ]
        return cpds

    def policy_profile_assignment(self, partial_policy: Iterable[StochasticFunctionCPD]) -> Dict:
        """Return a dictionary with the joint or partial policy profile assigned -
        ie a decision rule for each of the MACIM's decision nodes."""
        pp: Dict[str, Optional[TabularCPD]] = {d: None for d in self.decisions}
        pp.update({cpd.variable: cpd for cpd in partial_policy})
        return pp

    def get_all_spe(self, solver: Optional[str] = "enumpure") -> List[List[StochasticFunctionCPD]]:
        """Return a list of all subgame perfect Nash equilbiria (SPE) in the MACIM
        - Each SPE comes as a list of FunctionCPDs, one for each decision node in the MACID.
        - solver can be any of the pygambit solvers.
        See pygambit docs for details https://gambitproject.readthedocs.io/en/latest/pyapi.html#module-pygambit.nash
        """
        spes: List[List[StochasticFunctionCPD]] = [[]]

        # backwards induction over the sccs in the condensed relevance graph (handling tie-breaks)
        for scc in reversed(CondensedRelevanceGraph(self).get_scc_topological_ordering()):
            extended_spes = []
            for partial_profile in spes:
                self.add_cpds(*partial_profile)
                all_ne_in_sg = self.get_all_ne_in_sg(decisions_in_sg=scc, solver=solver)
                for ne in all_ne_in_sg:
                    extended_spes.append(partial_profile + list(ne))
            spes = extended_spes
        return spes

    def decs_in_each_maid_subgame(self) -> List[set]:
        """
        Return a list giving the set of decision nodes in each MAID subgame of the original MAID.
        """
        con_rel = CondensedRelevanceGraph(self)
        con_rel_sccs = con_rel.nodes  # the nodes of the condensed relevance graph are the maximal sccs of the MA(C)ID
        powerset = list(
            itertools.chain.from_iterable(
                itertools.combinations(con_rel_sccs, r) for r in range(1, len(con_rel_sccs) + 1)
            )
        )
        con_rel_subgames = copy.deepcopy(powerset)
        for subset in powerset:
            for node in subset:
                if not nx.descendants(con_rel, node).issubset(subset) and subset in con_rel_subgames:
                    con_rel_subgames.remove(subset)

        dec_subgames = [
            [con_rel.get_decisions_in_scc()[scc] for scc in con_rel_subgame] for con_rel_subgame in con_rel_subgames
        ]

        return [set(itertools.chain.from_iterable(i)) for i in dec_subgames]

    def copy_without_cpds(self) -> MACID:
        """copy the MACID structure"""
        new = MACID()
        new.add_nodes_from(self.nodes)
        new.add_edges_from(self.edges)
        for agent in self.agents:
            for decision in self.agent_decisions[agent]:
                new.make_decision(decision, agent)
            for utility in self.agent_utilities[agent]:
                new.make_utility(utility, agent)
        return new
