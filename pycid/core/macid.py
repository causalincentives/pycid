from __future__ import annotations

import copy
import itertools
from collections import defaultdict
from functools import partial, update_wrapper
from typing import Any, Callable, Dict, Hashable, Iterable, KeysView, List, Mapping, Optional, Set, Tuple, Union
from warnings import warn

import networkx as nx
import pygambit
from pgmpy.factors.discrete import TabularCPD

from pycid.core.cpd import DecisionDomain, StochasticFunctionCPD
from pycid.core.macid_base import MACIDBase
from pycid.core.relevance_graph import CondensedRelevanceGraph

Outcome = Any


class MACID(MACIDBase):
    """A Multi-Agent Causal Influence Diagram"""

    def get_ne(self, solver: Optional[str] = "enumpure") -> List[List[StochasticFunctionCPD]]:
        """
        Return a list of Nash equilbiria in the MACID. By default, this finds all pure NE using the 'enumpure'
        pygambit solver. Use the 'solver' argument to change this behaviour.
        Recommended Usage:
        - 2-player games: solver='enummixed' to find all mixed NE
        - N-player games: solver='enumpure' if one ones to find all pure NE, or solver={'simpdiv', 'ipa', 'gnm'}
        if one wants to find at least one mixed NE. See pygambit docs for details
        https://gambitproject.readthedocs.io/en/latest/pyapi.html#module-pygambit.nash
        - solver can be any of the pygambit solvers (default: "enumpure" - finds all pure NEs).
            - "enumpure": enumerate all pure NEs in the MACID.
                - for arbitrary N-player games
            - "enummixed": Valid for enumerate all mixed NEs in the MACID by computing the
              extreme points of the set of equilibria.
                - for 2-player games only
            - "lcp": Compute NE using the Linear Complementarity Program (LCP) solver.
                - for 2-player games only
            - "lp": Compute (one) NE using the Linear Programming solver.
                - for 2-player, constant sum games only
            - "simpdiv": Compute one mixed NE using the Simplicial Subdivision.
                - for arbitrary N-player games
            - "ipa": Compute one mixed NE using the Iterative Partial Assignment solver
                - for arbitrary N-player games
            - "gnm": Compute one mixed NE using the global newton method
                - for arbitrary N-player games

        Each NE comes as a list of FunctionCPDs, one for each decision node in the MACID.
        """
        return self.get_ne_in_sg(solver=solver)

    def joint_pure_policies(self, decisions: Iterable[str]) -> List[Tuple[StochasticFunctionCPD, ...]]:
        """return a list of tuples of all joint pure policies in the MACID. A joint pure policy assigns a
        pure decision rule to every decision node in the MACID."""
        all_dec_decision_rules = list(map(self.pure_decision_rules, decisions))
        return list(itertools.product(*all_dec_decision_rules))

    def get_ne_in_sg(
        self,
        decisions_in_sg: Optional[Iterable[str]] = None,
        solver: Optional[str] = "enumpure",
    ) -> List[List[StochasticFunctionCPD]]:
        """
        Return a list of NE in a MACID subgame. By default, this finds all pure NE in an arbitray N-player game.
        Use the 'solver' argument to change this behaviour (see get_ne method for details).
        - Each NE comes as a list of FunctionCPDs, one for each decision node in the MAID subgame.
        - If decisions_in_sg is not specified, this method finds NE in the full MACID.
        - If the MACID being operated on already has function CPDs for some decision nodes, it is
        assumed that these have already been optimised and so these are not changed.
        """
        # TODO: Check that the decisions in decisions_in_sg actually make up a MAID subgame
        if decisions_in_sg is None:
            decisions_in_sg = self.decisions
        else:
            decisions_in_sg = set(decisions_in_sg)  # For efficient membership checks
        agents_in_sg = list({self.decision_agent[dec] for dec in decisions_in_sg})

        # impute random decisions to non-instantiated, irrelevant decision nodes
        macid = self.copy()
        for d in macid.decisions:
            if not macid.is_s_reachable(decisions_in_sg, d) and isinstance(macid.get_cpds(d), DecisionDomain):
                macid.impute_random_decision(d)

        # pygambit NE solver
        efg, parents_to_infoset = self._macid_to_pygambit_efg(macid, decisions_in_sg, agents_in_sg)
        ne_behaviour_strategies = self._pygambit_ne_solver(efg, solver=solver)
        ne_in_sg = [
            self._behavior_to_cpd(macid, parents_to_infoset, strat, decisions_in_sg)
            for strat in ne_behaviour_strategies
        ]

        return ne_in_sg

    def policy_profile_assignment(self, partial_policy: Iterable[StochasticFunctionCPD]) -> Dict:
        """Return a dictionary with the joint or partial policy profile assigned -
        ie a decision rule for each of the MACIM's decision nodes."""
        pp: Dict[str, Optional[TabularCPD]] = {d: None for d in self.decisions}
        pp.update({cpd.variable: cpd for cpd in partial_policy})
        return pp

    def get_spe(self, solver: Optional[str] = "enumpure") -> List[List[StochasticFunctionCPD]]:
        """Return a list of subgame perfect Nash equilbiria (SPE) in the MACIM.
        By default, finds all pure SPE using the "enumpure" pygambit solver.
        Use the 'solver' argument to change this behaviour (see get_ne method for details).
        - Each SPE comes as a list of FunctionCPDs, one for each decision node in the MACID.
        """
        spes: List[List[StochasticFunctionCPD]] = [[]]

        # backwards induction over the sccs in the condensed relevance graph (handling tie-breaks)
        for scc in reversed(CondensedRelevanceGraph(self).get_scc_topological_ordering()):
            extended_spes = []
            for partial_profile in spes:
                self.add_cpds(*partial_profile)
                ne_in_sg = self.get_all_ne_in_sg(decisions_in_sg=scc, solver=solver)
                for ne in ne_in_sg:
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

    def _add_players(self, game: pygambit.Game, agents_in_sg: Iterable[Hashable]) -> Dict[Hashable, pygambit.Player]:
        """add players to the pygambit game"""
        agent_to_player = {}
        for agent in agents_in_sg:
            player = game.players.add(agent)
            agent_to_player[agent] = player

        return agent_to_player

    def _add_payoffs(
        self,
        macid: MACID,
        game: pygambit.Game,
        range_num_children: List[List],
        node_idx_to_state: Dict[Tuple[int, ...], Dict[str, Any]],
        agents_in_sg: Iterable[Hashable],
    ) -> None:
        """add payoffs to the game as leave nodes"""
        for node_idx in itertools.product(*range_num_children):
            cur_node = self._get_cur_node(game, node_idx)
            context = node_idx_to_state[node_idx]
            # name outcome as a string of the node_idx
            payoff_tuple = game.outcomes.add(str(node_idx))
            for i, agent in enumerate(agents_in_sg):
                payoff = macid.expected_utility(context=context, agent=agent)
                payoff_tuple[i] = pygambit.Decimal(payoff)
            cur_node.outcome = payoff_tuple

    def _get_cur_node(self, game: pygambit.Game, idx: Tuple[int, ...]) -> pygambit.Node:
        """Returns the current node in the game tree given the index of the node."""
        cur_node = game.root
        # first entry is the root node
        if len(idx) == 1:
            return cur_node
        # traverse the tree
        for i in idx[1:]:
            cur_node = cur_node.children[i]
        return cur_node

    def _macid_to_pygambit_efg(
        self,
        macid: Optional[MACID] = None,
        decisions_in_sg: Optional[Union[KeysView[str], Set[str]]] = None,
        agents_in_sg: Optional[Iterable[Hashable]] = None,
    ) -> Tuple[pygambit.Game, Mapping[Tuple[Hashable, Tuple[Tuple[Any, Any], ...]], pygambit.Infoset]]:
        """
        Creates the EFG from a MACID:
        1) Finds the MACID nodes needed for the EFG (decision nodes and informational parents S = {D u Pa_D})
        2) Creats an ordering of these nodes such that X_j precedes X_i in the ordering if and only if X_j
        is a descendant of X_i in the MACID.
        3) Labels each node X_i with a partial instantiation of the splits in the path to X_i in the EFG.
        """

        # can use on a subgame if copying, else do the whole game
        if macid is None:
            macid = self
        if decisions_in_sg is None:
            decisions_in_sg = macid.decisions
        if agents_in_sg is None:
            agents_in_sg = macid.agents

        # choose only relevant nodes
        game_tree_nodes = set(
            list(decisions_in_sg) + [parent for dec in decisions_in_sg for parent in macid.get_parents(dec)]
        )
        # topologically order them
        sorted_game_tree_nodes = macid.get_valid_order(game_tree_nodes)
        # create the pygambit efg
        game = pygambit.Game.new_tree()

        agent_to_player = self._add_players(game, agents_in_sg)

        # key is instantiation of parents, value is pygambit infoset
        parents_to_infoset: Dict[Tuple[Hashable, Tuple[Tuple[Any, Any], ...]], pygambit.Infoset] = defaultdict(dict)
        # nodes referenced in the game tree. Root has node_idx (0,), rest are (0, n, m, ...)
        # state is a dict of node_idx:state of partial instantiations of nodes
        node_idx_to_state: Dict[Tuple[int, ...], Dict[str, Any]] = defaultdict(dict)
        # get cardinality of each node
        num_children = [1] + [len(macid.model.domain[node]) for node in sorted_game_tree_nodes]
        range_num_children = [list(range(x)) for x in num_children]

        # grow the tree in topological order, breadth first
        for i, node in enumerate(sorted_game_tree_nodes, start=1):
            # iterate over all possible parents of the node
            for node_idx in itertools.product(*range_num_children[:i]):
                # get current node
                cur_node = self._get_cur_node(game, node_idx)
                parents = macid.get_parents(node)
                parents_actions = {parent: node_idx_to_state[node_idx][parent] for parent in parents}

                # if the node is a decision, consider infosets
                if node in decisions_in_sg:
                    # get agent and domain
                    agent = macid.decision_agent[node]
                    player = agent_to_player[agent]
                    actions = macid.model.domain[node]
                    parents_actions_tuple = (agent, tuple(parents_actions.items()))  # hashable
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
                        cur_infoset.actions[action_idx].label = str(action)
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
                        move.actions[action_idx].label = str(actions[action_idx])
                        move.actions[action_idx].prob = pygambit.Decimal(prob)
                        state_info = node_idx_to_state[node_idx].copy()
                        state_info.update({node: actions[action_idx]})
                        node_idx_to_state[node_idx + (action_idx,)] = state_info

        self._add_payoffs(macid, game, range_num_children, node_idx_to_state, agents_in_sg)

        return game, parents_to_infoset

    def _pygambit_ne_solver(
        self, game: pygambit.Game, solver: Optional[str] = "enumpure"
    ) -> List[pygambit.lib.libgambit.MixedStrategyProfile]:
        """Uses pygambit to find the Nash equilibria of the EFG.
        Pygambit will raise errors if solver not allowed for the game (e.g. not constant sum for lp)
        """
        # check if not 2 player game
        two_player = True if len(game.players) == 2 else False
        if solver in ["enummixed", "lcp", "lp"] and not two_player:
            warn(
                f"Solver {solver} not allowed for non-2 player games. Using 'simpdiv' instead."
                "Note this will find only one NE."
            )
            solver = "simpdiv"

        if solver == "enummixed":
            mixed_strategies = pygambit.nash.enummixed_solve(game, rational=False)
        elif solver == "enumpure":
            mixed_strategies = pygambit.nash.enumpure_solve(game)
        elif solver == "lcp":
            mixed_strategies = pygambit.nash.lcp_solve(game, rational=False)
        elif solver == "lp":
            mixed_strategies = pygambit.nash.lp_solve(game, rational=False)
        elif solver == "simpdiv":
            mixed_strategies = pygambit.nash.simpdiv_solve(game)
        elif solver == "ipa":
            mixed_strategies = pygambit.nash.ipa_solve(game)
        elif solver == "gnm":
            mixed_strategies = pygambit.nash.gnm_solve(game)
        else:
            raise ValueError(f"Solver {solver} not recognised")
        # convert to behavior strategies
        behavior_strategies = [x.as_behavior() if solver not in ["lp", "lcp"] else x for x in mixed_strategies]

        return behavior_strategies

    def _behavior_to_cpd(
        self,
        macid: MACID,
        state_to_infoset: Mapping[Tuple[Hashable, Tuple[Tuple[str, Any], ...]], pygambit.Infoset],
        behavior: pygambit.lib.libgambit.MixedStrategyProfile,
        decisions_in_sg: Union[KeysView[str], Set[str]] = None,
    ) -> List[StochasticFunctionCPD]:
        """Convert a behavior strategy to list of CPDs for each decision node"""

        if decisions_in_sg is None:
            decisions_in_sg = self.decisions

        def _decimal_from_fraction(item: Any) -> float:
            if isinstance(item, pygambit.Rational):
                return float(item.numerator / item.denominator)
            else:
                return float(item)

        def _action_prob_given_parents(node: Any, **pv: Outcome) -> Mapping[str, float]:
            """Takes the parent instantiation and outputs the prob from the infoset"""
            pv_tuple = (macid.decision_agent[node], tuple(pv.items()))
            # get the infoset for the node
            infoset = state_to_infoset[pv_tuple]
            # if the infoset does not exist, this is not a valid parent instantiation
            # TODO check what to do here
            if not infoset:
                return {}
            # get the action probs for the infoset
            action_probs = {
                macid.model.domain[node][i]: _decimal_from_fraction(prob) for i, prob in enumerate(behavior[infoset])
            }
            return action_probs

        def _wrapped_partial(func: Callable, *args: str) -> Callable:
            """Adds __name__ and __doc__ to partial functions"""
            partial_func = partial(func, *args)
            update_wrapper(partial_func, func)
            return partial_func

        # require domain to get cpd.values in the same order as in macid
        cpds = [
            StochasticFunctionCPD(
                variable=node,
                stochastic_function=_wrapped_partial(_action_prob_given_parents, node),
                cbn=macid,
                domain=macid.model.domain[node],
            )
            for node in decisions_in_sg
        ]
        return cpds
