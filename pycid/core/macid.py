from __future__ import annotations

import copy
import itertools
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
from pgmpy.factors.discrete import TabularCPD

from pycid.core.cpd import DecisionDomain, StochasticFunctionCPD
from pycid.core.macid_base import MACIDBase
from pycid.core.relevance_graph import CondensedRelevanceGraph
from pycid.export.gambit import behavior_to_cpd, macid_to_efg, pygambit_ne_solver

Outcome = Any


class MACID(MACIDBase):
    """A Multi-Agent Causal Influence Diagram"""

    def get_ne(self, solver: Optional[str] = None) -> List[List[StochasticFunctionCPD]]:
        """
        Return a list of Nash equilbiria in the MACID using pygambit solvers.
        By default, this does the following:
        - 2-player games: uses solver='enummixed' to find all mixed NE
        - N-player games (where N ≠ 2): uses solver='enumpure' to find all pure NE.
          If no pure NE exist, uses solver='simpdiv' to find 1 mixed NE if it exists.
        To change this behavior, use the 'solver' argument. The following solvers are available:
            - "enumpure": enumerate all pure NEs in the MACID.
                - for arbitrary N-player games
            - "enummixed": Valid for enumerate all mixed NEs in the MACID by computing the
              extreme points of the set of equilibria.
                - for 2-player games only
            - "lcp": Compute NE using the Linear Complementarity Program (LCP) solver.
                - for 2-player games only
            - "lp": Compute one NE using the Linear Programming solver.
                - for 2-player, constant sum games only
            - "simpdiv": Compute one mixed NE using the Simplicial Subdivision.
                - for arbitrary N-player games
            - "ipa": Compute one mixed NE using the Iterative Partial Assignment solver
                - for arbitrary N-player games
            - "gnm": Compute one mixed NE using the global newton method
                - for arbitrary N-player games
        See pygambit docs for more details
        https://gambitproject.readthedocs.io/en/latest/pyapi.html#module-pygambit.nash
        Each NE comes as a list of FunctionCPDs, one for each decision node in the MACID.
        """
        return self.get_ne_in_sg(solver=solver)

    def get_ne_in_sg(
        self,
        decisions_in_sg: Optional[Iterable[str]] = None,
        solver: Optional[str] = None,
    ) -> List[List[StochasticFunctionCPD]]:
        """
        Return a list of NE in a MACID subgame.
        By default, this does the following:
        - 2-player games: uses solver='enummixed' to find all mixed NE
        - N-player games (where N ≠ 2): uses solver='enumpure' to find all pure NE.
          If no pure NE exist, uses solver='simpdiv' to find 1 mixed NE if it exists.
        Use the 'solver' argument to change this behavior (see get_ne method for details).
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
        sg_macid = self.copy()
        for d in sg_macid.decisions:
            if not sg_macid.is_s_reachable(decisions_in_sg, d) and isinstance(sg_macid.get_cpds(d), DecisionDomain):
                sg_macid.impute_random_decision(d)

        # pygambit NE solver
        efg, parents_to_infoset = macid_to_efg(sg_macid, decisions_in_sg, agents_in_sg)
        ne_behavior_strategies = pygambit_ne_solver(efg, solver_override=solver)
        ne_in_sg = [
            behavior_to_cpd(sg_macid, parents_to_infoset, strat, decisions_in_sg) for strat in ne_behavior_strategies
        ]

        return ne_in_sg

    def get_spe(self, solver: Optional[str] = None) -> List[List[StochasticFunctionCPD]]:
        """Return a list of subgame perfect equilbiria (SPE) in the MACIM.
        By default, this finds mixed eq using the 'enummixed' pygambit solver for 2-player subgames, and
        pure eq using the 'enumpure' pygambit solver for N-player games. If pure NE do not exist,
        it uses the 'simpdiv' solver to find a mixed eq.
        Use the 'solver' argument to change this behavior (see get_ne method for details).
        - Each SPE comes as a list of FunctionCPDs, one for each decision node in the MACID.
        """
        spes: List[List[StochasticFunctionCPD]] = [[]]

        macid = self.copy()
        # backwards induction over the sccs in the condensed relevance graph (handling tie-breaks)
        for scc in reversed(CondensedRelevanceGraph(macid).get_scc_topological_ordering()):
            extended_spes = []
            for partial_profile in spes:
                macid.add_cpds(*partial_profile)
                ne_in_sg = macid.get_ne_in_sg(decisions_in_sg=scc, solver=solver)
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

    def joint_pure_policies(self, decisions: Iterable[str]) -> List[Tuple[StochasticFunctionCPD, ...]]:
        """return a list of tuples of all joint pure policies in the MACID. A joint pure policy assigns a
        pure decision rule to every decision node in the MACID."""
        all_dec_decision_rules = list(map(self.pure_decision_rules, decisions))
        return list(itertools.product(*all_dec_decision_rules))

    def policy_profile_assignment(self, partial_policy: Iterable[StochasticFunctionCPD]) -> Dict:
        """Return a dictionary with the joint or partial policy profile assigned -
        ie a decision rule for each of the MACIM's decision nodes."""
        pp: Dict[str, Optional[TabularCPD]] = {d: None for d in self.decisions}
        pp.update({cpd.variable: cpd for cpd in partial_policy})
        return pp

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
