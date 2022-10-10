from __future__ import annotations

import copy
import itertools
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
from pgmpy.factors.discrete import TabularCPD

from pycid.core.cpd import DecisionDomain, StochasticFunctionCPD
from pycid.core.macid_base import AgentLabel, MACIDBase
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

    def create_subgame(self, active_subgame_decs: Iterable[str]) -> MACID:
        """
        Creates a subgame from the full macid:
        1) Find the set of nodes that are r-relevant to the active decision nodes present in this subgame.
        2) Construct the subgame graph from a subgraph of the full MACID's graph (we keep edges that terminate
        in one of the active subgame decisions or nodes from (1)).
        3) Uniform randomly initialise decision nodes in the full MAID that aren't present in the subgame
        (and also haven't already been initialised)
        4) For nodes that are parents of active subgame decision nodes or nodes from (1), marginalise out their
        parents in the original MAID to create valid CPDs for the subgame.
        5) Copy over CPDs for every other variable from the original MACID.
        Args:
        - active_subgame_decs: The decisions to include in the subgame
        Returns:
        - the subgame (MACID) object
        """
        # find the nodes that are r-relevant for the active_subgame_decs
        r_nodes = [node for node in self.nodes if self.is_r_reachable(active_subgame_decs, node)]
        # adds the dection nodes to this set
        r_nodes_plus_decs = set(r_nodes).union(set(active_subgame_decs))
        # find the edges from the full MAID that are present in this subgame (these
        # are those that terminate at one of the nodes in r_nodes_plus_decs)
        edges_sg = [pair for pair in self.edges if pair[1] in r_nodes_plus_decs]
        #   sg_nodes = set([node for pair in edges_sg for node in pair])
        # finds the decision_agents mapping in this subgame
        sg_agents_decs: Dict[AgentLabel, List[str]] = {}
        for node in active_subgame_decs:
            agent = self.decision_agent[node]
            if agent in sg_agents_decs:
                sg_agents_decs[agent].append(node)
            else:
                sg_agents_decs[agent] = [node]
        # finds the utilities_agents mapping in this subgame
        sg_agents_utils: Dict[AgentLabel, List[str]] = {}
        for node in set(r_nodes).intersection(set(self.utilities)):
            agent = self.utility_agent[node]
            if agent in sg_agents_utils:
                sg_agents_utils[agent].append(node)
            else:
                sg_agents_utils[agent] = [node]
        # instantiate the skeleton of this MAID
        sg_macid = MACID(edges=edges_sg, agent_decisions=sg_agents_decs, agent_utilities=sg_agents_utils)

        # random initialise decisions not already instantiated that aren't active in this subgame.
        # (this is to ensure that they are all "fully mixed - i.e., every action is chosen with positive probability")
        macid_copy = self.copy()
        for d in macid_copy.decisions:
            if not macid_copy.is_s_reachable(active_subgame_decs, d) and isinstance(
                macid_copy.get_cpds(d), DecisionDomain
            ):
                macid_copy.impute_random_decision(d)

        # for every node in the subgame that is a parent of a r_nodes_plus_decs node,
        #  marginalise out its parents from its CPD
        parents_for_marginalisation_of_original_cpd = set(sg_macid.nodes) - r_nodes_plus_decs
        for node in parents_for_marginalisation_of_original_cpd:
            cpd = macid_copy.get_cpds(node)
            cpd.marginalize(macid_copy.get_parents(node))
            marginalised_probs = cpd.get_values()
            unpack_probs = [item for sublist in marginalised_probs for item in sublist]
            sg_macid.model[node] = dict(zip(cpd.domain, unpack_probs))

        # copy over the cpds for everything else
        for node in set(sg_macid.nodes) - parents_for_marginalisation_of_original_cpd:
            sg_macid.model[node] = macid_copy.get_cpds(node)  # or self.model[node]

        return sg_macid

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
        if decisions_in_sg is None:
            decisions_in_sg = self.decisions
        else:
            decisions_in_sg = set(decisions_in_sg)  # For efficient membership checks
        agents_in_sg = list({self.decision_agent[dec] for dec in decisions_in_sg})

        # We exploit the fact that we only need to create an EFG correspondsing to the MACID subgame
        # we don't need to create an EFG for the full MACID
        # - the macid_to_efg transformation is exponential in the number of MACID nodes
        sg_macid = self.create_subgame(decisions_in_sg)
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
