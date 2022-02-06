from __future__ import annotations

import copy
import itertools
from typing import Dict, Iterable, List, Optional, Tuple, Iterator

import networkx as nx
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import nashpy as nash

from pycid.core.cpd import DecisionDomain, StochasticFunctionCPD
from pycid.core.macid_base import MACIDBase
from pycid.core.relevance_graph import CondensedRelevanceGraph


class MACID(MACIDBase):
    """A Multi-Agent Causal Influence Diagram"""

    def get_all_pure_ne(self) -> List[List[StochasticFunctionCPD]]:
        """
        Return a list of all pure Nash equilbiria in the MACID.
        - Each NE comes as a list of FunctionCPDs, one for each decision node in the MACID.
        """
        return self.get_all_pure_ne_in_sg()

    def joint_pure_policies(self, decisions: Iterable[str]) -> List[Tuple[StochasticFunctionCPD, ...]]:
        """return a list of tuples of all joint pure policies in the MACID. A joint pure policy assigns a
        pure decision rule to every decision node in the MACID."""
        all_dec_decision_rules = list(map(self.pure_decision_rules, decisions))
        return list(itertools.product(*all_dec_decision_rules))

    def get_all_pure_ne_in_sg(
        self, decisions_in_sg: Optional[Iterable[str]] = None
    ) -> List[List[StochasticFunctionCPD]]:
        """
        Return a list of all pure Nash equilbiria in a MACID subgame.

        - Each NE comes as a list of FunctionCPDs, one for each decision node in the MAID subgame.
        - If decisions_in_sg is not specified, this method finds all pure NE in the full MACID.
        - If the MACID being operated on already has function CPDs for some decision nodes, it is
        assumed that these have already been optimised and so these are not changed.
        """
        # TODO: Check that the decisions in decisions_in_sg actually make up a MAID subgame
        if decisions_in_sg is None:
            decisions_in_sg = self.decisions
        else:
            decisions_in_sg = set(decisions_in_sg)  # For efficient membership checks

        agents_in_sg = list({self.decision_agent[dec] for dec in decisions_in_sg})
        agent_decs_in_sg = {
            agent: [dec for dec in self.agent_decisions[agent] if dec in decisions_in_sg] for agent in agents_in_sg
        }

        # impute random decisions to non-instantiated, irrelevant decision nodes
        macid = self.copy()
        for d in macid.decisions:
            if not macid.is_s_reachable(decisions_in_sg, d) and isinstance(macid.get_cpds(d), DecisionDomain):
                macid.impute_random_decision(d)

        # NE finder
        all_pure_ne_in_sg: List[List[StochasticFunctionCPD]] = []
        for pp in self.joint_pure_policies(decisions_in_sg):
            macid.add_cpds(*pp)  # impute the policy profile

            for a in agents_in_sg:  # check that each agent is happy
                eu_pp_agent_a = macid.expected_utility({}, agent=a)
                macid.add_cpds(*macid.optimal_pure_policies(agent_decs_in_sg[a])[0])
                max_eu_agent_a = macid.expected_utility({}, agent=a)
                if max_eu_agent_a > eu_pp_agent_a:  # not an NE
                    break
            else:
                # it's an NE
                all_pure_ne_in_sg.append(list(pp))

        return all_pure_ne_in_sg

    def policy_profile_assignment(self, partial_policy: Iterable[StochasticFunctionCPD]) -> Dict:
        """Return a dictionary with the joint or partial policy profile assigned -
        ie a decision rule for each of the MACIM's decision nodes."""
        pp: Dict[str, Optional[TabularCPD]] = {d: None for d in self.decisions}
        pp.update({cpd.variable: cpd for cpd in partial_policy})
        return pp

    def get_all_pure_spe(self) -> List[List[StochasticFunctionCPD]]:
        """Return a list of all pure subgame perfect Nash equilbiria (SPE) in the MACIM
        - Each SPE comes as a list of StochasticFunctionCPDs, one for each decision node in the MACID.
        """
        spes: List[List[StochasticFunctionCPD]] = [[]]

        # backwards induction over the sccs in the condensed relevance graph (handling tie-breaks)
        for scc in reversed(CondensedRelevanceGraph(self).get_scc_topological_ordering()):
            extended_spes = []
            for partial_profile in spes:
                self.add_cpds(*partial_profile)
                all_ne_in_sg = self.get_all_pure_ne_in_sg(scc)
                for ne in all_ne_in_sg:
                    extended_spes.append(partial_profile + list(ne))
            spes = extended_spes
        return spes

    def get_mixed_ne(self):
        """
        Returns a list of all mixed Nash equilibria in non-degnerate 2-player games using
        Nashpy's support enumeration implementation.
        Returns a list of most mixed Nash equilbria in degnerate 2-player games (see Nashpy's 
        documentation for more details)
        - Each NE comes as a list of StochasticFunctionCPDs, one for each decision node in the MACID.
        """
        agents = list(self.agents)
        if len(agents) != 2:
            raise ValueError(
                f"This MACID has {len(agents)} agents and yet this method currently only works for 2 agent games."
            )

        # convert MAID into normal form
        agent_pure_policies = [list(self.pure_policies(self.agent_decisions[agent])) for agent in agents]

        def _agent_util(pp, agent) -> float:
            self.add_cpds(*pp)
            return self.expected_utility({}, agent=agent)

        payoff1 = np.array(
            [[_agent_util(pp1 + pp2, agents[0]) for pp2 in agent_pure_policies[1]] for pp1 in agent_pure_policies[0]]
        )
        payoff2 = np.array(
            [[_agent_util(pp1 + pp2, agents[1]) for pp2 in agent_pure_policies[1]] for pp1 in agent_pure_policies[0]]
        )

        # find all mixed NE using Nashpy
        game = nash.Game(payoff1, payoff2)
        equilibria = game.support_enumeration()
        
        all_mixed_ne = []
        for eq in equilibria:
            mixed_ne = list(
                itertools.chain(
                    *[list(self._mixed_policy(agent_pure_policies[agent], eq[agent])) for agent in range(2)]
                )
            )
            all_mixed_ne.append(mixed_ne)
        return all_mixed_ne

    def _mixed_policy(self, agent_pure_policies, prob_dist) -> Iterator[StochasticFunctionCPD]:
        """
        Given a list of the agent's pure policies and a distribution over these pure policies, 
        return a generator of the equivalent mixed decision rules that make up this mixed policy
        """
    
        num_decision_rules = len(agent_pure_policies[0])  # how many decision nodes that agent has

        for i in range(num_decision_rules):
            decision = agent_pure_policies[0][i].variable
            domain = self.model.domain[decision]

            def mixed_dec_rule(pvs):  # construct mixed decision rule for each decision node
                cpd_dict = {}
                for d in self.model.domain[decision]:
                    cpd_dict[d] = sum([prob_dist[idx] for idx, p in enumerate(agent_pure_policies) if p[i].func(**pvs) ==d])
                return cpd_dict

            def produce_function():
                return lambda **parent_values: mixed_dec_rule(parent_values)

            yield StochasticFunctionCPD(decision, produce_function(), self, domain=domain)


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
