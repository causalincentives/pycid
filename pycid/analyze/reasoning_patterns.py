from typing import Any, Dict, List, Optional, Set, Union

import networkx as nx

from pycid.core.get_paths import (
    directed_decision_free_path,
    find_all_dir_paths,
    find_all_undir_paths,
    get_motif,
    is_active_indirect_frontdoor_trail,
    is_active_path,
)
from pycid.core.macid import MACID
from pycid.core.macid_base import MACIDBase


def _get_key_node(mb: MACIDBase, path: List[str]) -> str:
    """Find the key node of a path (ie the first "fork" node in the path)."""
    for _, b, _ in zip(path[:-2], path[1:-1], path[2:]):
        structure = get_motif(mb, path, path.index(b))
        if structure == "fork":
            return b
    raise ValueError("No key node found")


def _effective_dir_path_exists(mb: MACIDBase, start: str, finish: str, effective_set: Set[str]) -> bool:
    """Check whether an effective directed path exists."""
    return any(_path_is_effective(mb, path, effective_set) for path in find_all_dir_paths(mb, start, finish))


def _effective_undir_path_exists(mb: MACIDBase, start: str, finish: str, effective_set: Set[str]) -> bool:
    """Check whether an effective undirected path exists."""
    return any(_path_is_effective(mb, path, effective_set) for path in find_all_undir_paths(mb, start, finish))


def _path_is_effective(mb: MACIDBase, path: List[str], effective_set: Set[str]) -> bool:
    """Check whether a path is effective."""
    dec_nodes_in_path = set(mb.decisions).intersection(set(path[1:]))  # exclude first node of the path
    # all([]) evaluates to true => this covers case where path has no decision nodes
    return all(dec_node in effective_set for dec_node in dec_nodes_in_path)


def _directed_effective_path_not_through_set_y(
    mb: MACIDBase, start: str, finish: str, effective_set: Set[str], y: Set[str] = None
) -> bool:
    """Check whether a directed effective path exists that doesn't pass through any of the nodes in the set y."""
    if y is None:
        y = set()
    return any(
        y.isdisjoint(path) and _path_is_effective(mb, path, effective_set)
        for path in find_all_dir_paths(mb, start, finish)
    )


def _effective_backdoor_path_not_blocked_by_set_w(
    mb: MACIDBase, start: str, finish: str, effective_set: Set[str], w: Optional[Set[str]] = None
) -> Optional[List[str]]:
    """
    Returns the effective backdoor path not blocked if we condition on nodes in set w.
    If no such path exists, this returns None.
    """
    for path in find_all_undir_paths(mb, start, finish):
        is_backdoor_path = path[1] in mb.get_parents(path[0])
        not_blocked_by_w = is_active_path(mb, path, w)
        if is_backdoor_path and not_blocked_by_w and _path_is_effective(mb, path, effective_set):
            return path
    return None


def _effective_undir_path_not_blocked_by_set_w(
    mb: MACIDBase, start: str, finish: str, effective_set: Set[str], w: Optional[Set[str]] = None
) -> Union[List[str], None]:
    """
    Returns an effective undirected path not blocked if we condition on nodes in set w.
    If no such path exists, this returns None.
    """
    for path in find_all_undir_paths(mb, start, finish):
        not_blocked_by_w = is_active_path(mb, path, w)
        if not_blocked_by_w and _path_is_effective(mb, path, effective_set):
            return path
    else:
        return None


def direct_effect(macid: MACID, decision: str) -> bool:
    """Check whether a decision in motivated by a direct effect reasoning pattern.

    Graphical Criterion:
    1) There is a directed decision free path from D_A to a utility node U_A
    """
    agent = macid.decision_agent[decision]
    agent_utils = macid.agent_utilities[agent]
    for u in agent_utils:
        if directed_decision_free_path(macid, decision, u):
            return True
    else:
        return False


def manipulation(macid: MACID, decision: str, effective_set: Set[str]) -> bool:
    """Check whether a decision is motivated by an incentive for manipulation.

    Graphical Criterion:
    1) There is a directed decision-free path from D_A to an effective decision node D_B.
    2) There is a directed, effective path from D_B to U_A (an effective path is a path in which all
    decision nodes, except possibly the initial node, and except fork nodes, are effective)
    3) There is a directed, effective path from D_A to U_B that does not pass through D_B.
    """
    if any(node not in macid.nodes for node in effective_set):
        raise KeyError("One or many of the nodes in the effective_set are not present in the macid.")

    agent = macid.decision_agent[decision]
    agent_utils = macid.agent_utilities[agent]
    reachable_decisions = []  # set of possible D_B
    list_decs = list(macid.decisions)
    list_decs.remove(decision)
    for dec_reach in list_decs:
        if dec_reach in effective_set:
            if directed_decision_free_path(macid, decision, dec_reach):
                reachable_decisions.append(dec_reach)

    for decision_b in reachable_decisions:
        agent_b = macid.decision_agent[decision_b]
        agent_b_utils = macid.agent_utilities[agent_b]

        for u in agent_utils:
            if _effective_dir_path_exists(macid, decision_b, u, effective_set):

                for u_b in agent_b_utils:
                    if _directed_effective_path_not_through_set_y(macid, decision, u_b, effective_set, {decision_b}):
                        return True
    else:
        return False


def signaling(macid: MACID, decision: str, effective_set: Set[str]) -> bool:
    """checks to see whether this decision is motivated by an incentive for signaling

    Graphical Criterion:
    1) There is a directed decision-free path from D_A to an effective decision node D_B.
    2) There is a directed, effective path from D_B to U_A.
    3) There is an effective back-door path π from D_A to U_B that is not blocked by D_B U W^{D_A}_{D_B}.
    4) If C is the key node in π, there is an effective path from C to U_A that is not blocked by D_A U W^{C}_{D_A}

    """
    if any(node not in macid.nodes for node in effective_set):
        raise KeyError("One or many of the nodes in the effective_set are not present in the macid.")

    agent = macid.decision_agent[decision]
    agent_utils = macid.agent_utilities[agent]
    reachable_decisions = []  # set of possible D_B
    list_decs = list(macid.decisions)
    list_decs.remove(decision)
    for dec_reach in list_decs:
        if dec_reach in effective_set:
            if directed_decision_free_path(macid, decision, dec_reach):
                reachable_decisions.append(dec_reach)

    decision_descendants = set(nx.descendants(macid, decision))
    for decision_b in reachable_decisions:
        agent_b = macid.decision_agent[decision_b]
        agent_b_utils = macid.agent_utilities[agent_b]
        for u in agent_utils:
            if not _effective_dir_path_exists(macid, decision_b, u, effective_set):
                continue

            for u_b in agent_b_utils:
                cond_nodes = {decision_b}
                cond_nodes.update(node for node in macid.get_parents(decision_b) if node not in decision_descendants)

                path = _effective_backdoor_path_not_blocked_by_set_w(macid, decision, u_b, effective_set, cond_nodes)
                if not path:
                    continue

                if _get_key_node(macid, path):
                    key_node = _get_key_node(macid, path)
                else:
                    return False

                key_node_descendants = set(nx.descendants(macid, key_node))
                cond_nodes2 = {decision}
                cond_nodes2.update(node for node in macid.get_parents(decision) if node not in key_node_descendants)

                if _effective_undir_path_not_blocked_by_set_w(macid, key_node, u, effective_set, cond_nodes2):
                    return True
    else:
        return False


def revealing_or_denying(macid: MACID, decision: str, effective_set: Set[str]) -> bool:
    """Check whether a decision is motivated by an incentive for revealing or denying

    Graphical Criterion:
    1) There is a directed decision-free path from D_A to an effective decision node D_B.
    2) There is a direced, effective path from D_B to U_A.
    3) There is an effective indirect front-door path π from D_A to U_B that is not blocked by D_B U W^{D_A}_{D_B}.
    """
    if any(node not in macid.nodes for node in effective_set):
        raise KeyError("One or many of the nodes in the effective_set are not present in the macid.")

    agent = macid.decision_agent[decision]
    agent_utils = macid.agent_utilities[agent]
    reachable_decisions = []  # set of possible D_B
    list_decs = list(macid.decisions)
    list_decs.remove(decision)
    for dec_reach in list_decs:
        if dec_reach in effective_set:
            if directed_decision_free_path(macid, decision, dec_reach):
                reachable_decisions.append(dec_reach)

    decision_descendants = set(nx.descendants(macid, decision))
    for decision_b in reachable_decisions:
        agent_b = macid.decision_agent[decision_b]
        agent_b_utils = macid.agent_utilities[agent_b]

        for u in agent_utils:
            if not _effective_dir_path_exists(macid, decision_b, u, effective_set):
                continue

            for u_b in agent_b_utils:
                cond_nodes = {decision_b}
                cond_nodes.update(node for node in macid.get_parents(decision_b) if node not in decision_descendants)
                if is_active_indirect_frontdoor_trail(macid, decision, u_b, cond_nodes):
                    return True
    else:
        return False


def get_reasoning_patterns(mb: MACID) -> Dict[str, List[Any]]:
    """A dictionary matching each reasoning pattern with the decision nodes in the MAID which admit it.

    This finds all of the circumstances under which an agent in a MAID has a reason to prefer one policy over
    another, when all other agents are playing WD policies.
    (Pfeffer and Gal, 2007: On the Reasoning patterns of Agents in Games).
    """
    motivations: Dict[str, List[str]] = {"dir_effect": [], "sig": [], "manip": [], "rev_den": []}
    effective_set = set(mb.decisions)
    while True:
        new_set = {
            dec
            for dec in effective_set
            if direct_effect(mb, dec)
            or manipulation(mb, dec, effective_set)
            or signaling(mb, dec, effective_set)
            or revealing_or_denying(mb, dec, effective_set)
        }

        if len(new_set) == len(effective_set):
            break
        effective_set = new_set

    for decision in effective_set:
        if direct_effect(mb, decision):
            motivations["dir_effect"].append(decision)
        elif signaling(mb, decision, effective_set):
            motivations["sig"].append(decision)
        elif manipulation(mb, decision, effective_set):
            motivations["manip"].append(decision)
        elif revealing_or_denying(mb, decision, effective_set):
            motivations["rev_den"].append(decision)

    return motivations
