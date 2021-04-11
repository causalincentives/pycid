import random
from typing import List

from pgmpy.base import DAG


def add_random_edge(dag: DAG, node_order: List[str], max_in_degree: int = 4) -> None:
    """Add a random edge to the graph, that respects the given node_order, and
    also doesn't add a link if the sampled node has maximal in_degree already.

    It may not add any edge.
    """
    n1, n2 = random.sample(node_order, 2)
    if node_order.index(n1) < node_order.index(n2) and dag.in_degree(n2) < max_in_degree:
        dag.add_edge(n1, n2)
    elif node_order.index(n2) < node_order.index(n1) and dag.in_degree(n1) < max_in_degree:
        dag.add_edge(n2, n1)


def random_dag(number_of_nodes: int = 5, edge_density: float = 0.4, max_in_degree: int = 4) -> DAG:
    """Create a connected, random directed acyclic graph (DAG), with the given number of nodes,
    the given edge density, and with no node exceeding having too high in degree"""
    node_names = [f"X{i}" for i in range(number_of_nodes)]
    dag = DAG()

    # First make sure the dag is connected
    visited = list()
    unvisited = list(node_names)
    node = random.choice(unvisited)
    unvisited.remove(node)
    visited.append(node)
    dag.add_node(node)

    while unvisited:
        node = random.choice(unvisited)
        neighbor = random.choice(visited)
        if node_names.index(node) < node_names.index(neighbor) and dag.in_degree(neighbor) < max_in_degree:
            dag.add_edge(node, neighbor)
        elif node_names.index(neighbor) < node_names.index(node):
            dag.add_edge(neighbor, node)
        else:
            continue
        unvisited.remove(node)
        visited.append(node)

    # Then add edges until desired density is reached
    maximum_number_of_edges = number_of_nodes * (number_of_nodes - 1) / 2
    while dag.number_of_edges() < int(edge_density * maximum_number_of_edges):
        add_random_edge(dag, node_names)

    return dag
