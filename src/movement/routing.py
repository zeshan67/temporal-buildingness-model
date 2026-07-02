"""
routing.py

Basic routing helpers for the Temporal Buildingness model.

This first version supports conventional vehicle routing using shortest paths.
Later, this file will also support informal transport routing with demand,
anchor, curb, and pedestrian-attraction logic.
"""

import networkx as nx
import numpy as np


def get_shortest_route(
    graph: nx.Graph,
    origin: str,
    destination: str,
    weight: str = "length",
) -> list[str]:
    """
    Return the shortest route between two nodes.

    Args:
        graph: synthetic street graph
        origin: starting node ID
        destination: ending node ID
        weight: edge attribute used for shortest path calculation

    Returns:
        list of node IDs representing the route
    """

    if origin == destination:
        raise ValueError("Origin and destination must be different.")

    try:
        route = nx.shortest_path(
            graph,
            source=origin,
            target=destination,
            weight=weight,
        )
    except nx.NetworkXNoPath as error:
        raise nx.NetworkXNoPath(
            f"No route found between {origin} and {destination}"
        ) from error

    return route


def choose_random_node(
    graph: nx.Graph,
    rng: np.random.Generator,
) -> str:
    """
    Select one random node from the graph.
    """

    nodes = list(graph.nodes)
    return str(rng.choice(nodes))


def choose_random_origin_destination(
    graph: nx.Graph,
    rng: np.random.Generator,
) -> tuple[str, str]:
    """
    Select a random origin and destination pair.
    """

    origin = choose_random_node(graph, rng)
    destination = choose_random_node(graph, rng)

    while destination == origin:
        destination = choose_random_node(graph, rng)

    return origin, destination


def choose_random_destination(
    graph: nx.Graph,
    origin: str,
    rng: np.random.Generator,
) -> str:
    """
    Select a destination that is different from the origin.
    """

    destination = choose_random_node(graph, rng)

    while destination == origin:
        destination = choose_random_node(graph, rng)

    return destination


def calculate_route_distance(
    graph: nx.Graph,
    route: list[str],
    weight: str = "length",
) -> float:
    """
    Calculate the total distance of a route.
    """

    if len(route) < 2:
        return 0.0

    total_distance = 0.0

    for current_node, next_node in zip(route[:-1], route[1:]):
        total_distance += graph[current_node][next_node][weight]

    return float(total_distance)
