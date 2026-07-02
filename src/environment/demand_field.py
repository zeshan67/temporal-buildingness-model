"""
demand_field.py

Creates synthetic demand zones for the Temporal Buildingness model.

Demand zones represent places where pedestrian demand is likely to appear:
- market edge
- school gate
- transit stop
- hospital entrance
- alley mouth
- residential cluster
- workplace entrance

This file does not use real-world data.
The demand zones are theoretically generated from the synthetic street graph.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.environment.street_graph import load_base_config, create_street_graph


OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_SUMMARIES_DIR = PROJECT_ROOT / "outputs" / "summaries"


DEMAND_ZONE_TYPES = [
    "market_edge",
    "school_gate",
    "transit_stop",
    "hospital_entrance",
    "alley_mouth",
    "residential_cluster",
    "workplace_entrance",
]


def calculate_node_activity_score(graph: nx.Graph, node_id: str) -> float:
    """
    Calculate a synthetic activity score for a node.

    A node becomes a stronger demand candidate when its connected street
    segments have high pedestrian intensity, street-edge activity, and curb access.
    """

    connected_edges = graph.edges(node_id, data=True)

    scores = []

    for _, _, edge_data in connected_edges:
        score = (
            0.40 * edge_data["pedestrian_intensity"]
            + 0.35 * edge_data["street_edge_activity"]
            + 0.25 * edge_data["curb_accessibility"]
        )
        scores.append(score)

    if not scores:
        return 0.0

    return float(np.mean(scores))


def create_demand_zones(
    graph: nx.Graph,
    number_of_zones: int = 5,
    random_seed: int = 42,
) -> list[dict]:
    """
    Select demand zones from high-activity nodes in the graph.

    Returns:
        list[dict]: list of synthetic demand zone dictionaries
    """

    rng = np.random.default_rng(random_seed)

    node_scores = []

    for node_id in graph.nodes:
        activity_score = calculate_node_activity_score(graph, node_id)
        node_scores.append((node_id, activity_score))

    # Sort nodes by activity score from high to low.
    node_scores = sorted(node_scores, key=lambda item: item[1], reverse=True)

    # Take a larger candidate pool so the result is not only the top 5 nodes.
    candidate_pool = node_scores[:20]

    selected_indices = rng.choice(
        len(candidate_pool),
        size=number_of_zones,
        replace=False,
    )

    demand_zones = []

    for zone_number, candidate_index in enumerate(selected_indices, start=1):
        node_id, activity_score = candidate_pool[candidate_index]

        zone_type = rng.choice(DEMAND_ZONE_TYPES)

        demand_zone = {
            "zone_id": f"DZ_{zone_number}",
            "node_id": node_id,
            "zone_type": str(zone_type),
            "demand_intensity": round(float(activity_score), 3),
            "peak_start": int(rng.integers(100, 300)),
            "peak_end": int(rng.integers(600, 900)),
            "waiting_tolerance": round(float(rng.uniform(0.30, 1.00)), 3),
            "trip_distance_preference": round(float(rng.uniform(0.20, 0.80)), 3),
        }

        demand_zones.append(demand_zone)

    return demand_zones


def save_demand_zones(demand_zones: list[dict]) -> Path:
    """
    Save demand zones as a CSV file.
    """

    OUTPUT_SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_SUMMARIES_DIR / "demand_zones.csv"

    demand_zones_df = pd.DataFrame(demand_zones)
    demand_zones_df.to_csv(output_path, index=False)

    return output_path


def plot_demand_zones(
    graph: nx.Graph,
    demand_zones: list[dict],
) -> Path:
    """
    Plot the synthetic street graph with demand zones highlighted.
    """

    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    positions = nx.get_node_attributes(graph, "pos")

    demand_node_ids = [zone["node_id"] for zone in demand_zones]

    plt.figure(figsize=(8, 8))

    nx.draw_networkx_edges(
        graph,
        positions,
        edge_color="lightgray",
        width=1,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=25,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=demand_node_ids,
        node_size=180,
    )

    labels = {
        zone["node_id"]: zone["zone_id"]
        for zone in demand_zones
    }

    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=8,
    )

    plt.title("Synthetic Demand Zones")
    plt.axis("equal")
    plt.axis("off")

    output_path = OUTPUT_FIGURES_DIR / "demand_zones_preview.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def print_demand_zones(demand_zones: list[dict]) -> None:
    """
    Print demand zones in a readable format.
    """

    print("Demand zones created successfully.")
    print(f"Number of demand zones: {len(demand_zones)}")

    for zone in demand_zones:
        print(
            f"{zone['zone_id']} | "
            f"{zone['zone_type']} | "
            f"node={zone['node_id']} | "
            f"demand_intensity={zone['demand_intensity']}"
        )


if __name__ == "__main__":
    base_config = load_base_config()
    street_graph = create_street_graph(base_config)

    demand_zones = create_demand_zones(
        street_graph,
        number_of_zones=5,
        random_seed=base_config["simulation"]["random_seed"],
    )

    print_demand_zones(demand_zones)

    csv_path = save_demand_zones(demand_zones)
    figure_path = plot_demand_zones(street_graph, demand_zones)

    print(f"\nDemand zone CSV saved to: {csv_path}")
    print(f"Demand zone preview saved to: {figure_path}")
