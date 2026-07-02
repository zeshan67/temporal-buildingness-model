"""
anchor_points.py

Creates synthetic anchor points for the Temporal Buildingness model.

Anchor points are places where informal transport agents repeatedly pause,
wait, cluster, or return.

Examples:
- curbside cluster
- market entrance
- informal stand
- school gate
- station edge
- shaded corner

This file does not use real-world data.
Anchor points are generated from synthetic street attributes.
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


ANCHOR_TYPES = [
    "curbside_cluster",
    "market_entrance",
    "informal_stand",
    "school_gate",
    "station_edge",
    "shaded_corner",
]


def calculate_anchor_score(graph: nx.Graph, node_id: str) -> float:
    """
    Calculate how suitable a node is as an informal transport anchor.

    Anchor strength increases when:
    - curb accessibility is high
    - street-edge activity is high
    - anchor probability is high
    - pedestrian intensity is high
    """

    connected_edges = graph.edges(node_id, data=True)

    scores = []

    for _, _, edge_data in connected_edges:
        score = (
            0.35 * edge_data["anchor_probability"]
            + 0.30 * edge_data["curb_accessibility"]
            + 0.25 * edge_data["street_edge_activity"]
            + 0.10 * edge_data["pedestrian_intensity"]
        )
        scores.append(score)

    if not scores:
        return 0.0

    return float(np.mean(scores))


def create_anchor_points(
    graph: nx.Graph,
    number_of_anchors: int = 5,
    random_seed: int = 42,
) -> list[dict]:
    """
    Select anchor points from high-anchor-score nodes.

    Returns:
        list[dict]: list of synthetic anchor point dictionaries
    """

    rng = np.random.default_rng(random_seed + 100)

    node_scores = []

    for node_id in graph.nodes:
        anchor_score = calculate_anchor_score(graph, node_id)
        node_scores.append((node_id, anchor_score))

    node_scores = sorted(node_scores, key=lambda item: item[1], reverse=True)

    candidate_pool = node_scores[:20]

    selected_indices = rng.choice(
        len(candidate_pool),
        size=number_of_anchors,
        replace=False,
    )

    anchor_points = []

    for anchor_number, candidate_index in enumerate(selected_indices, start=1):
        node_id, anchor_score = candidate_pool[candidate_index]

        anchor_type = rng.choice(ANCHOR_TYPES)

        anchor_point = {
            "anchor_id": f"AP_{anchor_number}",
            "node_id": node_id,
            "anchor_type": str(anchor_type),
            "anchor_strength": round(float(anchor_score), 3),
            "edge_activity": round(float(rng.uniform(0.40, 1.00)), 3),
            "pedestrian_overlap": round(float(rng.uniform(0.40, 1.00)), 3),
            "allowed_wait_time_min": int(rng.integers(10, 30)),
            "allowed_wait_time_max": int(rng.integers(40, 90)),
        }

        anchor_points.append(anchor_point)

    return anchor_points


def save_anchor_points(anchor_points: list[dict]) -> Path:
    """
    Save anchor points as a CSV file.
    """

    OUTPUT_SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_SUMMARIES_DIR / "anchor_points.csv"

    anchor_points_df = pd.DataFrame(anchor_points)
    anchor_points_df.to_csv(output_path, index=False)

    return output_path


def plot_anchor_points(
    graph: nx.Graph,
    anchor_points: list[dict],
) -> Path:
    """
    Plot the synthetic street graph with anchor points highlighted.
    """

    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    positions = nx.get_node_attributes(graph, "pos")

    anchor_node_ids = [anchor["node_id"] for anchor in anchor_points]

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
        nodelist=anchor_node_ids,
        node_size=180,
    )

    labels = {
        anchor["node_id"]: anchor["anchor_id"]
        for anchor in anchor_points
    }

    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=8,
    )

    plt.title("Synthetic Anchor Points")
    plt.axis("equal")
    plt.axis("off")

    output_path = OUTPUT_FIGURES_DIR / "anchor_points_preview.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def print_anchor_points(anchor_points: list[dict]) -> None:
    """
    Print anchor points in a readable format.
    """

    print("Anchor points created successfully.")
    print(f"Number of anchor points: {len(anchor_points)}")

    for anchor in anchor_points:
        print(
            f"{anchor['anchor_id']} | "
            f"{anchor['anchor_type']} | "
            f"node={anchor['node_id']} | "
            f"anchor_strength={anchor['anchor_strength']}"
        )


if __name__ == "__main__":
    base_config = load_base_config()
    street_graph = create_street_graph(base_config)

    anchor_points = create_anchor_points(
        street_graph,
        number_of_anchors=5,
        random_seed=base_config["simulation"]["random_seed"],
    )

    print_anchor_points(anchor_points)

    csv_path = save_anchor_points(anchor_points)
    figure_path = plot_anchor_points(street_graph, anchor_points)

    print(f"\nAnchor point CSV saved to: {csv_path}")
    print(f"Anchor point preview saved to: {figure_path}")
