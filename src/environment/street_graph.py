"""
street_graph.py

Creates a synthetic urban street network for the Temporal Buildingness model.

This file does not use real-world data.
It generates a theoretical 12 x 12 grid network where:
- nodes represent intersections / curb access points
- edges represent street segments
- edges receive synthetic attributes such as pedestrian intensity,
  curb accessibility, congestion, street-edge activity, and anchor probability
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "base_config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_base_config(config_path: Path = CONFIG_PATH) -> dict:
    """
    Load the base simulation configuration from YAML.
    """

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def make_node_id(x: int, y: int) -> str:
    """
    Convert grid coordinates into a readable node ID.
    Example: x=3, y=5 becomes 'N_3_5'
    """

    return f"N_{x}_{y}"


def create_edge_attributes(
    rng: np.random.Generator,
    default_edge_length: float,
) -> dict:
    """
    Create synthetic street attributes for one street segment.

    These values are theoretical. They are not measured from the real world.
    They allow the future movement model to decide which streets are more
    attractive to informal transport agents.
    """

    pedestrian_intensity = rng.uniform(0.10, 1.00)
    curb_accessibility = rng.uniform(0.10, 1.00)
    congestion_level = rng.uniform(0.05, 0.80)
    street_edge_activity = rng.uniform(0.10, 1.00)

    # Anchor probability is higher when street-edge activity and curb access are higher.
    anchor_probability = (
        0.50 * street_edge_activity
        + 0.30 * curb_accessibility
        + 0.20 * pedestrian_intensity
    )

    return {
        "length": float(default_edge_length),
        "vehicular_speed_limit": float(rng.uniform(8.0, 14.0)),
        "pedestrian_intensity": float(round(pedestrian_intensity, 3)),
        "curb_accessibility": float(round(curb_accessibility, 3)),
        "congestion_level": float(round(congestion_level, 3)),
        "street_edge_activity": float(round(street_edge_activity, 3)),
        "anchor_probability": float(round(anchor_probability, 3)),
    }


def create_street_graph(config: dict) -> nx.Graph:
    """
    Create a synthetic grid street graph.

    Returns:
        networkx.Graph: synthetic street network
    """

    grid_width = config["network"]["grid_width"]
    grid_height = config["network"]["grid_height"]
    default_edge_length = config["network"]["default_edge_length"]
    random_seed = config["simulation"]["random_seed"]

    rng = np.random.default_rng(random_seed)

    graph = nx.Graph()

    # 1. Create nodes.
    for x in range(grid_width):
        for y in range(grid_height):
            node_id = make_node_id(x, y)

            graph.add_node(
                node_id,
                x=x,
                y=y,
                pos=(x, y),
                is_boundary=(
                    x == 0
                    or y == 0
                    or x == grid_width - 1
                    or y == grid_height - 1
                ),
            )

    # 2. Create horizontal street edges.
    for x in range(grid_width - 1):
        for y in range(grid_height):
            node_a = make_node_id(x, y)
            node_b = make_node_id(x + 1, y)

            graph.add_edge(
                node_a,
                node_b,
                **create_edge_attributes(rng, default_edge_length),
            )

    # 3. Create vertical street edges.
    for x in range(grid_width):
        for y in range(grid_height - 1):
            node_a = make_node_id(x, y)
            node_b = make_node_id(x, y + 1)

            graph.add_edge(
                node_a,
                node_b,
                **create_edge_attributes(rng, default_edge_length),
            )

    return graph


def summarize_graph(graph: nx.Graph) -> None:
    """
    Print a simple summary of the generated street graph.
    """

    print("Street graph created successfully.")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    sample_edge = list(graph.edges(data=True))[0]
    print("\nSample street segment:")
    print(sample_edge)


def plot_street_graph(graph: nx.Graph) -> Path:
    """
    Save a preview image of the synthetic street graph.

    Edge darkness represents street-edge activity.
    This is only a visual check, not a final research figure.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    positions = nx.get_node_attributes(graph, "pos")

    edge_activity_values = [
        graph[u][v]["street_edge_activity"]
        for u, v in graph.edges()
    ]

    plt.figure(figsize=(8, 8))

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=30,
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        edge_color=edge_activity_values,
        edge_cmap=plt.cm.viridis,
        width=2,
    )

    plt.title("Synthetic 12 x 12 Street Graph")
    plt.axis("equal")
    plt.axis("off")

    output_path = OUTPUT_DIR / "street_graph_preview.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


if __name__ == "__main__":
    base_config = load_base_config()
    street_graph = create_street_graph(base_config)

    summarize_graph(street_graph)

    preview_path = plot_street_graph(street_graph)
    print(f"\nPreview image saved to: {preview_path}")
