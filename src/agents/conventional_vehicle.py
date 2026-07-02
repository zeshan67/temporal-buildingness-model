"""
conventional_vehicle.py

Creates and tests the conventional vehicle baseline agent.

The conventional vehicle represents pure vehicular movement:
- origin-to-destination routing
- shortest-path movement
- limited random stopping
- no informal anchoring
- no threshold behavior
- no demand-seeking behavior
- no passenger-interior logic
"""

from dataclasses import dataclass, field
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.environment.street_graph import load_base_config, create_street_graph
from src.movement.routing import (
    choose_random_origin_destination,
    choose_random_destination,
    get_shortest_route,
)


OUTPUT_LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


@dataclass
class ConventionalVehicle:
    """
    Conventional vehicle baseline agent.
    """

    agent_id: str
    graph: nx.Graph
    rng: np.random.Generator
    max_speed: float
    stop_probability: float

    origin: str = ""
    destination: str = ""
    route: list[str] = field(default_factory=list)
    route_index: int = 0
    state: str = "MOVING"
    speed: float = 0.0
    trip_id: int = 1

    def __post_init__(self) -> None:
        """
        Create the first origin-destination trip.
        """

        self.origin, self.destination = choose_random_origin_destination(
            self.graph,
            self.rng,
        )

        self.route = get_shortest_route(
            self.graph,
            self.origin,
            self.destination,
        )

        self.route_index = 0
        self.state = "MOVING"
        self.speed = self.max_speed

    @property
    def current_node(self) -> str:
        """
        Return the current node of the vehicle.
        """

        return self.route[self.route_index]

    def is_near_active_edge(self) -> bool:
        """
        Check whether the current node is connected to active street edges.

        This does not mean the conventional vehicle is producing buildingness.
        It only records whether it is passing through an active street-edge area.
        """

        connected_edges = self.graph.edges(self.current_node, data=True)

        activity_scores = [
            edge_data["street_edge_activity"]
            for _, _, edge_data in connected_edges
        ]

        if not activity_scores:
            return False

        mean_activity = float(np.mean(activity_scores))

        return mean_activity >= 0.70

    def start_new_trip(self) -> None:
        """
        Start a new origin-destination trip from the current node.
        """

        self.origin = self.current_node
        self.destination = choose_random_destination(
            self.graph,
            self.origin,
            self.rng,
        )

        self.route = get_shortest_route(
            self.graph,
            self.origin,
            self.destination,
        )

        self.route_index = 0
        self.trip_id += 1
        self.state = "MOVING"
        self.speed = self.max_speed

    def step(
        self,
        step_number: int,
        timestep_seconds: int,
    ) -> dict:
        """
        Move the vehicle forward by one simulation step.

        Returns:
            event log dictionary
        """

        previous_node = self.current_node
        current_edge = ""

        pause_event = False

        # Very small chance of a temporary stop.
        if self.rng.random() < self.stop_probability:
            self.state = "TEMPORARY_STOP"
            self.speed = 0.0
            pause_event = True

        else:
            if self.route_index < len(self.route) - 1:
                next_node = self.route[self.route_index + 1]
                self.route_index += 1

                current_edge = f"{previous_node}->{next_node}"
                self.state = "MOVING"
                self.speed = self.max_speed

            if self.route_index == len(self.route) - 1:
                self.state = "ARRIVED"
                self.speed = 0.0

        event_log = {
            "step": step_number,
            "time_seconds": step_number * timestep_seconds,
            "agent_id": self.agent_id,
            "agent_type": "conventional_vehicle",
            "trip_id": self.trip_id,
            "state": self.state,
            "current_node": self.current_node,
            "current_edge": current_edge,
            "origin": self.origin,
            "destination": self.destination,
            "occupancy": 0,
            "speed": self.speed,
            "near_anchor": False,
            "near_active_edge": self.is_near_active_edge(),
            "threshold_event": False,
            "pause_event": pause_event,
            "cycle_completed": False,
        }

        # After logging arrival, immediately prepare a new trip for the next step.
        if self.state == "ARRIVED":
            self.start_new_trip()

        return event_log


def create_conventional_vehicle_agents(
    graph: nx.Graph,
    count: int,
    max_speed: float,
    stop_probability: float,
    random_seed: int,
) -> list[ConventionalVehicle]:
    """
    Create multiple conventional vehicle agents.
    """

    rng = np.random.default_rng(random_seed)

    agents = []

    for agent_number in range(1, count + 1):
        agent = ConventionalVehicle(
            agent_id=f"CV_{agent_number}",
            graph=graph,
            rng=rng,
            max_speed=max_speed,
            stop_probability=stop_probability,
        )

        agents.append(agent)

    return agents


def simulate_conventional_vehicles(
    graph: nx.Graph,
    config: dict,
) -> pd.DataFrame:
    """
    Run a simple conventional vehicle simulation.
    """

    total_steps = config["simulation"]["total_steps"]
    timestep_seconds = config["simulation"]["timestep_seconds"]
    random_seed = config["simulation"]["random_seed"]

    vehicle_count = config["agents"]["conventional_vehicle_count"]
    max_speed = config["conventional_vehicle"]["max_speed"]
    stop_probability = config["conventional_vehicle"]["stop_probability"]

    agents = create_conventional_vehicle_agents(
        graph=graph,
        count=vehicle_count,
        max_speed=max_speed,
        stop_probability=stop_probability,
        random_seed=random_seed,
    )

    logs = []

    for step_number in range(total_steps):
        for agent in agents:
            event_log = agent.step(
                step_number=step_number,
                timestep_seconds=timestep_seconds,
            )
            logs.append(event_log)

    logs_df = pd.DataFrame(logs)

    return logs_df


def save_conventional_vehicle_logs(logs_df: pd.DataFrame) -> Path:
    """
    Save conventional vehicle logs as CSV.
    """

    OUTPUT_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_LOGS_DIR / "conventional_vehicle_test_events.csv"

    logs_df.to_csv(output_path, index=False)

    return output_path


def plot_conventional_vehicle_paths(
    graph: nx.Graph,
    logs_df: pd.DataFrame,
    max_agents_to_plot: int = 5,
) -> Path:
    """
    Plot paths for a few conventional vehicle agents.
    """

    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    positions = nx.get_node_attributes(graph, "pos")

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
        node_size=20,
    )

    agent_ids = logs_df["agent_id"].unique()[:max_agents_to_plot]

    for agent_id in agent_ids:
        agent_logs = logs_df[logs_df["agent_id"] == agent_id]

        x_values = []
        y_values = []

        for node_id in agent_logs["current_node"]:
            x, y = positions[node_id]
            x_values.append(x)
            y_values.append(y)

        plt.plot(
            x_values,
            y_values,
            linewidth=1.5,
            label=agent_id,
        )

    plt.title("Conventional Vehicle Movement Preview")
    plt.axis("equal")
    plt.axis("off")
    plt.legend(loc="upper right", fontsize=8)

    output_path = OUTPUT_FIGURES_DIR / "conventional_vehicle_paths_preview.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def print_simulation_summary(logs_df: pd.DataFrame) -> None:
    """
    Print basic simulation summary.
    """

    print("Conventional vehicle simulation completed.")
    print(f"Total event logs: {len(logs_df)}")
    print(f"Number of agents: {logs_df['agent_id'].nunique()}")
    print(f"Number of timesteps: {logs_df['step'].nunique()}")

    print("\nState counts:")
    print(logs_df["state"].value_counts())

    print("\nSample logs:")
    print(logs_df.head())


if __name__ == "__main__":
    base_config = load_base_config()
    street_graph = create_street_graph(base_config)

    conventional_logs = simulate_conventional_vehicles(
        graph=street_graph,
        config=base_config,
    )

    print_simulation_summary(conventional_logs)

    csv_path = save_conventional_vehicle_logs(conventional_logs)
    figure_path = plot_conventional_vehicle_paths(
        street_graph,
        conventional_logs,
    )

    print(f"\nConventional vehicle logs saved to: {csv_path}")
    print(f"Conventional vehicle path preview saved to: {figure_path}")
