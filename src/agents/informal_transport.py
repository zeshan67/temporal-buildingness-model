"""
informal_transport.py

Creates and tests the informal transport agent.

The informal transport agent represents movement that can produce
Temporal Buildingness through:
- anchoring
- searching for pedestrian demand
- boarding threshold
- passenger occupancy / temporary interior
- temporary pause
- alighting threshold
- repositioning toward anchors

This is still a theoretical model. It does not use real-world data,
GPS traces, image data, interviews, or observed human-subject behavior.
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

from src.agents.state_machine import InformalTransportState
from src.environment.anchor_points import create_anchor_points
from src.environment.demand_field import create_demand_zones
from src.environment.street_graph import load_base_config, create_street_graph
from src.movement.routing import (
    choose_random_destination,
    get_shortest_route,
)


OUTPUT_LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


@dataclass
class InformalTransport:
    """
    Informal transport agent.

    This agent does not only move from origin to destination.
    It performs a sequence of buildingness-relevant behaviors.
    """

    agent_id: str
    graph: nx.Graph
    demand_zones: list[dict]
    anchor_points: list[dict]
    rng: np.random.Generator

    max_speed: float
    slow_speed: float
    capacity: int
    threshold_duration_min: int
    threshold_duration_max: int
    max_anchor_wait_time: int
    demand_detection_radius: int

    demand_sensitivity: float = 0.85
    waiting_tendency: float = 0.70
    curb_sensitivity: float = 0.70

    current_node: str = ""
    state: str = InformalTransportState.IDLE_AT_ANCHOR.value

    route: list[str] = field(default_factory=list)
    route_index: int = 0

    target_node: str = ""
    target_demand_zone: str = ""
    target_anchor: str = ""

    speed: float = 0.0
    occupancy: int = 0
    trip_id: int = 1

    wait_timer: int = 0
    threshold_timer: int = 0
    threshold_duration: int = 0
    pause_timer: int = 0
    pause_duration: int = 0
    completed_cycles: int = 0

    def __post_init__(self) -> None:
        """
        Place the agent at a starting anchor.
        """

        if self.anchor_points:
            starting_anchor = self.rng.choice(self.anchor_points)
            self.current_node = starting_anchor["node_id"]
            self.target_anchor = starting_anchor["anchor_id"]
        else:
            self.current_node = str(self.rng.choice(list(self.graph.nodes)))

        self.route = [self.current_node]
        self.route_index = 0
        self.state = InformalTransportState.IDLE_AT_ANCHOR.value
        self.speed = 0.0

    def set_route_to(self, destination: str) -> None:
        """
        Set a shortest route from the current node to a destination.
        """

        self.target_node = destination

        if self.current_node == destination:
            self.route = [self.current_node]
            self.route_index = 0
            return

        self.route = get_shortest_route(
            self.graph,
            self.current_node,
            destination,
        )
        self.route_index = 0

    def move_along_route(self) -> str:
        """
        Move one node forward along the current route.

        Returns:
            current_edge as a readable string.
        """

        if len(self.route) < 2:
            return ""

        if self.route_index >= len(self.route) - 1:
            return ""

        previous_node = self.current_node

        self.route_index += 1
        self.current_node = self.route[self.route_index]

        return f"{previous_node}->{self.current_node}"

    def reached_target(self) -> bool:
        """
        Check whether the agent has reached the end of its current route.
        """

        if len(self.route) <= 1:
            return self.current_node == self.target_node

        return self.route_index >= len(self.route) - 1

    def is_near_anchor(self) -> bool:
        """
        Check whether the current node is one of the synthetic anchor points.
        """

        anchor_node_ids = {
            anchor["node_id"]
            for anchor in self.anchor_points
        }

        return self.current_node in anchor_node_ids

    def is_near_active_edge(self) -> bool:
        """
        Check whether the current node touches active street edges.
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

    def get_current_congestion(self) -> float:
        """
        Return the mean congestion level around the current node.
        """

        connected_edges = self.graph.edges(self.current_node, data=True)

        congestion_scores = [
            edge_data["congestion_level"]
            for _, _, edge_data in connected_edges
        ]

        if not congestion_scores:
            return 0.0

        return float(np.mean(congestion_scores))

    def select_nearest_anchor(self) -> dict:
        """
        Select the nearest anchor point using shortest-path distance.
        """

        best_anchor = None
        best_distance = float("inf")

        for anchor in self.anchor_points:
            try:
                distance = nx.shortest_path_length(
                    self.graph,
                    self.current_node,
                    anchor["node_id"],
                )
            except nx.NetworkXNoPath:
                continue

            if distance < best_distance:
                best_distance = distance
                best_anchor = anchor

        if best_anchor is None:
            best_anchor = self.rng.choice(self.anchor_points)

        return best_anchor

    def detect_demand(
        self,
        step_number: int,
        allow_distant: bool = False,
    ) -> dict | None:
        """
        Detect a demand zone.

        If allow_distant is False, the agent only detects demand within
        the demand detection radius.

        If allow_distant is True, the agent can select farther demand,
        but distance reduces the score.
        """

        candidates = []

        for zone in self.demand_zones:
            try:
                distance = nx.shortest_path_length(
                    self.graph,
                    self.current_node,
                    zone["node_id"],
                )
            except nx.NetworkXNoPath:
                continue

            within_radius = distance <= self.demand_detection_radius

            if not allow_distant and not within_radius:
                continue

            is_peak_time = (
                zone["peak_start"] <= step_number <= zone["peak_end"]
            )

            peak_multiplier = 1.20 if is_peak_time else 0.80

            distance_penalty = 1.0 / (1.0 + 0.15 * distance)

            score = (
                zone["demand_intensity"]
                * peak_multiplier
                * distance_penalty
                * self.demand_sensitivity
            )

            candidates.append((zone, score))

        if not candidates:
            return None

        candidates = sorted(
            candidates,
            key=lambda item: item[1],
            reverse=True,
        )

        best_zone, best_score = candidates[0]

        detection_probability = min(0.95, best_score)

        if self.rng.random() <= detection_probability:
            return best_zone

        return None

    def choose_search_neighbor(self) -> str:
        """
        Choose the next node while searching for demand.

        The agent is attracted to:
        - pedestrian intensity
        - curb accessibility
        - street-edge activity
        - anchor probability
        """

        neighbors = list(self.graph.neighbors(self.current_node))

        if not neighbors:
            return self.current_node

        scored_neighbors = []

        for neighbor in neighbors:
            edge_data = self.graph[self.current_node][neighbor]

            score = (
                0.30 * edge_data["pedestrian_intensity"]
                + 0.25 * edge_data["curb_accessibility"]
                + 0.25 * edge_data["street_edge_activity"]
                + 0.20 * edge_data["anchor_probability"]
            )

            scored_neighbors.append((neighbor, score))

        scored_neighbors = sorted(
            scored_neighbors,
            key=lambda item: item[1],
            reverse=True,
        )

        # Usually choose the best neighbor, sometimes wander randomly.
        if self.rng.random() < 0.85:
            return scored_neighbors[0][0]

        return str(self.rng.choice(neighbors))

    def move_directly_to_neighbor(self, next_node: str) -> str:
        """
        Move directly to a neighboring node during searching behavior.
        """

        if next_node == self.current_node:
            return ""

        previous_node = self.current_node
        self.current_node = next_node

        self.route = [self.current_node]
        self.route_index = 0

        return f"{previous_node}->{self.current_node}"

    def begin_threshold_state(self, threshold_state: str) -> None:
        """
        Start a boarding or alighting threshold.
        """

        self.state = threshold_state
        self.threshold_timer = 0

        self.threshold_duration = int(
            self.rng.integers(
                self.threshold_duration_min,
                self.threshold_duration_max + 1,
            )
        )

        self.speed = 0.0

    def maybe_start_temporary_pause(self) -> bool:
        """
        Decide whether the agent pauses while moving with a passenger.

        Pauses become more likely in high congestion.
        """

        congestion = self.get_current_congestion()

        if congestion >= 0.65:
            pause_probability = 0.25
        else:
            pause_probability = 0.04

        if self.rng.random() < pause_probability:
            self.state = InformalTransportState.TEMPORARY_PAUSE.value
            self.pause_timer = 0
            self.pause_duration = int(self.rng.integers(2, 8))
            self.speed = 0.0
            return True

        return False

    def select_passenger_destination(self) -> str:
        """
        Select a destination for the simulated passenger.

        First version:
        - usually send the passenger to another demand zone
        - otherwise choose a random node
        """

        possible_destinations = [
            zone["node_id"]
            for zone in self.demand_zones
            if zone["node_id"] != self.current_node
        ]

        if possible_destinations and self.rng.random() < 0.75:
            return str(self.rng.choice(possible_destinations))

        return choose_random_destination(
            self.graph,
            self.current_node,
            self.rng,
        )

    def step(
        self,
        step_number: int,
        timestep_seconds: int,
    ) -> dict:
        """
        Run one timestep of the informal transport agent.

        Returns:
            event log dictionary.
        """

        current_edge = ""
        threshold_event = False
        pause_event = False
        cycle_completed = False

        if self.state == InformalTransportState.IDLE_AT_ANCHOR.value:
            self.speed = 0.0
            self.wait_timer += 1

            detected_zone = self.detect_demand(
                step_number,
                allow_distant=False,
            )

            if detected_zone is not None:
                self.target_demand_zone = detected_zone["zone_id"]
                self.set_route_to(detected_zone["node_id"])
                self.state = InformalTransportState.APPROACHING_PASSENGER.value
                self.wait_timer = 0

            elif self.wait_timer >= self.max_anchor_wait_time:
                self.state = InformalTransportState.SEARCHING_FOR_DEMAND.value
                self.wait_timer = 0

        elif self.state == InformalTransportState.SEARCHING_FOR_DEMAND.value:
            detected_zone = self.detect_demand(
                step_number,
                allow_distant=True,
            )

            if detected_zone is not None:
                self.target_demand_zone = detected_zone["zone_id"]
                self.set_route_to(detected_zone["node_id"])
                self.state = InformalTransportState.APPROACHING_PASSENGER.value

            else:
                next_node = self.choose_search_neighbor()
                current_edge = self.move_directly_to_neighbor(next_node)
                self.speed = self.slow_speed

                if self.is_near_anchor() and self.rng.random() < 0.08:
                    self.state = InformalTransportState.IDLE_AT_ANCHOR.value

        elif self.state == InformalTransportState.APPROACHING_PASSENGER.value:
            self.speed = self.slow_speed

            if self.current_node == self.target_node:
                self.begin_threshold_state(
                    InformalTransportState.BOARDING_THRESHOLD.value
                )
                threshold_event = True

            else:
                current_edge = self.move_along_route()

                if self.reached_target():
                    self.begin_threshold_state(
                        InformalTransportState.BOARDING_THRESHOLD.value
                    )
                    threshold_event = True

        elif self.state == InformalTransportState.BOARDING_THRESHOLD.value:
            self.speed = 0.0
            threshold_event = True
            self.threshold_timer += 1

            if self.threshold_timer >= self.threshold_duration:
                self.occupancy = min(self.capacity, self.occupancy + 1)

                destination = self.select_passenger_destination()
                self.set_route_to(destination)

                self.state = InformalTransportState.MOVING_WITH_PASSENGER.value
                self.threshold_timer = 0
                self.speed = self.max_speed

        elif self.state == InformalTransportState.MOVING_WITH_PASSENGER.value:
            self.speed = self.max_speed

            pause_started = self.maybe_start_temporary_pause()

            if pause_started:
                pause_event = True

            else:
                current_edge = self.move_along_route()

                if self.reached_target():
                    self.begin_threshold_state(
                        InformalTransportState.ALIGHTING_THRESHOLD.value
                    )
                    threshold_event = True

        elif self.state == InformalTransportState.TEMPORARY_PAUSE.value:
            self.speed = 0.0
            pause_event = True
            self.pause_timer += 1

            if self.pause_timer >= self.pause_duration:
                self.state = InformalTransportState.MOVING_WITH_PASSENGER.value
                self.pause_timer = 0

        elif self.state == InformalTransportState.ALIGHTING_THRESHOLD.value:
            self.speed = 0.0
            threshold_event = True
            self.threshold_timer += 1

            if self.threshold_timer >= self.threshold_duration:
                self.occupancy = 0
                self.completed_cycles += 1
                cycle_completed = True

                nearest_anchor = self.select_nearest_anchor()
                self.target_anchor = nearest_anchor["anchor_id"]
                self.set_route_to(nearest_anchor["node_id"])

                self.state = InformalTransportState.REPOSITIONING.value
                self.threshold_timer = 0
                self.trip_id += 1

        elif self.state == InformalTransportState.REPOSITIONING.value:
            self.speed = self.slow_speed

            if self.current_node == self.target_node:
                self.state = InformalTransportState.IDLE_AT_ANCHOR.value
                self.speed = 0.0

            else:
                current_edge = self.move_along_route()

                if self.reached_target():
                    self.state = InformalTransportState.IDLE_AT_ANCHOR.value
                    self.speed = 0.0

        event_log = {
            "step": step_number,
            "time_seconds": step_number * timestep_seconds,
            "agent_id": self.agent_id,
            "agent_type": "informal_transport",
            "trip_id": self.trip_id,
            "state": self.state,
            "current_node": self.current_node,
            "current_edge": current_edge,
            "target_node": self.target_node,
            "target_demand_zone": self.target_demand_zone,
            "target_anchor": self.target_anchor,
            "occupancy": self.occupancy,
            "speed": self.speed,
            "near_anchor": self.is_near_anchor(),
            "near_active_edge": self.is_near_active_edge(),
            "threshold_event": threshold_event,
            "pause_event": pause_event,
            "cycle_completed": cycle_completed,
            "completed_cycles": self.completed_cycles,
        }

        return event_log


def create_informal_transport_agents(
    graph: nx.Graph,
    demand_zones: list[dict],
    anchor_points: list[dict],
    count: int,
    config: dict,
) -> list[InformalTransport]:
    """
    Create multiple informal transport agents.
    """

    rng = np.random.default_rng(config["simulation"]["random_seed"])

    informal_config = config["informal_transport"]

    agents = []

    for agent_number in range(1, count + 1):
        agent = InformalTransport(
            agent_id=f"IT_{agent_number}",
            graph=graph,
            demand_zones=demand_zones,
            anchor_points=anchor_points,
            rng=rng,
            max_speed=informal_config["max_speed"],
            slow_speed=informal_config["slow_speed"],
            capacity=informal_config["capacity"],
            threshold_duration_min=informal_config["threshold_duration_min"],
            threshold_duration_max=informal_config["threshold_duration_max"],
            max_anchor_wait_time=informal_config["max_anchor_wait_time"],
            demand_detection_radius=informal_config["demand_detection_radius"],
            demand_sensitivity=informal_config.get("demand_sensitivity", 0.85),
            waiting_tendency=informal_config.get("waiting_tendency", 0.70),
            curb_sensitivity=informal_config.get("curb_sensitivity", 0.70),
        )

        agents.append(agent)

    return agents


def simulate_informal_transport(
    graph: nx.Graph,
    demand_zones: list[dict],
    anchor_points: list[dict],
    config: dict,
) -> pd.DataFrame:
    """
    Run a simple informal transport simulation.
    """

    total_steps = config["simulation"]["total_steps"]
    timestep_seconds = config["simulation"]["timestep_seconds"]

    agent_count = config["agents"]["informal_transport_count"]

    agents = create_informal_transport_agents(
        graph=graph,
        demand_zones=demand_zones,
        anchor_points=anchor_points,
        count=agent_count,
        config=config,
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


def save_informal_transport_logs(logs_df: pd.DataFrame) -> Path:
    """
    Save informal transport event logs as CSV.
    """

    OUTPUT_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_LOGS_DIR / "informal_transport_test_events.csv"

    logs_df.to_csv(output_path, index=False)

    return output_path


def plot_informal_transport_paths(
    graph: nx.Graph,
    logs_df: pd.DataFrame,
    max_agents_to_plot: int = 5,
) -> Path:
    """
    Plot movement paths for a few informal transport agents.
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

    plt.title("Informal Transport Movement Preview")
    plt.axis("equal")
    plt.axis("off")
    plt.legend(loc="upper right", fontsize=8)

    output_path = OUTPUT_FIGURES_DIR / "informal_transport_paths_preview.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def print_simulation_summary(logs_df: pd.DataFrame) -> None:
    """
    Print basic simulation summary.
    """

    print("Informal transport simulation completed.")
    print(f"Total event logs: {len(logs_df)}")
    print(f"Number of agents: {logs_df['agent_id'].nunique()}")
    print(f"Number of timesteps: {logs_df['step'].nunique()}")

    print("\nState counts:")
    print(logs_df["state"].value_counts())

    print("\nBuildingness event counts:")
    print(f"Threshold events: {int(logs_df['threshold_event'].sum())}")
    print(f"Pause events: {int(logs_df['pause_event'].sum())}")
    print(f"Completed cycles: {int(logs_df['cycle_completed'].sum())}")
    print(f"Occupied timesteps: {int((logs_df['occupancy'] > 0).sum())}")
    print(f"Near-anchor timesteps: {int(logs_df['near_anchor'].sum())}")
    print(f"Near-active-edge timesteps: {int(logs_df['near_active_edge'].sum())}")

    print("\nSample logs:")
    print(logs_df.head())


if __name__ == "__main__":
    base_config = load_base_config()
    street_graph = create_street_graph(base_config)

    demand_zones = create_demand_zones(
        street_graph,
        number_of_zones=5,
        random_seed=base_config["simulation"]["random_seed"],
    )

    anchor_points = create_anchor_points(
        street_graph,
        number_of_anchors=5,
        random_seed=base_config["simulation"]["random_seed"],
    )

    informal_logs = simulate_informal_transport(
        graph=street_graph,
        demand_zones=demand_zones,
        anchor_points=anchor_points,
        config=base_config,
    )

    print_simulation_summary(informal_logs)

    csv_path = save_informal_transport_logs(informal_logs)
    figure_path = plot_informal_transport_paths(
        street_graph,
        informal_logs,
    )

    print(f"\nInformal transport logs saved to: {csv_path}")
    print(f"Informal transport path preview saved to: {figure_path}")
