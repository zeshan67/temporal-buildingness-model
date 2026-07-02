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
- repair or delay states

This version supports transport mode profiles, such as:
- Rickshaw_Dhaka
- Matatu_Nairobi
- Jeepney_Manila
- Colectivo_Pesero_MexicoCity
- TroTro_Accra
- TukTuk_Bangkok
- AutoRickshaw_NewDelhi
- BodaBoda_Kampala
- Jitney_USA

The model remains theoretical. It does not use real-world data,
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
from src.modes.transport_mode_profiles import get_transport_mode_profile
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

    Mode profiles calibrate how each informal transport type behaves.
    """

    agent_id: str
    graph: nx.Graph
    demand_zones: list[dict]
    anchor_points: list[dict]
    rng: np.random.Generator

    mode_name: str
    mode_profile: dict

    max_speed: float
    slow_speed: float
    capacity: int
    threshold_duration_min: int
    threshold_duration_max: int
    max_anchor_wait_time: int
    demand_detection_radius: int

    anchor_preference: float
    demand_sensitivity: float
    curb_sensitivity: float
    waiting_tendency: float
    stop_flexibility: float
    route_formality: float
    terminal_dependence: float
    enclosure_factor: float
    edge_coupling_tendency: float
    repair_delay_probability: float
    movement_family: str

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
    repair_timer: int = 0
    repair_duration: int = 0
    completed_cycles: int = 0

    # A passenger trip is not automatically a buildingness cycle.
    # A buildingness cycle must originate from anchor-based behavior.
    cycle_started_from_anchor: bool = False

    def __post_init__(self) -> None:
        """
        Place the agent at a starting anchor when anchors exist.

        If no anchors exist, the agent starts in searching mode.
        """

        if self.anchor_points:
            starting_anchor = self.rng.choice(self.anchor_points)
            self.current_node = starting_anchor["node_id"]
            self.target_anchor = starting_anchor["anchor_id"]
            self.state = InformalTransportState.IDLE_AT_ANCHOR.value
            self.speed = 0.0
            self.cycle_started_from_anchor = True

        else:
            self.current_node = str(self.rng.choice(list(self.graph.nodes)))
            self.target_anchor = ""
            self.state = InformalTransportState.SEARCHING_FOR_DEMAND.value
            self.speed = self.slow_speed
            self.cycle_started_from_anchor = False

        self.route = [self.current_node]
        self.route_index = 0

    def get_effective_anchor_wait_limit(self) -> int:
        """
        Calculate mode-specific anchor waiting limit.

        Higher waiting tendency and terminal dependence increase anchor waiting.
        Lower values make agents leave anchors sooner to search for demand.
        """

        multiplier = 0.45 + 0.55 * self.waiting_tendency
        multiplier += 0.25 * self.terminal_dependence

        wait_limit = int(self.max_anchor_wait_time * multiplier)

        return max(5, wait_limit)

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

        if not self.anchor_points:
            return False

        anchor_node_ids = {
            anchor["node_id"]
            for anchor in self.anchor_points
        }

        return self.current_node in anchor_node_ids

    def is_near_active_edge(self) -> bool:
        """
        Check whether the current node touches active street edges.

        Mode-specific edge coupling tendency changes how easily a mode couples
        with street-edge activity.
        """

        connected_edges = self.graph.edges(self.current_node, data=True)

        activity_scores = [
            edge_data["street_edge_activity"]
            for _, _, edge_data in connected_edges
        ]

        if not activity_scores:
            return False

        mean_activity = float(np.mean(activity_scores))

        activity_threshold = 0.80 - 0.25 * self.edge_coupling_tendency
        activity_threshold = max(0.40, min(0.80, activity_threshold))

        return mean_activity >= activity_threshold

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

    def select_nearest_anchor(self) -> dict | None:
        """
        Select the nearest anchor point using shortest-path distance.

        If the scenario has no anchors, return None.
        """

        if not self.anchor_points:
            return None

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

        Flexible modes usually detect nearby demand.
        Route-based collective modes can detect more distant demand because
        they operate with stronger corridor/terminal logic.
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

            if self.movement_family == "route_based_collective":
                distance_penalty = 1.0 / (1.0 + 0.08 * distance)
            elif self.movement_family == "route_flexible_collective":
                distance_penalty = 1.0 / (1.0 + 0.11 * distance)
            else:
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

        Mode profile changes the relative importance of these values.
        """

        neighbors = list(self.graph.neighbors(self.current_node))

        if not neighbors:
            return self.current_node

        scored_neighbors = []

        for neighbor in neighbors:
            edge_data = self.graph[self.current_node][neighbor]

            score = (
                0.30 * self.demand_sensitivity * edge_data["pedestrian_intensity"]
                + 0.25 * self.curb_sensitivity * edge_data["curb_accessibility"]
                + 0.25 * self.edge_coupling_tendency * edge_data["street_edge_activity"]
                + 0.20 * self.anchor_preference * edge_data["anchor_probability"]
            )

            # Route-based modes are less likely to wander randomly.
            if self.movement_family == "route_based_collective":
                score += 0.10 * self.route_formality

            scored_neighbors.append((neighbor, score))

        scored_neighbors = sorted(
            scored_neighbors,
            key=lambda item: item[1],
            reverse=True,
        )

        random_wander_probability = 1.0 - self.stop_flexibility
        random_wander_probability = max(0.05, min(0.35, random_wander_probability))

        if self.rng.random() > random_wander_probability:
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

        Larger collective modes usually have longer thresholds.
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

    def maybe_start_repair_or_delay(self) -> bool:
        """
        Decide whether the agent enters a repair or delay state.

        The profile value is interpreted conservatively so that delay is
        occasional, not constant.
        """

        effective_probability = self.repair_delay_probability * 0.04

        if self.rng.random() < effective_probability:
            self.state = InformalTransportState.REPAIR_OR_DELAY.value
            self.repair_timer = 0
            self.repair_duration = int(self.rng.integers(4, 16))
            self.speed = 0.0
            return True

        return False

    def maybe_start_temporary_pause(self) -> bool:
        """
        Decide whether the agent pauses while moving with a passenger.

        Pauses become more likely in high congestion and in modes with high
        stop flexibility.
        """

        congestion = self.get_current_congestion()

        pause_probability = 0.02
        pause_probability += 0.12 * congestion
        pause_probability += 0.06 * self.stop_flexibility
        pause_probability += 0.03 * self.edge_coupling_tendency

        if self.movement_family == "route_based_collective":
            pause_probability += 0.02

        pause_probability = min(0.28, pause_probability)

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

        Route-based collective modes strongly prefer demand-zone destinations.
        Flexible modes can go to demand zones or random destinations.
        """

        possible_destinations = [
            zone["node_id"]
            for zone in self.demand_zones
            if zone["node_id"] != self.current_node
        ]

        if not possible_destinations:
            return choose_random_destination(
                self.graph,
                self.current_node,
                self.rng,
            )

        if self.movement_family == "route_based_collective":
            demand_zone_probability = 0.95
        elif self.movement_family == "route_flexible_collective":
            demand_zone_probability = 0.82
        else:
            demand_zone_probability = 0.60 + 0.25 * self.route_formality

        demand_zone_probability = min(0.98, demand_zone_probability)

        if self.rng.random() < demand_zone_probability:
            return str(self.rng.choice(possible_destinations))

        return choose_random_destination(
            self.graph,
            self.current_node,
            self.rng,
        )

    def should_reposition_to_anchor_after_trip(self) -> bool:
        """
        Decide whether the agent returns to an anchor after alighting.

        Terminal-dependent and anchor-preferring modes are more likely to
        reposition to anchors.
        """

        if not self.anchor_points:
            return False

        probability = (
            0.35 * self.anchor_preference
            + 0.45 * self.terminal_dependence
            + 0.20 * self.route_formality
        )

        probability = max(0.05, min(0.95, probability))

        return self.rng.random() < probability

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
        repair_delay_event = False
        cycle_completed = False

        if self.state == InformalTransportState.IDLE_AT_ANCHOR.value:
            self.speed = 0.0
            self.wait_timer += 1
            self.cycle_started_from_anchor = True

            allow_distant_demand = (
                self.movement_family in {
                    "route_based_collective",
                    "route_flexible_collective",
                }
                and self.rng.random() < self.route_formality
            )

            detected_zone = self.detect_demand(
                step_number,
                allow_distant=allow_distant_demand,
            )

            if detected_zone is not None:
                self.target_demand_zone = detected_zone["zone_id"]
                self.set_route_to(detected_zone["node_id"])
                self.state = InformalTransportState.APPROACHING_PASSENGER.value
                self.wait_timer = 0

            elif self.wait_timer >= self.get_effective_anchor_wait_limit():
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

                if self.is_near_anchor():
                    anchor_stop_probability = 0.05 + 0.20 * self.anchor_preference
                    if self.rng.random() < anchor_stop_probability:
                        self.state = InformalTransportState.IDLE_AT_ANCHOR.value
                        self.cycle_started_from_anchor = True

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

            delay_started = self.maybe_start_repair_or_delay()

            if delay_started:
                repair_delay_event = True

            else:
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

        elif self.state == InformalTransportState.REPAIR_OR_DELAY.value:
            self.speed = 0.0
            repair_delay_event = True
            self.repair_timer += 1

            if self.repair_timer >= self.repair_duration:
                self.state = InformalTransportState.MOVING_WITH_PASSENGER.value
                self.repair_timer = 0

        elif self.state == InformalTransportState.ALIGHTING_THRESHOLD.value:
            self.speed = 0.0
            threshold_event = True
            self.threshold_timer += 1

            if self.threshold_timer >= self.threshold_duration:
                self.occupancy = 0

                should_anchor = self.should_reposition_to_anchor_after_trip()
                nearest_anchor = self.select_nearest_anchor() if should_anchor else None

                if nearest_anchor is not None:
                    self.target_anchor = nearest_anchor["anchor_id"]
                    self.set_route_to(nearest_anchor["node_id"])
                    self.state = InformalTransportState.REPOSITIONING.value

                    if self.cycle_started_from_anchor:
                        self.completed_cycles += 1
                        cycle_completed = True

                else:
                    self.target_anchor = ""
                    self.target_node = ""
                    self.route = [self.current_node]
                    self.route_index = 0
                    self.state = InformalTransportState.SEARCHING_FOR_DEMAND.value
                    cycle_completed = False

                self.threshold_timer = 0
                self.trip_id += 1
                self.cycle_started_from_anchor = False

        elif self.state == InformalTransportState.REPOSITIONING.value:
            self.speed = self.slow_speed

            if self.current_node == self.target_node:
                self.state = InformalTransportState.IDLE_AT_ANCHOR.value
                self.speed = 0.0
                self.cycle_started_from_anchor = True

            else:
                current_edge = self.move_along_route()

                if self.reached_target():
                    self.state = InformalTransportState.IDLE_AT_ANCHOR.value
                    self.speed = 0.0
                    self.cycle_started_from_anchor = True

        event_log = {
            "step": step_number,
            "time_seconds": step_number * timestep_seconds,
            "agent_id": self.agent_id,
            "agent_type": "informal_transport",
            "mode_name": self.mode_name,
            "display_name": self.mode_profile["display_name"],
            "city": self.mode_profile["city"],
            "country": self.mode_profile["country"],
            "movement_family": self.movement_family,
            "trip_id": self.trip_id,
            "state": self.state,
            "current_node": self.current_node,
            "current_edge": current_edge,
            "target_node": self.target_node,
            "target_demand_zone": self.target_demand_zone,
            "target_anchor": self.target_anchor,
            "occupancy": self.occupancy,
            "capacity": self.capacity,
            "speed": self.speed,
            "enclosure_factor": self.enclosure_factor,
            "near_anchor": self.is_near_anchor(),
            "near_active_edge": self.is_near_active_edge(),
            "threshold_event": threshold_event,
            "pause_event": pause_event,
            "repair_delay_event": repair_delay_event,
            "cycle_completed": cycle_completed,
            "completed_cycles": self.completed_cycles,
            "cycle_started_from_anchor": self.cycle_started_from_anchor,
        }

        return event_log


def create_informal_transport_agents(
    graph: nx.Graph,
    demand_zones: list[dict],
    anchor_points: list[dict],
    count: int,
    config: dict,
    mode_name: str = "Rickshaw_Dhaka",
    mode_profile: dict | None = None,
) -> list[InformalTransport]:
    """
    Create multiple informal transport agents using a transport mode profile.
    """

    rng = np.random.default_rng(config["simulation"]["random_seed"])

    if mode_profile is None:
        mode_profile = get_transport_mode_profile(mode_name)

    agents = []

    for agent_number in range(1, count + 1):
        agent = InformalTransport(
            agent_id=f"IT_{agent_number}",
            graph=graph,
            demand_zones=demand_zones,
            anchor_points=anchor_points,
            rng=rng,
            mode_name=mode_name,
            mode_profile=mode_profile,
            max_speed=float(mode_profile["max_speed"]),
            slow_speed=float(mode_profile["slow_speed"]),
            capacity=int(mode_profile["capacity"]),
            threshold_duration_min=int(mode_profile["threshold_duration_min"]),
            threshold_duration_max=int(mode_profile["threshold_duration_max"]),
            max_anchor_wait_time=int(config["informal_transport"]["max_anchor_wait_time"]),
            demand_detection_radius=int(
                config["informal_transport"]["demand_detection_radius"]
            ),
            anchor_preference=float(mode_profile["anchor_preference"]),
            demand_sensitivity=float(mode_profile["demand_sensitivity"]),
            curb_sensitivity=float(mode_profile["curb_sensitivity"]),
            waiting_tendency=float(mode_profile["waiting_tendency"]),
            stop_flexibility=float(mode_profile["stop_flexibility"]),
            route_formality=float(mode_profile["route_formality"]),
            terminal_dependence=float(mode_profile["terminal_dependence"]),
            enclosure_factor=float(mode_profile["enclosure_factor"]),
            edge_coupling_tendency=float(mode_profile["edge_coupling_tendency"]),
            repair_delay_probability=float(mode_profile["repair_delay_probability"]),
            movement_family=str(mode_profile["movement_family"]),
        )

        agents.append(agent)

    return agents


def simulate_informal_transport(
    graph: nx.Graph,
    demand_zones: list[dict],
    anchor_points: list[dict],
    config: dict,
    mode_name: str = "Rickshaw_Dhaka",
    mode_profile: dict | None = None,
) -> pd.DataFrame:
    """
    Run a simple informal transport simulation.
    """

    total_steps = config["simulation"]["total_steps"]
    timestep_seconds = config["simulation"]["timestep_seconds"]

    agent_count = config["agents"]["informal_transport_count"]

    if mode_profile is None:
        mode_profile = get_transport_mode_profile(mode_name)

    agents = create_informal_transport_agents(
        graph=graph,
        demand_zones=demand_zones,
        anchor_points=anchor_points,
        count=agent_count,
        config=config,
        mode_name=mode_name,
        mode_profile=mode_profile,
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


def save_informal_transport_logs(
    logs_df: pd.DataFrame,
    mode_name: str = "Rickshaw_Dhaka",
) -> Path:
    """
    Save informal transport event logs as CSV.
    """

    OUTPUT_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_LOGS_DIR / f"informal_transport_{mode_name}_test_events.csv"

    logs_df.to_csv(output_path, index=False)

    # Also save a generic copy for compatibility with existing TBI scripts.
    generic_output_path = OUTPUT_LOGS_DIR / "informal_transport_test_events.csv"
    logs_df.to_csv(generic_output_path, index=False)

    return output_path


def plot_informal_transport_paths(
    graph: nx.Graph,
    logs_df: pd.DataFrame,
    mode_name: str = "Rickshaw_Dhaka",
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

    display_name = logs_df["display_name"].iloc[0] if "display_name" in logs_df else mode_name
    city = logs_df["city"].iloc[0] if "city" in logs_df else ""

    plt.title(f"{display_name} Movement Preview ({city})")
    plt.axis("equal")
    plt.axis("off")
    plt.legend(loc="upper right", fontsize=8)

    output_path = OUTPUT_FIGURES_DIR / f"informal_transport_{mode_name}_paths_preview.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Also save a generic copy for compatibility.
    generic_output_path = OUTPUT_FIGURES_DIR / "informal_transport_paths_preview.png"
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(graph, positions, edge_color="lightgray", width=1)
    nx.draw_networkx_nodes(graph, positions, node_size=20)

    for agent_id in agent_ids:
        agent_logs = logs_df[logs_df["agent_id"] == agent_id]
        x_values = []
        y_values = []

        for node_id in agent_logs["current_node"]:
            x, y = positions[node_id]
            x_values.append(x)
            y_values.append(y)

        plt.plot(x_values, y_values, linewidth=1.5, label=agent_id)

    plt.title(f"{display_name} Movement Preview ({city})")
    plt.axis("equal")
    plt.axis("off")
    plt.legend(loc="upper right", fontsize=8)
    plt.savefig(generic_output_path, dpi=300, bbox_inches="tight")
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

    if "mode_name" in logs_df.columns:
        print(f"Mode: {logs_df['mode_name'].iloc[0]}")
        print(f"City: {logs_df['city'].iloc[0]}")
        print(f"Movement family: {logs_df['movement_family'].iloc[0]}")
        print(f"Enclosure factor: {logs_df['enclosure_factor'].iloc[0]}")

    print("\nState counts:")
    print(logs_df["state"].value_counts())

    print("\nBuildingness event counts:")
    print(f"Threshold events: {int(logs_df['threshold_event'].sum())}")
    print(f"Pause events: {int(logs_df['pause_event'].sum())}")

    if "repair_delay_event" in logs_df.columns:
        print(f"Repair/delay events: {int(logs_df['repair_delay_event'].sum())}")

    print(f"Completed cycles: {int(logs_df['cycle_completed'].sum())}")
    print(f"Occupied timesteps: {int((logs_df['occupancy'] > 0).sum())}")
    print(f"Near-anchor timesteps: {int(logs_df['near_anchor'].sum())}")
    print(f"Near-active-edge timesteps: {int(logs_df['near_active_edge'].sum())}")

    print("\nSample logs:")
    print(logs_df.head())


if __name__ == "__main__":
    test_mode_name = "Rickshaw_Dhaka"
    test_mode_profile = get_transport_mode_profile(test_mode_name)

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
        mode_name=test_mode_name,
        mode_profile=test_mode_profile,
    )

    print_simulation_summary(informal_logs)

    csv_path = save_informal_transport_logs(
        informal_logs,
        mode_name=test_mode_name,
    )

    figure_path = plot_informal_transport_paths(
        street_graph,
        informal_logs,
        mode_name=test_mode_name,
    )

    print(f"\nInformal transport logs saved to: {csv_path}")
    print(f"Informal transport path preview saved to: {figure_path}")
