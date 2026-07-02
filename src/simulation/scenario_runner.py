"""
scenario_runner.py

Runs all Temporal Buildingness model scenarios.

Scenarios:
- conventional_baseline
- informal_no_anchors
- informal_with_anchors
- high_pedestrian_intensity
- regulation_disruption

This script generates event logs for each scenario and calculates
Temporal Buildingness Index scores across all scenarios.
"""

from copy import deepcopy
from pathlib import Path
import sys

import networkx as nx
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.agents.conventional_vehicle import simulate_conventional_vehicles
from src.agents.informal_transport import simulate_informal_transport
from src.environment.anchor_points import create_anchor_points
from src.environment.demand_field import create_demand_zones
from src.environment.street_graph import load_base_config, create_street_graph
from src.metrics.temporal_buildingness_index import (
    calculate_tbi_for_log_file,
    create_scenario_summary,
    load_tbi_weights,
    plot_tbi_comparison,
    save_tbi_outputs,
    print_tbi_summary,
)


CONFIG_DIR = PROJECT_ROOT / "config"
SCENARIOS_PATH = CONFIG_DIR / "scenarios.yaml"

OUTPUT_LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
OUTPUT_SUMMARIES_DIR = PROJECT_ROOT / "outputs" / "summaries"


def load_scenarios(scenarios_path: Path = SCENARIOS_PATH) -> dict:
    """
    Load scenario definitions from config/scenarios.yaml.
    """

    if not scenarios_path.exists():
        raise FileNotFoundError(f"Scenarios file not found: {scenarios_path}")

    with open(scenarios_path, "r", encoding="utf-8") as file:
        scenarios_config = yaml.safe_load(file)

    return scenarios_config["scenarios"]


def cap_between_zero_and_one(value: float) -> float:
    """
    Keep a value between 0 and 1.
    """

    return max(0.0, min(1.0, float(value)))


def create_scenario_graph(
    base_graph: nx.Graph,
    scenario_settings: dict,
) -> nx.Graph:
    """
    Create a modified graph for a scenario.

    Scenario multipliers change synthetic street attributes.

    demand_multiplier affects:
    - pedestrian_intensity
    - street_edge_activity

    curb_accessibility_multiplier affects:
    - curb_accessibility

    anchor_strength_multiplier affects:
    - anchor_probability
    """

    scenario_graph = deepcopy(base_graph)

    demand_multiplier = scenario_settings.get("demand_multiplier", 1.0)
    curb_multiplier = scenario_settings.get("curb_accessibility_multiplier", 1.0)
    anchor_multiplier = scenario_settings.get("anchor_strength_multiplier", 1.0)

    for _, _, edge_data in scenario_graph.edges(data=True):
        edge_data["pedestrian_intensity"] = cap_between_zero_and_one(
            edge_data["pedestrian_intensity"] * demand_multiplier
        )

        edge_data["street_edge_activity"] = cap_between_zero_and_one(
            edge_data["street_edge_activity"] * demand_multiplier
        )

        edge_data["curb_accessibility"] = cap_between_zero_and_one(
            edge_data["curb_accessibility"] * curb_multiplier
        )

        edge_data["anchor_probability"] = cap_between_zero_and_one(
            edge_data["anchor_probability"] * anchor_multiplier
        )

    return scenario_graph


def create_scenario_demand_zones(
    graph: nx.Graph,
    scenario_settings: dict,
    random_seed: int,
) -> list[dict]:
    """
    Create demand zones for a scenario.
    """

    demand_multiplier = scenario_settings.get("demand_multiplier", 1.0)

    demand_zones = create_demand_zones(
        graph,
        number_of_zones=5,
        random_seed=random_seed,
    )

    for zone in demand_zones:
        zone["demand_intensity"] = round(
            cap_between_zero_and_one(
                zone["demand_intensity"] * demand_multiplier
            ),
            3,
        )

    return demand_zones


def create_scenario_anchor_points(
    graph: nx.Graph,
    scenario_settings: dict,
    random_seed: int,
) -> list[dict]:
    """
    Create anchor points for a scenario.

    If anchor_strength_multiplier is 0, no anchors are created.
    """

    anchor_multiplier = scenario_settings.get("anchor_strength_multiplier", 1.0)

    if anchor_multiplier <= 0:
        return []

    anchor_points = create_anchor_points(
        graph,
        number_of_anchors=5,
        random_seed=random_seed,
    )

    for anchor in anchor_points:
        anchor["anchor_strength"] = round(
            cap_between_zero_and_one(
                anchor["anchor_strength"] * anchor_multiplier
            ),
            3,
        )

    return anchor_points


def save_scenario_logs(
    logs_df: pd.DataFrame,
    scenario_name: str,
) -> Path:
    """
    Save event logs for one scenario.
    """

    OUTPUT_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_LOGS_DIR / f"{scenario_name}_events.csv"

    logs_df.to_csv(output_path, index=False)

    return output_path


def run_single_scenario(
    scenario_name: str,
    scenario_settings: dict,
    base_graph: nx.Graph,
    base_config: dict,
    scenario_number: int,
) -> Path:
    """
    Run one scenario and return the path to its event log.
    """

    print(f"\nRunning scenario: {scenario_name}")

    random_seed = base_config["simulation"]["random_seed"] + scenario_number

    scenario_graph = create_scenario_graph(
        base_graph,
        scenario_settings,
    )

    agent_type = scenario_settings["agent_type"]

    if agent_type == "conventional_vehicle":
        logs_df = simulate_conventional_vehicles(
            graph=scenario_graph,
            config=base_config,
        )

    elif agent_type == "informal_transport":
        demand_zones = create_scenario_demand_zones(
            graph=scenario_graph,
            scenario_settings=scenario_settings,
            random_seed=random_seed,
        )

        anchor_points = create_scenario_anchor_points(
            graph=scenario_graph,
            scenario_settings=scenario_settings,
            random_seed=random_seed,
        )

        logs_df = simulate_informal_transport(
            graph=scenario_graph,
            demand_zones=demand_zones,
            anchor_points=anchor_points,
            config=base_config,
        )

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    logs_df["scenario_name"] = scenario_name

    log_path = save_scenario_logs(
        logs_df,
        scenario_name,
    )

    print(f"Saved logs: {log_path}")
    print(f"Total event logs: {len(logs_df)}")

    if "state" in logs_df.columns:
        print("State counts:")
        print(logs_df["state"].value_counts())

    return log_path


def run_all_scenarios() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all scenarios and calculate TBI.
    """

    base_config = load_base_config()
    base_graph = create_street_graph(base_config)
    scenarios = load_scenarios()
    weights = load_tbi_weights()

    all_agent_scores = []

    for scenario_number, (scenario_name, scenario_settings) in enumerate(
        scenarios.items(),
        start=1,
    ):
        log_path = run_single_scenario(
            scenario_name=scenario_name,
            scenario_settings=scenario_settings,
            base_graph=base_graph,
            base_config=base_config,
            scenario_number=scenario_number,
        )

        scenario_scores = calculate_tbi_for_log_file(
            log_path=log_path,
            scenario_name=scenario_name,
            weights=weights,
        )

        all_agent_scores.append(scenario_scores)

    all_agent_scores_df = pd.concat(
        all_agent_scores,
        ignore_index=True,
    )

    scenario_summary_df = create_scenario_summary(all_agent_scores_df)

    agent_scores_path, scenario_summary_path = save_tbi_outputs(
        all_agent_scores_df,
        scenario_summary_df,
    )

    figure_path = plot_tbi_comparison(scenario_summary_df)

    print_tbi_summary(scenario_summary_df)

    print(f"\nAll scenario agent scores saved to: {agent_scores_path}")
    print(f"All scenario summary saved to: {scenario_summary_path}")
    print(f"Scenario TBI comparison figure saved to: {figure_path}")

    return all_agent_scores_df, scenario_summary_df


if __name__ == "__main__":
    run_all_scenarios()
