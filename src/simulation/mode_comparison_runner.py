"""
mode_comparison_runner.py

Runs a comparative Temporal Buildingness simulation across multiple
city-tied informal transport modes.

Modes currently included:
- Rickshaw_Dhaka
- Matatu_Nairobi
- Jeepney_Manila
- Colectivo_Pesero_MexicoCity
- TroTro_Accra
- TukTuk_Bangkok
- AutoRickshaw_NewDelhi
- BodaBoda_Kampala
- Jitney_USA

This runner keeps the grid structure synthetic and consistent, but applies
mode-specific city context multipliers and behavior profiles.

The purpose is to compare how different informal public transport modes
produce different forms of Temporal Buildingness.
"""

from copy import deepcopy
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


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
    print_tbi_summary,
)
from src.modes.transport_mode_profiles import (
    load_transport_modes,
    transport_modes_to_dataframe,
)


OUTPUT_LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
OUTPUT_SUMMARIES_DIR = PROJECT_ROOT / "outputs" / "summaries"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def cap_between_zero_and_one(value: float) -> float:
    """
    Keep a value between 0 and 1.
    """

    return max(0.0, min(1.0, float(value)))


def create_mode_context_graph(
    base_graph: nx.Graph,
    mode_profile: dict,
) -> nx.Graph:
    """
    Create a graph modified by the city context of one transport mode.

    The grid structure remains the same.
    Only synthetic edge attributes are modified.

    Context affects:
    - pedestrian intensity
    - congestion
    - curb accessibility
    - anchor probability
    """

    mode_graph = deepcopy(base_graph)

    context = mode_profile["context"]

    pedestrian_multiplier = context["pedestrian_intensity_multiplier"]
    congestion_multiplier = context["congestion_multiplier"]
    curb_multiplier = context["curb_accessibility_multiplier"]
    anchor_multiplier = context["anchor_strength_multiplier"]

    for _, _, edge_data in mode_graph.edges(data=True):
        edge_data["pedestrian_intensity"] = cap_between_zero_and_one(
            edge_data["pedestrian_intensity"] * pedestrian_multiplier
        )

        edge_data["street_edge_activity"] = cap_between_zero_and_one(
            edge_data["street_edge_activity"] * pedestrian_multiplier
        )

        edge_data["congestion_level"] = cap_between_zero_and_one(
            edge_data["congestion_level"] * congestion_multiplier
        )

        edge_data["curb_accessibility"] = cap_between_zero_and_one(
            edge_data["curb_accessibility"] * curb_multiplier
        )

        edge_data["anchor_probability"] = cap_between_zero_and_one(
            edge_data["anchor_probability"] * anchor_multiplier
        )

    return mode_graph


def create_mode_demand_zones(
    graph: nx.Graph,
    mode_profile: dict,
    random_seed: int,
    number_of_zones: int = 5,
) -> list[dict]:
    """
    Create demand zones calibrated by the mode's city context.
    """

    context = mode_profile["context"]
    pedestrian_multiplier = context["pedestrian_intensity_multiplier"]

    demand_zones = create_demand_zones(
        graph,
        number_of_zones=number_of_zones,
        random_seed=random_seed,
    )

    for zone in demand_zones:
        zone["demand_intensity"] = round(
            cap_between_zero_and_one(
                zone["demand_intensity"] * pedestrian_multiplier
            ),
            3,
        )

    return demand_zones


def create_mode_anchor_points(
    graph: nx.Graph,
    mode_profile: dict,
    random_seed: int,
    number_of_anchors: int = 5,
) -> list[dict]:
    """
    Create anchor points calibrated by the mode's city context.

    Regulation pressure reduces effective anchor strength.
    This does not remove anchors completely; it weakens their attraction.
    """

    context = mode_profile["context"]

    anchor_multiplier = context["anchor_strength_multiplier"]
    regulation_pressure = context["regulation_pressure"]

    effective_anchor_multiplier = anchor_multiplier * (1.0 - 0.45 * regulation_pressure)

    anchor_points = create_anchor_points(
        graph,
        number_of_anchors=number_of_anchors,
        random_seed=random_seed,
    )

    for anchor in anchor_points:
        anchor["anchor_strength"] = round(
            cap_between_zero_and_one(
                anchor["anchor_strength"] * effective_anchor_multiplier
            ),
            3,
        )

    return anchor_points


def save_mode_logs(
    logs_df: pd.DataFrame,
    mode_name: str,
) -> Path:
    """
    Save logs for one transport mode.
    """

    OUTPUT_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_LOGS_DIR / f"mode_{mode_name}_events.csv"

    logs_df.to_csv(output_path, index=False)

    return output_path


def save_conventional_baseline_logs(
    logs_df: pd.DataFrame,
) -> Path:
    """
    Save conventional baseline logs for comparison.
    """

    OUTPUT_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_LOGS_DIR / "mode_comparison_conventional_baseline_events.csv"

    logs_df.to_csv(output_path, index=False)

    return output_path


def run_conventional_baseline(
    base_graph: nx.Graph,
    base_config: dict,
) -> Path:
    """
    Run the conventional vehicle baseline once.
    """

    print("\nRunning baseline: conventional vehicle")

    logs_df = simulate_conventional_vehicles(
        graph=base_graph,
        config=base_config,
    )

    logs_df["scenario_name"] = "conventional_baseline"

    log_path = save_conventional_baseline_logs(logs_df)

    print(f"Saved logs: {log_path}")
    print(f"Total event logs: {len(logs_df)}")

    print("State counts:")
    print(logs_df["state"].value_counts())

    return log_path


def run_single_mode(
    mode_name: str,
    mode_profile: dict,
    base_graph: nx.Graph,
    base_config: dict,
    mode_number: int,
) -> Path:
    """
    Run one transport mode simulation.
    """

    display_name = mode_profile["display_name"]
    city = mode_profile["city"]
    country = mode_profile["country"]
    movement_family = mode_profile["movement_family"]

    print(f"\nRunning mode: {mode_name}")
    print(f"Display name: {display_name}")
    print(f"City: {city}, {country}")
    print(f"Movement family: {movement_family}")

    random_seed = base_config["simulation"]["random_seed"] + mode_number

    mode_graph = create_mode_context_graph(
        base_graph,
        mode_profile,
    )

    demand_zones = create_mode_demand_zones(
        graph=mode_graph,
        mode_profile=mode_profile,
        random_seed=random_seed,
        number_of_zones=5,
    )

    anchor_points = create_mode_anchor_points(
        graph=mode_graph,
        mode_profile=mode_profile,
        random_seed=random_seed,
        number_of_anchors=5,
    )

    logs_df = simulate_informal_transport(
        graph=mode_graph,
        demand_zones=demand_zones,
        anchor_points=anchor_points,
        config=base_config,
        mode_name=mode_name,
        mode_profile=mode_profile,
    )

    logs_df["scenario_name"] = mode_name

    log_path = save_mode_logs(
        logs_df,
        mode_name,
    )

    print(f"Saved logs: {log_path}")
    print(f"Total event logs: {len(logs_df)}")

    print("State counts:")
    print(logs_df["state"].value_counts())

    print("Buildingness event counts:")
    print(f"Threshold events: {int(logs_df['threshold_event'].sum())}")
    print(f"Pause events: {int(logs_df['pause_event'].sum())}")
    print(f"Repair/delay events: {int(logs_df['repair_delay_event'].sum())}")
    print(f"Completed cycles: {int(logs_df['cycle_completed'].sum())}")
    print(f"Occupied timesteps: {int((logs_df['occupancy'] > 0).sum())}")
    print(f"Near-anchor timesteps: {int(logs_df['near_anchor'].sum())}")
    print(f"Near-active-edge timesteps: {int(logs_df['near_active_edge'].sum())}")

    return log_path


def save_mode_comparison_outputs(
    agent_scores_df: pd.DataFrame,
    mode_summary_df: pd.DataFrame,
) -> tuple[Path, Path]:
    """
    Save mode comparison TBI outputs.
    """

    OUTPUT_SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    agent_scores_path = OUTPUT_SUMMARIES_DIR / "mode_comparison_agent_scores.csv"
    mode_summary_path = OUTPUT_SUMMARIES_DIR / "mode_comparison_summary.csv"

    agent_scores_df.to_csv(agent_scores_path, index=False)
    mode_summary_df.to_csv(mode_summary_path, index=False)

    return agent_scores_path, mode_summary_path


def plot_mode_tbi_comparison(
    mode_summary_df: pd.DataFrame,
) -> Path:
    """
    Plot mean TBI comparison across modes.
    """

    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = mode_summary_df.copy()

    plot_df = plot_df.sort_values("mean_TBI", ascending=False)

    labels = []

    for _, row in plot_df.iterrows():
        if row["scenario_name"] == "conventional_baseline":
            labels.append("Conventional baseline")
        else:
            labels.append(f"{row['display_name']} ({row['city']})")

    plt.figure(figsize=(12, 6))

    plt.bar(
        labels,
        plot_df["mean_TBI"],
    )

    plt.title("Temporal Buildingness Index by Informal Transport Mode")
    plt.xlabel("Transport mode")
    plt.ylabel("Mean TBI")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    output_path = OUTPUT_FIGURES_DIR / "mode_tbi_comparison.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def plot_mode_dimension_breakdown(
    mode_summary_df: pd.DataFrame,
) -> Path:
    """
    Plot TBI dimension breakdown across transport modes.

    This uses grouped bars for the six TBI dimensions.
    """

    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = mode_summary_df.copy()
    plot_df = plot_df[plot_df["scenario_name"] != "conventional_baseline"]
    plot_df = plot_df.sort_values("mean_TBI", ascending=False)

    labels = [
        f"{row['display_name']}\n{row['city']}"
        for _, row in plot_df.iterrows()
    ]

    dimension_columns = [
        "mean_anchoring_score",
        "mean_thresholding_score",
        "mean_interiority_score",
        "mean_pausing_score",
        "mean_edge_coupling_score",
        "mean_rhythmic_repetition_score",
    ]

    dimension_labels = [
        "Anchoring",
        "Thresholding",
        "Interiority",
        "Pausing",
        "Edge Coupling",
        "Rhythmic Repetition",
    ]

    x_positions = range(len(plot_df))
    bar_width = 0.12

    plt.figure(figsize=(14, 7))

    for index, column in enumerate(dimension_columns):
        offsets = [
            x + (index - 2.5) * bar_width
            for x in x_positions
        ]

        plt.bar(
            offsets,
            plot_df[column],
            width=bar_width,
            label=dimension_labels[index],
        )

    plt.title("Temporal Buildingness Dimension Breakdown by Mode")
    plt.xlabel("Transport mode")
    plt.ylabel("Mean dimension score")
    plt.xticks(list(x_positions), labels, rotation=35, ha="right")
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_FIGURES_DIR / "mode_tbi_dimension_breakdown.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def print_mode_comparison_summary(
    mode_summary_df: pd.DataFrame,
) -> None:
    """
    Print a readable mode comparison table.
    """

    display_columns = [
        "scenario_name",
        "display_name",
        "city",
        "country",
        "movement_family",
        "agent_count",
        "mean_TBI",
        "mean_anchoring_score",
        "mean_thresholding_score",
        "mean_occupied_score_raw",
        "mean_enclosure_factor",
        "mean_interiority_score",
        "mean_pausing_score",
        "mean_edge_coupling_score",
        "mean_rhythmic_repetition_score",
        "mean_completed_cycles",
    ]

    existing_columns = [
        column for column in display_columns
        if column in mode_summary_df.columns
    ]

    print("\nMode comparison summary:")

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 300,
        "display.max_colwidth", None,
    ):
        print(
            mode_summary_df[existing_columns]
            .sort_values("mean_TBI", ascending=False)
            .to_string(index=False)
        )


def run_all_modes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run conventional baseline and all informal transport modes.
    """

    base_config = load_base_config()
    base_graph = create_street_graph(base_config)
    transport_modes = load_transport_modes()
    weights = load_tbi_weights()

    # Save profile table again as a current reference.
    modes_df = transport_modes_to_dataframe(transport_modes)
    OUTPUT_SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    modes_df.to_csv(
        OUTPUT_SUMMARIES_DIR / "transport_mode_profiles.csv",
        index=False,
    )

    all_agent_scores = []

    conventional_log_path = run_conventional_baseline(
        base_graph=base_graph,
        base_config=base_config,
    )

    conventional_scores = calculate_tbi_for_log_file(
        log_path=conventional_log_path,
        scenario_name="conventional_baseline",
        weights=weights,
    )

    all_agent_scores.append(conventional_scores)

    for mode_number, (mode_name, mode_profile) in enumerate(
        transport_modes.items(),
        start=1,
    ):
        mode_log_path = run_single_mode(
            mode_name=mode_name,
            mode_profile=mode_profile,
            base_graph=base_graph,
            base_config=base_config,
            mode_number=mode_number,
        )

        mode_scores = calculate_tbi_for_log_file(
            log_path=mode_log_path,
            scenario_name=mode_name,
            weights=weights,
        )

        all_agent_scores.append(mode_scores)

    all_agent_scores_df = pd.concat(
        all_agent_scores,
        ignore_index=True,
    )

    mode_summary_df = create_scenario_summary(all_agent_scores_df)

    agent_scores_path, mode_summary_path = save_mode_comparison_outputs(
        all_agent_scores_df,
        mode_summary_df,
    )

    tbi_figure_path = plot_mode_tbi_comparison(mode_summary_df)
    dimension_figure_path = plot_mode_dimension_breakdown(mode_summary_df)

    print_mode_comparison_summary(mode_summary_df)

    print(f"\nMode comparison agent scores saved to: {agent_scores_path}")
    print(f"Mode comparison summary saved to: {mode_summary_path}")
    print(f"Mode TBI comparison figure saved to: {tbi_figure_path}")
    print(f"Mode TBI dimension breakdown figure saved to: {dimension_figure_path}")

    return all_agent_scores_df, mode_summary_df


if __name__ == "__main__":
    run_all_modes()
