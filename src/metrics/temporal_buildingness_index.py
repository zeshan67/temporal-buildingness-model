"""
temporal_buildingness_index.py

Calculates the Temporal Buildingness Index (TBI).

The TBI measures how strongly an agent performs building-like behavior over time.

TBI dimensions:
- Anchoring
- Thresholding
- Interiority
- Pausing
- Edge Coupling
- Rhythmic Repetition

This script reads event logs from:
- conventional vehicle simulation
- informal transport simulation

Then it calculates agent-level and scenario-level TBI scores.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

OUTPUT_LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
OUTPUT_SUMMARIES_DIR = PROJECT_ROOT / "outputs" / "summaries"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

WEIGHTS_PATH = PROJECT_ROOT / "config" / "weights.yaml"

CONVENTIONAL_LOG_PATH = OUTPUT_LOGS_DIR / "conventional_vehicle_test_events.csv"
INFORMAL_LOG_PATH = OUTPUT_LOGS_DIR / "informal_transport_test_events.csv"


def load_tbi_weights(weights_path: Path = WEIGHTS_PATH) -> dict:
    """
    Load Temporal Buildingness Index weights from config/weights.yaml.
    """

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    with open(weights_path, "r", encoding="utf-8") as file:
        weights_config = yaml.safe_load(file)

    return weights_config["temporal_buildingness_weights"]


def convert_to_bool(series: pd.Series) -> pd.Series:
    """
    Safely convert a pandas Series into boolean values.

    This handles both true booleans and text values like 'True'/'False'.
    """

    if series.dtype == bool:
        return series

    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def prepare_log_columns(logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required columns exist and are correctly typed.
    """

    logs_df = logs_df.copy()

    required_boolean_columns = [
        "near_anchor",
        "near_active_edge",
        "threshold_event",
        "pause_event",
        "cycle_completed",
    ]

    for column in required_boolean_columns:
        if column not in logs_df.columns:
            logs_df[column] = False

        logs_df[column] = convert_to_bool(logs_df[column])

    if "occupancy" not in logs_df.columns:
        logs_df["occupancy"] = 0

    if "state" not in logs_df.columns:
        logs_df["state"] = "UNKNOWN"

    return logs_df


def calculate_agent_tbi(
    agent_logs: pd.DataFrame,
    weights: dict,
    minimum_cycle_steps: int = 20,
) -> dict:
    """
    Calculate TBI score for a single agent.

    Args:
        agent_logs: event logs for one agent
        weights: TBI dimension weights
        minimum_cycle_steps: theoretical minimum number of timesteps required
            to complete one buildingness cycle

    Returns:
        Dictionary containing dimension scores and total TBI
    """

    total_timesteps = len(agent_logs)

    if total_timesteps == 0:
        raise ValueError("Agent log is empty.")

    agent_id = agent_logs["agent_id"].iloc[0]
    agent_type = agent_logs["agent_type"].iloc[0]

    # 1. Anchoring
    # Strong anchoring means the agent spends time in IDLE_AT_ANCHOR.
    anchoring_steps = (
        agent_logs["state"].astype(str).eq("IDLE_AT_ANCHOR")
    ).sum()
    anchoring_score = anchoring_steps / total_timesteps

    # 2. Thresholding
    # Boarding and alighting events create temporary thresholds.
    thresholding_score = agent_logs["threshold_event"].sum() / total_timesteps

    # 3. Interiority
    # Occupied movement represents temporary interior formation.
    interiority_score = (
        agent_logs["occupancy"].astype(float).gt(0).sum()
        / total_timesteps
    )

    # 4. Pausing
    # Temporary stops produce micro-occupation in the street.
    pausing_score = agent_logs["pause_event"].sum() / total_timesteps

    # 5. Edge Coupling
    # Time near active street edges measures relation to urban edge activity.
    edge_coupling_score = agent_logs["near_active_edge"].sum() / total_timesteps

    # 6. Rhythmic Repetition
    # A cycle is anchor/threshold/movement/threshold/repositioning behavior.
    completed_cycles = agent_logs["cycle_completed"].sum()
    maximum_possible_cycles = max(1, total_timesteps / minimum_cycle_steps)
    rhythmic_repetition_score = completed_cycles / maximum_possible_cycles

    # Keep score between 0 and 1.
    rhythmic_repetition_score = min(1.0, rhythmic_repetition_score)

    total_tbi = (
        weights["anchoring"] * anchoring_score
        + weights["thresholding"] * thresholding_score
        + weights["interiority"] * interiority_score
        + weights["pausing"] * pausing_score
        + weights["edge_coupling"] * edge_coupling_score
        + weights["rhythmic_repetition"] * rhythmic_repetition_score
    )

    return {
        "agent_id": agent_id,
        "agent_type": agent_type,
        "total_timesteps": total_timesteps,
        "anchoring_score": round(float(anchoring_score), 4),
        "thresholding_score": round(float(thresholding_score), 4),
        "interiority_score": round(float(interiority_score), 4),
        "pausing_score": round(float(pausing_score), 4),
        "edge_coupling_score": round(float(edge_coupling_score), 4),
        "rhythmic_repetition_score": round(float(rhythmic_repetition_score), 4),
        "completed_cycles": int(completed_cycles),
        "TBI": round(float(total_tbi), 4),
    }


def calculate_tbi_for_log_file(
    log_path: Path,
    scenario_name: str,
    weights: dict,
) -> pd.DataFrame:
    """
    Calculate TBI for all agents in one log file.
    """

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    logs_df = pd.read_csv(log_path)
    logs_df = prepare_log_columns(logs_df)

    agent_scores = []

    for agent_id, agent_logs in logs_df.groupby("agent_id"):
        score = calculate_agent_tbi(
            agent_logs=agent_logs,
            weights=weights,
        )
        score["scenario_name"] = scenario_name
        agent_scores.append(score)

    scores_df = pd.DataFrame(agent_scores)

    return scores_df


def create_scenario_summary(agent_scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create scenario-level summary from agent-level TBI scores.
    """

    summary_df = (
        agent_scores_df
        .groupby("scenario_name")
        .agg(
            agent_type=("agent_type", "first"),
            agent_count=("agent_id", "count"),
            mean_TBI=("TBI", "mean"),
            mean_anchoring_score=("anchoring_score", "mean"),
            mean_thresholding_score=("thresholding_score", "mean"),
            mean_interiority_score=("interiority_score", "mean"),
            mean_pausing_score=("pausing_score", "mean"),
            mean_edge_coupling_score=("edge_coupling_score", "mean"),
            mean_rhythmic_repetition_score=("rhythmic_repetition_score", "mean"),
            mean_completed_cycles=("completed_cycles", "mean"),
        )
        .reset_index()
    )

    numeric_columns = summary_df.select_dtypes(include="number").columns

    for column in numeric_columns:
        summary_df[column] = summary_df[column].round(4)

    return summary_df


def save_tbi_outputs(
    agent_scores_df: pd.DataFrame,
    scenario_summary_df: pd.DataFrame,
) -> tuple[Path, Path]:
    """
    Save TBI agent scores and scenario summary.
    """

    OUTPUT_SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    agent_scores_path = OUTPUT_SUMMARIES_DIR / "tbi_agent_scores.csv"
    scenario_summary_path = OUTPUT_SUMMARIES_DIR / "tbi_scenario_summary.csv"

    agent_scores_df.to_csv(agent_scores_path, index=False)
    scenario_summary_df.to_csv(scenario_summary_path, index=False)

    return agent_scores_path, scenario_summary_path


def plot_tbi_comparison(scenario_summary_df: pd.DataFrame) -> Path:
    """
    Create a bar chart comparing mean TBI across scenarios.
    """

    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.bar(
        scenario_summary_df["scenario_name"],
        scenario_summary_df["mean_TBI"],
    )

    plt.title("Temporal Buildingness Index Comparison")
    plt.xlabel("Scenario")
    plt.ylabel("Mean TBI")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    output_path = OUTPUT_FIGURES_DIR / "tbi_comparison.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def print_tbi_summary(scenario_summary_df: pd.DataFrame) -> None:
    """
    Print a readable summary of TBI results.
    """

    print("Temporal Buildingness Index calculation completed.")

    print("\nScenario summary:")
    print(scenario_summary_df)

    if len(scenario_summary_df) >= 2:
        conventional_row = scenario_summary_df[
            scenario_summary_df["scenario_name"] == "conventional_baseline"
        ]

        informal_row = scenario_summary_df[
            scenario_summary_df["scenario_name"] == "informal_with_anchors"
        ]

        if not conventional_row.empty and not informal_row.empty:
            conventional_tbi = float(conventional_row["mean_TBI"].iloc[0])
            informal_tbi = float(informal_row["mean_TBI"].iloc[0])

            print("\nCore comparison:")
            print(f"TBI_conventional_vehicle = {conventional_tbi}")
            print(f"TBI_informal_transport = {informal_tbi}")

            if informal_tbi > conventional_tbi:
                print(
                    "\nResult: informal transport produced a higher "
                    "Temporal Buildingness Index than the conventional vehicle baseline."
                )
            else:
                print(
                    "\nResult: informal transport did not produce a higher "
                    "Temporal Buildingness Index in this run."
                )


if __name__ == "__main__":
    weights = load_tbi_weights()

    conventional_scores = calculate_tbi_for_log_file(
        log_path=CONVENTIONAL_LOG_PATH,
        scenario_name="conventional_baseline",
        weights=weights,
    )

    informal_scores = calculate_tbi_for_log_file(
        log_path=INFORMAL_LOG_PATH,
        scenario_name="informal_with_anchors",
        weights=weights,
    )

    all_agent_scores = pd.concat(
        [conventional_scores, informal_scores],
        ignore_index=True,
    )

    scenario_summary = create_scenario_summary(all_agent_scores)

    agent_scores_path, scenario_summary_path = save_tbi_outputs(
        all_agent_scores,
        scenario_summary,
    )

    figure_path = plot_tbi_comparison(scenario_summary)

    print_tbi_summary(scenario_summary)

    print(f"\nAgent TBI scores saved to: {agent_scores_path}")
    print(f"Scenario TBI summary saved to: {scenario_summary_path}")
    print(f"TBI comparison figure saved to: {figure_path}")
