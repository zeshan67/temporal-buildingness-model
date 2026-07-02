"""
transport_mode_profiles.py

Loads and validates mode-specific informal transport profiles.

Each transport mode is tied to one city context for the current version
of the model.

The universal Temporal Buildingness Index remains the same, but each mode
changes the behavior that produces TBI through calibrated parameters.
"""

from pathlib import Path
import sys

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

TRANSPORT_MODES_PATH = PROJECT_ROOT / "config" / "transport_modes.yaml"
OUTPUT_SUMMARIES_DIR = PROJECT_ROOT / "outputs" / "summaries"


REQUIRED_TOP_LEVEL_FIELDS = [
    "display_name",
    "city",
    "country",
    "movement_family",
    "max_speed",
    "slow_speed",
    "capacity",
    "threshold_duration_min",
    "threshold_duration_max",
    "anchor_preference",
    "demand_sensitivity",
    "curb_sensitivity",
    "waiting_tendency",
    "stop_flexibility",
    "route_formality",
    "terminal_dependence",
    "enclosure_factor",
    "edge_coupling_tendency",
    "repair_delay_probability",
    "context",
]


REQUIRED_CONTEXT_FIELDS = [
    "pedestrian_intensity_multiplier",
    "congestion_multiplier",
    "curb_accessibility_multiplier",
    "anchor_strength_multiplier",
    "regulation_pressure",
]


VALID_MOVEMENT_FAMILIES = {
    "flexible_point_to_point",
    "route_based_collective",
    "route_flexible_collective",
}


def load_transport_modes(
    modes_path: Path = TRANSPORT_MODES_PATH,
) -> dict:
    """
    Load all transport mode profiles from config/transport_modes.yaml.
    """

    if not modes_path.exists():
        raise FileNotFoundError(f"Transport modes file not found: {modes_path}")

    with open(modes_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if "transport_modes" not in config:
        raise KeyError("transport_modes.yaml must contain a 'transport_modes' key.")

    modes = config["transport_modes"]

    validate_transport_modes(modes)

    return modes


def validate_numeric_range(
    mode_name: str,
    field_name: str,
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> None:
    """
    Validate that a numeric field is between minimum and maximum.
    """

    if not minimum <= float(value) <= maximum:
        raise ValueError(
            f"{mode_name}: '{field_name}' must be between "
            f"{minimum} and {maximum}. Current value: {value}"
        )


def validate_transport_modes(modes: dict) -> None:
    """
    Validate all transport mode profiles.
    """

    if not isinstance(modes, dict):
        raise TypeError("transport_modes must be a dictionary.")

    for mode_name, profile in modes.items():
        for field in REQUIRED_TOP_LEVEL_FIELDS:
            if field not in profile:
                raise KeyError(f"{mode_name} is missing required field: {field}")

        if profile["movement_family"] not in VALID_MOVEMENT_FAMILIES:
            raise ValueError(
                f"{mode_name}: movement_family must be one of "
                f"{sorted(VALID_MOVEMENT_FAMILIES)}"
            )

        if int(profile["capacity"]) < 1:
            raise ValueError(f"{mode_name}: capacity must be at least 1.")

        if float(profile["slow_speed"]) > float(profile["max_speed"]):
            raise ValueError(
                f"{mode_name}: slow_speed cannot be greater than max_speed."
            )

        if int(profile["threshold_duration_min"]) > int(
            profile["threshold_duration_max"]
        ):
            raise ValueError(
                f"{mode_name}: threshold_duration_min cannot be greater "
                "than threshold_duration_max."
            )

        range_fields = [
            "anchor_preference",
            "demand_sensitivity",
            "curb_sensitivity",
            "waiting_tendency",
            "stop_flexibility",
            "route_formality",
            "terminal_dependence",
            "enclosure_factor",
            "edge_coupling_tendency",
            "repair_delay_probability",
        ]

        for field in range_fields:
            validate_numeric_range(mode_name, field, profile[field])

        context = profile["context"]

        for field in REQUIRED_CONTEXT_FIELDS:
            if field not in context:
                raise KeyError(
                    f"{mode_name} context is missing required field: {field}"
                )

        validate_numeric_range(
            mode_name,
            "regulation_pressure",
            context["regulation_pressure"],
        )


def get_transport_mode_profile(
    mode_name: str,
    modes: dict | None = None,
) -> dict:
    """
    Return one transport mode profile by name.
    """

    if modes is None:
        modes = load_transport_modes()

    if mode_name not in modes:
        available_modes = ", ".join(modes.keys())
        raise KeyError(
            f"Transport mode not found: {mode_name}. "
            f"Available modes: {available_modes}"
        )

    return modes[mode_name]


def get_transport_mode_names(
    modes: dict | None = None,
) -> list[str]:
    """
    Return all available transport mode names.
    """

    if modes is None:
        modes = load_transport_modes()

    return list(modes.keys())


def transport_modes_to_dataframe(
    modes: dict | None = None,
) -> pd.DataFrame:
    """
    Convert transport mode profiles to a flat dataframe.
    """

    if modes is None:
        modes = load_transport_modes()

    rows = []

    for mode_name, profile in modes.items():
        context = profile["context"]

        row = {
            "mode_name": mode_name,
            "display_name": profile["display_name"],
            "city": profile["city"],
            "country": profile["country"],
            "movement_family": profile["movement_family"],
            "max_speed": profile["max_speed"],
            "slow_speed": profile["slow_speed"],
            "capacity": profile["capacity"],
            "threshold_duration_min": profile["threshold_duration_min"],
            "threshold_duration_max": profile["threshold_duration_max"],
            "anchor_preference": profile["anchor_preference"],
            "demand_sensitivity": profile["demand_sensitivity"],
            "curb_sensitivity": profile["curb_sensitivity"],
            "waiting_tendency": profile["waiting_tendency"],
            "stop_flexibility": profile["stop_flexibility"],
            "route_formality": profile["route_formality"],
            "terminal_dependence": profile["terminal_dependence"],
            "enclosure_factor": profile["enclosure_factor"],
            "edge_coupling_tendency": profile["edge_coupling_tendency"],
            "repair_delay_probability": profile["repair_delay_probability"],
            "pedestrian_intensity_multiplier": context[
                "pedestrian_intensity_multiplier"
            ],
            "congestion_multiplier": context["congestion_multiplier"],
            "curb_accessibility_multiplier": context[
                "curb_accessibility_multiplier"
            ],
            "anchor_strength_multiplier": context[
                "anchor_strength_multiplier"
            ],
            "regulation_pressure": context["regulation_pressure"],
        }

        rows.append(row)

    return pd.DataFrame(rows)


def save_transport_mode_summary(
    modes_df: pd.DataFrame,
) -> Path:
    """
    Save a CSV summary of all transport mode profiles.
    """

    OUTPUT_SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_SUMMARIES_DIR / "transport_mode_profiles.csv"

    modes_df.to_csv(output_path, index=False)

    return output_path


def print_transport_mode_summary(modes_df: pd.DataFrame) -> None:
    """
    Print a readable summary of transport modes.
    """

    display_columns = [
        "mode_name",
        "city",
        "country",
        "movement_family",
        "capacity",
        "max_speed",
        "enclosure_factor",
        "anchor_preference",
        "route_formality",
        "terminal_dependence",
    ]

    print("Transport mode profiles loaded successfully.")
    print(f"Number of modes: {len(modes_df)}")

    print("\nMode summary:")

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.max_colwidth", None,
    ):
        print(modes_df[display_columns].to_string(index=False))


if __name__ == "__main__":
    transport_modes = load_transport_modes()
    transport_modes_df = transport_modes_to_dataframe(transport_modes)

    print_transport_mode_summary(transport_modes_df)

    csv_path = save_transport_mode_summary(transport_modes_df)

    print(f"\nTransport mode profile summary saved to: {csv_path}")
