"""
main.py

FastAPI web application for the Temporal Buildingness Model.

This app serves a custom HTML/CSS/JavaScript interface and exposes API
endpoints that replay generated simulation logs.

Run from the project root with:

python -m uvicorn app.main:app --reload

Then open:

http://127.0.0.1:8000
"""

from functools import lru_cache
from pathlib import Path
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.environment.street_graph import load_base_config, create_street_graph


APP_DIR = PROJECT_ROOT / "app"
STATIC_DIR = APP_DIR / "static"

OUTPUT_LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
OUTPUT_SUMMARIES_DIR = PROJECT_ROOT / "outputs" / "summaries"


MODE_LOG_FILES = {
    "conventional_baseline": {
        "label": "Conventional baseline",
        "filename": "mode_comparison_conventional_baseline_events.csv",
    },
    "Rickshaw_Dhaka": {
        "label": "Rickshaw — Dhaka",
        "filename": "mode_Rickshaw_Dhaka_events.csv",
    },
    "Matatu_Nairobi": {
        "label": "Matatu — Nairobi",
        "filename": "mode_Matatu_Nairobi_events.csv",
    },
    "Jeepney_Manila": {
        "label": "Jeepney — Manila",
        "filename": "mode_Jeepney_Manila_events.csv",
    },
    "Colectivo_Pesero_MexicoCity": {
        "label": "Colectivo / Pesero — Mexico City",
        "filename": "mode_Colectivo_Pesero_MexicoCity_events.csv",
    },
    "TroTro_Accra": {
        "label": "Tro-tro — Accra",
        "filename": "mode_TroTro_Accra_events.csv",
    },
    "TukTuk_Bangkok": {
        "label": "Tuk-tuk — Bangkok",
        "filename": "mode_TukTuk_Bangkok_events.csv",
    },
    "AutoRickshaw_NewDelhi": {
        "label": "Auto-rickshaw — New Delhi",
        "filename": "mode_AutoRickshaw_NewDelhi_events.csv",
    },
    "BodaBoda_Kampala": {
        "label": "Boda-boda — Kampala",
        "filename": "mode_BodaBoda_Kampala_events.csv",
    },
    "Jitney_USA": {
        "label": "Jitney — USA",
        "filename": "mode_Jitney_USA_events.csv",
    },
}


STATE_COLORS = {
    "IDLE_AT_ANCHOR": "#2F80ED",
    "SEARCHING_FOR_DEMAND": "#F2994A",
    "APPROACHING_PASSENGER": "#27AE60",
    "BOARDING_THRESHOLD": "#EB5757",
    "MOVING_WITH_PASSENGER": "#9B51E0",
    "TEMPORARY_PAUSE": "#8D6E63",
    "ALIGHTING_THRESHOLD": "#F06292",
    "REPOSITIONING": "#828282",
    "REPAIR_OR_DELAY": "#BDB32A",
    "MOVING": "#00A8CC",
    "ARRIVED": "#4E6CEF",
    "TEMPORARY_STOP": "#FF6B6B",
}


app = FastAPI(
    title="Temporal Buildingness Model API",
    description="API and custom frontend for replaying agent-based transport simulations.",
    version="0.1.0",
)


app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


def to_bool(series: pd.Series) -> pd.Series:
    """
    Convert a dataframe column to boolean.
    """

    if series.dtype == bool:
        return series

    return series.astype(str).str.lower().isin(["true", "1", "yes"])


@lru_cache(maxsize=1)
def load_graph_data() -> dict:
    """
    Recreate the synthetic graph and return positions plus edges.
    """

    base_config = load_base_config()
    graph = create_street_graph(base_config)

    positions = {}

    for node_id, node_data in graph.nodes(data=True):
        positions[node_id] = {
            "x": float(node_data["x"]),
            "y": float(node_data["y"]),
        }

    edges = []

    for node_a, node_b in graph.edges():
        edges.append(
            {
                "node_a": node_a,
                "node_b": node_b,
                "x0": positions[node_a]["x"],
                "y0": positions[node_a]["y"],
                "x1": positions[node_b]["x"],
                "y1": positions[node_b]["y"],
            }
        )

    return {
        "positions": positions,
        "edges": edges,
    }


@lru_cache(maxsize=20)
def load_mode_log(mode_key: str) -> pd.DataFrame:
    """
    Load and prepare one mode event log.
    """

    if mode_key not in MODE_LOG_FILES:
        raise KeyError(f"Unknown mode key: {mode_key}")

    log_filename = MODE_LOG_FILES[mode_key]["filename"]
    log_path = OUTPUT_LOGS_DIR / log_filename

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    logs_df = pd.read_csv(log_path)

    graph_data = load_graph_data()
    positions = graph_data["positions"]

    logs_df["x"] = logs_df["current_node"].map(
        lambda node_id: positions.get(str(node_id), {}).get("x")
    )

    logs_df["y"] = logs_df["current_node"].map(
        lambda node_id: positions.get(str(node_id), {}).get("y")
    )

    logs_df = logs_df.dropna(subset=["x", "y"])

    boolean_columns = [
        "near_anchor",
        "near_active_edge",
        "threshold_event",
        "pause_event",
        "repair_delay_event",
        "cycle_completed",
    ]

    for column in boolean_columns:
        if column not in logs_df.columns:
            logs_df[column] = False

        logs_df[column] = to_bool(logs_df[column])

    if "occupancy" not in logs_df.columns:
        logs_df["occupancy"] = 0

    if "capacity" not in logs_df.columns:
        logs_df["capacity"] = 1

    if "speed" not in logs_df.columns:
        logs_df["speed"] = 0.0

    if "enclosure_factor" not in logs_df.columns:
        logs_df["enclosure_factor"] = 0.0

    for column in [
        "mode_name",
        "display_name",
        "city",
        "country",
        "movement_family",
    ]:
        if column not in logs_df.columns:
            logs_df[column] = ""

    return logs_df


@lru_cache(maxsize=1)
def load_mode_summary() -> pd.DataFrame:
    """
    Load final mode comparison summary.
    """

    summary_path = OUTPUT_SUMMARIES_DIR / "mode_comparison_summary.csv"

    if not summary_path.exists():
        return pd.DataFrame()

    return pd.read_csv(summary_path)


def calculate_live_scores(logs_until_step_df: pd.DataFrame) -> dict:
    """
    Calculate live TBI-like scores up to a selected timestep.

    This mirrors the official TBI logic closely enough for real-time display.
    The final official score remains the CSV output from the metric module.
    """

    if logs_until_step_df.empty:
        return {
            "anchoring": 0.0,
            "thresholding": 0.0,
            "interiority": 0.0,
            "pausing": 0.0,
            "edge_coupling": 0.0,
            "rhythmic_repetition": 0.0,
            "live_tbi": 0.0,
        }

    total_events = len(logs_until_step_df)

    anchoring = (
        logs_until_step_df["state"].astype(str).eq("IDLE_AT_ANCHOR").sum()
        / total_events
    )

    thresholding = logs_until_step_df["threshold_event"].sum() / total_events

    occupied_score_raw = (
        logs_until_step_df["occupancy"].astype(float).gt(0).sum()
        / total_events
    )

    enclosure_factor = float(
        logs_until_step_df["enclosure_factor"].fillna(0.0).mean()
    )

    interiority = occupied_score_raw * enclosure_factor

    pausing = logs_until_step_df["pause_event"].sum() / total_events

    edge_coupling = logs_until_step_df["near_active_edge"].sum() / total_events

    has_anchor_behavior = logs_until_step_df["near_anchor"].sum() > 0

    if has_anchor_behavior:
        completed_cycles = logs_until_step_df["cycle_completed"].sum()
    else:
        completed_cycles = 0

    unique_steps = max(1, logs_until_step_df["step"].nunique())
    maximum_possible_cycles = max(1, unique_steps / 20)

    rhythmic_repetition = min(
        1.0,
        completed_cycles / maximum_possible_cycles,
    )

    live_tbi = (
        0.20 * anchoring
        + 0.20 * thresholding
        + 0.20 * interiority
        + 0.10 * pausing
        + 0.15 * edge_coupling
        + 0.15 * rhythmic_repetition
    )

    return {
        "anchoring": round(float(anchoring), 4),
        "thresholding": round(float(thresholding), 4),
        "interiority": round(float(interiority), 4),
        "pausing": round(float(pausing), 4),
        "edge_coupling": round(float(edge_coupling), 4),
        "rhythmic_repetition": round(float(rhythmic_repetition), 4),
        "live_tbi": round(float(live_tbi), 4),
    }


@app.get("/")
def index() -> FileResponse:
    """
    Serve the main frontend page.
    """

    index_path = STATIC_DIR / "index.html"

    return FileResponse(index_path)


@app.get("/api/modes")
def get_modes() -> dict:
    """
    Return available transport modes and whether their log files exist.
    """

    modes = []

    for mode_key, mode_info in MODE_LOG_FILES.items():
        log_path = OUTPUT_LOGS_DIR / mode_info["filename"]

        modes.append(
            {
                "mode_key": mode_key,
                "label": mode_info["label"],
                "filename": mode_info["filename"],
                "exists": log_path.exists(),
            }
        )

    return {
        "modes": modes,
        "state_colors": STATE_COLORS,
    }


@app.get("/api/graph")
def get_graph() -> dict:
    """
    Return graph edges for frontend canvas drawing.
    """

    graph_data = load_graph_data()

    return {
        "edges": graph_data["edges"],
    }


@app.get("/api/summary")
def get_summary() -> dict:
    """
    Return final mode comparison summary.
    """

    summary_df = load_mode_summary()

    if summary_df.empty:
        return {
            "rows": [],
        }

    return {
        "rows": summary_df.fillna("").to_dict(orient="records"),
    }


@app.get("/api/simulation/{mode_key}/metadata")
def get_simulation_metadata(mode_key: str) -> dict:
    """
    Return metadata for one mode simulation.
    """

    try:
        logs_df = load_mode_log(mode_key)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    first_row = logs_df.iloc[0]

    metadata = {
        "mode_key": mode_key,
        "label": MODE_LOG_FILES[mode_key]["label"],
        "min_step": int(logs_df["step"].min()),
        "max_step": int(logs_df["step"].max()),
        "agent_count": int(logs_df["agent_id"].nunique()),
        "row_count": int(len(logs_df)),
        "display_name": str(first_row.get("display_name", "")),
        "city": str(first_row.get("city", "")),
        "country": str(first_row.get("country", "")),
        "movement_family": str(first_row.get("movement_family", "")),
        "enclosure_factor": float(first_row.get("enclosure_factor", 0.0)),
    }

    return metadata


@app.get("/api/simulation/{mode_key}/frame")
def get_simulation_frame(
    mode_key: str,
    step: int = Query(..., ge=0),
    agent_limit: int = Query(20, ge=1, le=200),
) -> dict:
    """
    Return agent positions and state counts for one timestep.
    """

    try:
        logs_df = load_mode_log(mode_key)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    current_step_df = logs_df[logs_df["step"] == step].copy()

    current_step_df = current_step_df.sort_values("agent_id").head(agent_limit)

    agents = []

    for _, row in current_step_df.iterrows():
        agents.append(
            {
                "agent_id": str(row["agent_id"]),
                "state": str(row["state"]),
                "current_node": str(row["current_node"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "occupancy": float(row.get("occupancy", 0)),
                "capacity": float(row.get("capacity", 1)),
                "speed": float(row.get("speed", 0)),
                "near_anchor": bool(row.get("near_anchor", False)),
                "near_active_edge": bool(row.get("near_active_edge", False)),
            }
        )

    state_counts = (
        current_step_df["state"]
        .value_counts()
        .reset_index()
    )

    state_counts.columns = ["state", "count"]

    return {
        "mode_key": mode_key,
        "step": step,
        "agents": agents,
        "state_counts": state_counts.to_dict(orient="records"),
    }


@app.get("/api/simulation/{mode_key}/trails")
def get_simulation_trails(
    mode_key: str,
    step: int = Query(..., ge=0),
    agent_limit: int = Query(20, ge=1, le=200),
    sample_every: int = Query(1, ge=1, le=25),
) -> dict:
    """
    Return path trails and event markers up to a selected timestep.

    This powers the exportable simulation image in the browser.
    """

    try:
        logs_df = load_mode_log(mode_key)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    current_step_df = logs_df[logs_df["step"] == step].copy()
    selected_agent_ids = (
        current_step_df.sort_values("agent_id")["agent_id"]
        .head(agent_limit)
        .astype(str)
        .tolist()
    )

    if not selected_agent_ids:
        return {
            "mode_key": mode_key,
            "step": step,
            "trails": [],
            "event_markers": [],
        }

    logs_until_step_df = logs_df[
        (logs_df["step"] <= step)
        & (logs_df["agent_id"].astype(str).isin(selected_agent_ids))
    ].copy()

    trails = []
    event_markers = []

    event_states = {
        "IDLE_AT_ANCHOR",
        "BOARDING_THRESHOLD",
        "ALIGHTING_THRESHOLD",
        "TEMPORARY_PAUSE",
        "REPAIR_OR_DELAY",
        "TEMPORARY_STOP",
        "ARRIVED",
    }

    for agent_id, agent_logs in logs_until_step_df.groupby("agent_id"):
        agent_logs = agent_logs.sort_values("step")

        sampled_logs = agent_logs.iloc[::sample_every].copy()

        points = []

        for _, row in sampled_logs.iterrows():
            points.append(
                {
                    "step": int(row["step"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "state": str(row["state"]),
                    "occupancy": float(row.get("occupancy", 0)),
                }
            )

        trails.append(
            {
                "agent_id": str(agent_id),
                "points": points,
            }
        )

        marker_logs = agent_logs[
            (
                agent_logs["state"].astype(str).isin(event_states)
                | agent_logs["pause_event"]
                | agent_logs["repair_delay_event"]
                | agent_logs["threshold_event"]
                | agent_logs["cycle_completed"]
            )
        ].copy()

        previous_marker_key = None

        for _, row in marker_logs.iterrows():
            marker_key = (
                str(row["agent_id"]),
                str(row["current_node"]),
                str(row["state"]),
                bool(row.get("threshold_event", False)),
                bool(row.get("pause_event", False)),
                bool(row.get("repair_delay_event", False)),
            )

            if marker_key == previous_marker_key:
                continue

            previous_marker_key = marker_key

            event_type = str(row["state"])

            if bool(row.get("repair_delay_event", False)):
                event_type = "REPAIR_OR_DELAY"
            elif bool(row.get("pause_event", False)):
                event_type = "TEMPORARY_PAUSE"
            elif bool(row.get("threshold_event", False)):
                event_type = str(row["state"])

            event_markers.append(
                {
                    "agent_id": str(row["agent_id"]),
                    "step": int(row["step"]),
                    "event_type": event_type,
                    "state": str(row["state"]),
                    "current_node": str(row["current_node"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "occupancy": float(row.get("occupancy", 0)),
                    "threshold_event": bool(row.get("threshold_event", False)),
                    "pause_event": bool(row.get("pause_event", False)),
                    "repair_delay_event": bool(row.get("repair_delay_event", False)),
                    "cycle_completed": bool(row.get("cycle_completed", False)),
                }
            )

    return {
        "mode_key": mode_key,
        "step": step,
        "trails": trails,
        "event_markers": event_markers,
    }


@app.get("/api/simulation/{mode_key}/live")
def get_live_scores(
    mode_key: str,
    step: int = Query(..., ge=0),
) -> dict:
    """
    Return cumulative live TBI-like scores up to a selected timestep.
    """

    try:
        logs_df = load_mode_log(mode_key)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    logs_until_step_df = logs_df[logs_df["step"] <= step].copy()

    live_scores = calculate_live_scores(logs_until_step_df)

    cumulative_state_counts = (
        logs_until_step_df["state"]
        .value_counts()
        .reset_index()
    )

    cumulative_state_counts.columns = ["state", "count"]

    return {
        "mode_key": mode_key,
        "step": step,
        "live_scores": live_scores,
        "cumulative_state_counts": cumulative_state_counts.to_dict(orient="records"),
    }


@app.get("/api/health")
def health_check() -> dict:
    """
    Simple health check.
    """

    return {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
    }
