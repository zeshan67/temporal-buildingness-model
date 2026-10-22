"""
Controlled mode comparison (v1.1, fixes B2, B4, B4b).

For each base seed:
- one shared environment is built (graph, demand zones, anchors) for the chosen
  context level and used by every condition (common random numbers);
- each condition's agents get their own reproducible seed derived from
  (base seed, condition name), so agent randomness is independent across modes
  and stable when modes are added, removed or reordered;
- mode profiles never modify the environment; city "context" blocks in
  transport_modes.yaml are provenance only and are not read here.

Usage:
    python -m src.simulation.controlled_runner --seeds 30
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yaml

from src.agents.conventional_vehicle import simulate_conventional_vehicles
from src.agents.informal_transport import simulate_informal_transport
from src.environment.shared_environment import agent_seed, build_environment
from src.metrics.temporal_buildingness_index import calculate_tbi_for_logs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "controlled"

BASELINE = "conventional_baseline"

INDICATORS = [
    "anchoring_score", "thresholding_score", "interiority_score",
    "pausing_score", "edge_coupling_score", "rhythmic_repetition_score", "TBI",
]


def _load(name: str) -> dict:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_inputs() -> tuple[dict, dict, dict]:
    base_config = _load("base_config.yaml")
    modes = _load("transport_modes.yaml")["transport_modes"]
    weights = _load("weights.yaml")["temporal_buildingness_weights"]
    return base_config, modes, weights


def _agent_config(base_config: dict, seed: int, condition: str) -> dict:
    """Copy of base_config whose random_seed is the condition's agent seed."""
    cfg = copy.deepcopy(base_config)
    cfg["simulation"]["random_seed"] = agent_seed(seed, condition)
    return cfg


def run_seed(seed: int, context: str | None = None) -> pd.DataFrame:
    """Run the baseline and every mode at one base seed in one shared environment."""
    base_config, modes, weights = load_inputs()
    env = build_environment(base_config, seed, context)
    frames = []

    with contextlib.redirect_stdout(io.StringIO()):
        logs = simulate_conventional_vehicles(
            graph=copy.deepcopy(env.graph),
            config=_agent_config(base_config, seed, BASELINE),
        )
        frames.append(calculate_tbi_for_logs(logs, BASELINE, weights))

        for name, profile in modes.items():
            logs = simulate_informal_transport(
                graph=copy.deepcopy(env.graph),
                demand_zones=copy.deepcopy(env.demand_zones),
                anchor_points=copy.deepcopy(env.anchor_points),
                config=_agent_config(base_config, seed, name),
                mode_name=name,
                mode_profile=profile,
            )
            frames.append(calculate_tbi_for_logs(logs, name, weights))

    df = pd.concat(frames, ignore_index=True)
    df["seed"] = seed
    df["context"] = env.context
    df["environment_fingerprint"] = env.fingerprint
    return df


def _run_seed_star(args: tuple[int, str | None]) -> pd.DataFrame:
    return run_seed(*args)


def run_seeds(seeds: list[int], context: str | None = None, workers: int = 2) -> pd.DataFrame:
    with Pool(workers) as pool:
        frames = pool.map(_run_seed_star, [(s, context) for s in seeds])
    return pd.concat(frames, ignore_index=True)


def summarise(agent_scores: pd.DataFrame) -> pd.DataFrame:
    """Mean of per-seed condition means, with SD across seeds."""
    seed_means = agent_scores.groupby(["scenario_name", "seed"])[INDICATORS].mean().reset_index()
    summary = seed_means.groupby("scenario_name")[INDICATORS].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    return summary.reset_index().sort_values("TBI_mean", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--first-seed", type=int, default=42)
    ap.add_argument("--context", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(OUTPUT_DIR))
    args = ap.parse_args()

    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    agent_scores = run_seeds(seeds, args.context, args.workers)
    summary = summarise(agent_scores)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    agent_scores.to_csv(out / "agent_scores.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary[["scenario_name", "TBI_mean", "TBI_std"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
