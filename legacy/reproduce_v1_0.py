"""
Reproduce the v1.0 manuscript results (30 seeds) with the v1.0 model code.

Seeding scheme (reconstructed from mode_comparison_runner.run_all_modes):
- base seed s in SEEDS sets config["simulation"]["random_seed"]
- street graph and agent RNGs use s
- each mode's demand zones and anchors use s + mode_number
  (mode_number = 1..10 in YAML order; FormalBus_Boundary = 10)

Usage:
    python legacy/reproduce_v1_0.py            # 30 seeds, all 11 conditions
    python legacy/reproduce_v1_0.py --seeds 5  # quick check

Writes outputs/v1.0_reproduction/agent_scores.csv and summary.csv and
prints a comparison with the values reported in the submitted manuscript.
"""

import argparse
import contextlib
import copy
import io
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.simulation.mode_comparison_runner as R  # noqa: E402

OUT_DIR = ROOT / "outputs" / "v1.0_reproduction"

# Mean TBI reported in the submitted manuscript (Section 6, Figure 5).
MANUSCRIPT_TBI = {
    "conventional_baseline": 0.022,
    "Jeepney_Manila": 0.410,
    "Matatu_Nairobi": 0.403,
    "TroTro_Accra": 0.397,
    "Colectivo_Pesero_MexicoCity": 0.392,
    "Rickshaw_Dhaka": 0.349,
    "BodaBoda_Kampala": 0.266,
    "Jitney_USA": 0.259,
    "FormalBus_Boundary": 0.278,
}

INDICATORS = [
    "anchoring_score", "thresholding_score", "interiority_score",
    "pausing_score", "edge_coupling_score", "rhythmic_repetition_score", "TBI",
]


def score_logs(logs_df: pd.DataFrame, scenario: str, weights: dict) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "events.csv"
        logs_df.to_csv(path, index=False)
        return R.calculate_tbi_for_log_file(log_path=path, scenario_name=scenario, weights=weights)


def run_seed(seed: int) -> pd.DataFrame:
    base_config = R.load_base_config()
    base_config["simulation"]["random_seed"] = seed
    weights = R.load_tbi_weights()
    modes = R.load_transport_modes()
    graph = R.create_street_graph(base_config)
    frames = []

    with contextlib.redirect_stdout(io.StringIO()):
        conv = R.simulate_conventional_vehicles(graph=graph, config=base_config)
        conv["scenario_name"] = "conventional_baseline"
        frames.append(score_logs(conv, "conventional_baseline", weights))

        for mode_number, (name, profile) in enumerate(modes.items(), start=1):
            mode_seed = seed + mode_number
            mode_graph = R.create_mode_context_graph(graph, profile)
            zones = R.create_mode_demand_zones(graph=mode_graph, mode_profile=profile,
                                               random_seed=mode_seed, number_of_zones=5)
            anchors = R.create_mode_anchor_points(graph=mode_graph, mode_profile=profile,
                                                  random_seed=mode_seed, number_of_anchors=5)
            logs = R.simulate_informal_transport(graph=mode_graph, demand_zones=zones,
                                                 anchor_points=anchors, config=copy.deepcopy(base_config),
                                                 mode_name=name, mode_profile=profile)
            logs["scenario_name"] = name
            frames.append(score_logs(logs, name, weights))

    df = pd.concat(frames, ignore_index=True)
    df["seed"] = seed
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--first-seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))

    with Pool(args.workers) as pool:
        agent_scores = pd.concat(pool.map(run_seed, seeds), ignore_index=True)

    seed_means = agent_scores.groupby(["scenario_name", "seed"])[INDICATORS].mean().reset_index()
    summary = seed_means.groupby("scenario_name")[INDICATORS].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary["manuscript_TBI"] = summary["scenario_name"].map(MANUSCRIPT_TBI)
    summary["difference"] = summary["TBI_mean"] - summary["manuscript_TBI"]
    summary["within_0.005"] = summary["difference"].abs() <= 0.005
    summary["seeds"] = f"{seeds[0]}-{seeds[-1]}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agent_scores.to_csv(OUT_DIR / "agent_scores.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    cols = ["scenario_name", "TBI_mean", "TBI_std", "manuscript_TBI", "difference", "within_0.005"]
    print(summary.sort_values("TBI_mean", ascending=False)[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
