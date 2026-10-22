"""
Shared environment for controlled comparisons (v1.1, fixes B2 and B4).

One environment (street graph, demand zones, anchor points) is built per seed
and context level, and every transport mode runs in that same environment.
Mode differences therefore cannot come from differences in the environment.

Seed streams are derived from one base seed with numpy SeedSequence, so the
graph, demand zones, anchors and each mode's agents draw from independent,
reproducible random streams.
"""

from __future__ import annotations

import copy
import hashlib
import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import yaml

from src.environment.anchor_points import create_anchor_points
from src.environment.demand_field import create_demand_zones
from src.environment.street_graph import create_street_graph

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTEXTS_PATH = PROJECT_ROOT / "config" / "contexts.yaml"

# Fixed stream identifiers. Changing these changes every result.
STREAM_GRAPH = 0
STREAM_DEMAND = 1
STREAM_ANCHORS = 2
STREAM_AGENTS = 3

NUMBER_OF_DEMAND_ZONES = 5
NUMBER_OF_ANCHORS = 5


def derive_seed(base_seed: int, *keys: int) -> int:
    """Return a reproducible 32-bit seed for a named sub-stream of base_seed."""
    return int(np.random.SeedSequence([int(base_seed), *[int(k) for k in keys]]).generate_state(1)[0])


def stable_id(name: str) -> int:
    """Stable integer for a string (independent of Python's hash randomisation)."""
    return zlib.crc32(name.encode("utf-8"))


def agent_seed(base_seed: int, condition_name: str) -> int:
    """Seed for the agents of one condition (mode or baseline) at one base seed."""
    return derive_seed(base_seed, STREAM_AGENTS, stable_id(condition_name))


def load_contexts(path: Path = CONTEXTS_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cap(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def apply_context(graph: nx.Graph, multipliers: dict) -> nx.Graph:
    """Return a copy of graph with edge attributes scaled by context multipliers."""
    out = copy.deepcopy(graph)
    for _, _, data in out.edges(data=True):
        for attribute, multiplier in multipliers.items():
            data[attribute] = round(_cap(data[attribute] * float(multiplier)), 3)
    return out


@dataclass
class SharedEnvironment:
    seed: int
    context: str
    graph: nx.Graph
    demand_zones: list[dict]
    anchor_points: list[dict]
    fingerprint: str = field(default="")


def environment_fingerprint(graph: nx.Graph, demand_zones: list[dict], anchor_points: list[dict]) -> str:
    """Hash of everything an agent can perceive in the environment."""
    payload = {
        "edges": sorted(
            [sorted([str(u), str(v)]) + [sorted((k, str(x)) for k, x in d.items())] for u, v, d in graph.edges(data=True)]
        ),
        "demand_zones": demand_zones,
        "anchor_points": anchor_points,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def build_environment(base_config: dict, seed: int, context: str | None = None) -> SharedEnvironment:
    """Build the single environment used by every condition at this seed and context."""
    contexts = load_contexts()
    context = context or contexts["default_context"]
    multipliers = contexts["contexts"][context]

    graph_config = copy.deepcopy(base_config)
    graph_config["simulation"]["random_seed"] = derive_seed(seed, STREAM_GRAPH)
    graph = apply_context(create_street_graph(graph_config), multipliers)

    demand_zones = create_demand_zones(
        graph,
        number_of_zones=NUMBER_OF_DEMAND_ZONES,
        random_seed=derive_seed(seed, STREAM_DEMAND),
    )
    anchor_points = create_anchor_points(
        graph,
        number_of_anchors=NUMBER_OF_ANCHORS,
        random_seed=derive_seed(seed, STREAM_ANCHORS),
    )

    env = SharedEnvironment(seed=seed, context=context, graph=graph,
                            demand_zones=demand_zones, anchor_points=anchor_points)
    env.fingerprint = environment_fingerprint(graph, demand_zones, anchor_points)
    return env
