"""Step 2 invariance tests: common random numbers and context separation."""

import copy

import pytest
import yaml

from src.environment.shared_environment import agent_seed, build_environment
from src.simulation import controlled_runner as CR


@pytest.fixture(scope="module")
def inputs():
    return CR.load_inputs()


def test_environment_is_deterministic(inputs):
    base_config, _, _ = inputs
    a = build_environment(base_config, 42)
    b = build_environment(base_config, 42)
    assert a.fingerprint == b.fingerprint


def test_environment_changes_with_seed(inputs):
    base_config, _, _ = inputs
    assert build_environment(base_config, 42).fingerprint != build_environment(base_config, 43).fingerprint


def test_environment_independent_of_mode_profiles(inputs, monkeypatch):
    """Editing a mode's city context must not change the environment."""
    base_config, modes, weights = inputs
    reference = build_environment(base_config, 42).fingerprint
    altered = copy.deepcopy(modes)
    for profile in altered.values():
        for k in profile.get("context", {}):
            profile["context"][k] = 9.9
    monkeypatch.setattr(CR, "load_inputs", lambda: (base_config, altered, weights))
    assert build_environment(base_config, 42).fingerprint == reference


def test_all_conditions_share_one_environment():
    df = CR.run_seed(42)
    assert df["environment_fingerprint"].nunique() == 1
    assert df["scenario_name"].nunique() == 11


def test_agent_seeds_differ_by_condition_and_are_order_free(inputs):
    _, modes, _ = inputs
    seeds = {name: agent_seed(42, name) for name in list(modes) + [CR.BASELINE]}
    assert len(set(seeds.values())) == len(seeds)
    assert agent_seed(42, "Jeepney_Manila") == seeds["Jeepney_Manila"]


def test_context_changes_environment(inputs):
    base_config, _, _ = inputs
    medium = build_environment(base_config, 42, "medium").fingerprint
    high = build_environment(base_config, 42, "high").fingerprint
    assert medium != high


def test_medium_context_is_neutral(inputs):
    base_config, _, _ = inputs
    with open(CR.CONFIG_DIR / "contexts.yaml") as f:
        medium = yaml.safe_load(f)["contexts"]["medium"]
    assert all(v == 1.0 for v in medium.values())
