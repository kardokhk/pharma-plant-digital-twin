"""Known-answer, edge-case and regression tests for the functions behind reported numbers."""
import itertools
import json
import os

import numpy as np
import pytest

from plant_twin import dryer as d
from plant_twin import sequencing as s
from plant_twin.plant import Config, PlantSim, is_open, next_open, qc_open, simulate
from plant_twin.stats import boot_ci, wilson

HERE = os.path.dirname(__file__)


# ---- statistics --------------------------------------------------------------------------
def test_wilson_known_values():
    lo, hi = wilson(984, 1000)              # reference: Wilson score interval, z = 1.96
    assert lo == pytest.approx(97.4168, abs=1e-3) and hi == pytest.approx(99.0128, abs=1e-3)
    assert wilson(0, 1000)[0] == pytest.approx(0.0, abs=1e-12)
    assert wilson(1000, 1000)[1] == pytest.approx(100.0, abs=1e-12)


def test_boot_ci_contains_estimate_and_is_reproducible():
    x = np.arange(100.0)
    a = boot_ci(x, seed=1)
    assert a == boot_ci(x, seed=1)
    assert a[1] < a[0] < a[2] and a[0] == pytest.approx(49.5)


# ---- dryer ------------------------------------------------------------------------------
def test_lod_round_trip():
    for w in (0.01, 0.02, 0.2):
        assert d.lod(d.x_from_lod(w)) == pytest.approx(w)


def test_truth_moisture_monotone_and_bounded():
    rng = np.random.default_rng(0)
    b = d.sample_batch(rng)
    t, x, ytp, yr = d.simulate_truth(b, rng)
    assert np.all(np.diff(x) <= 1e-12) and x.min() >= d.X_E - 1e-12


def test_batch_outcome_redries_until_in_spec():
    t = np.arange(0, 200, d.DT)
    x = np.maximum(0.25 - 0.004 * t, 0.012)
    early = int(np.searchsorted(t, 30))
    out = d.batch_outcome(t, x, early)
    assert out["n_redry"] > 0 and out["lod_final"] <= d.LOD_HIGH
    assert out["cycle_min"] == pytest.approx(out["dry_min"] + d.CHECK_MIN * (1 + out["n_redry"]))


def test_stop_rules_edge_cases():
    t = np.arange(0, 10, d.DT)
    assert d.stop_fixed_time(t, 1000.0) == len(t) - 1
    assert d.stop_product_temp(np.full(len(t), 20.0), 50.0) == len(t) - 1


# ---- sequencing -------------------------------------------------------------------------
def test_milp_matches_enumeration():
    for k in range(5):
        C = s.make_instance(6, np.random.default_rng([6, k]))
        assert s.exact_milp(C)[0] == pytest.approx(s.brute_force(C)[0])


def test_qubo_energy_equals_path_cost_for_every_permutation():
    C = s.make_instance(4, np.random.default_rng(3))
    Q, const = s.to_qubo(C)
    n = 4
    for perm in itertools.permutations(range(1, n + 1)):
        x = np.zeros((n, n))
        for t, p in enumerate(perm):
            x[p - 1, t] = 1
        v = x.ravel()
        assert v @ Q @ v + const == pytest.approx(s.path_cost(C, perm))


def test_qubo_penalises_infeasible_states():
    C = s.make_instance(4, np.random.default_rng(3))
    Q, const = s.to_qubo(C)
    assert np.zeros(16) @ Q @ np.zeros(16) + const > C.sum()


# ---- plant ------------------------------------------------------------------------------
def test_calendars():
    assert is_open(0.0, 2) and not is_open(16.5, 2) and is_open(16.5, 3)
    assert not is_open(5 * 24 + 1, 3) and is_open(5 * 24 + 1, "24/7")
    assert next_open(16.5, 2) == pytest.approx(24.0)
    assert qc_open(3.0) and not qc_open(12.0)


DRYER = {"fixed": [90.0, 95.0, 100.0], "temp": [70.0, 75.0], "twin": [80.0]}


def test_simulation_is_deterministic_per_seed():
    a = simulate(Config(), 5, DRYER)
    b = simulate(Config(), 5, DRYER)
    assert a == b


def test_simulation_accounting():
    sim = PlantSim(Config(shifts=3, campaign=3), 2, DRYER)
    k = sim.run()
    assert k["released"] == sum(k[f"rel_{p}"] for p in ("P1", "P2", "P3", "P4", "P5"))
    assert 0.0 <= k["service_level"] <= 1.0
    for eq, ms in sim.machines.items():
        for m in ms:
            assert m.busy + m.cleaning + m.down <= sim.env.now + 1e-6
    assert k["margin_eur"] == pytest.approx(k["contribution_eur"] - k["cost_eur"])


def test_regression_fixture():
    """Pinned KPIs for one configuration and seed (guards against silent model changes)."""
    with open(os.path.join(HERE, "fixture_plant.json")) as fh:
        ref = json.load(fh)
    k = simulate(Config(**ref["config"]), ref["seed"], DRYER)
    for key, val in ref["kpis"].items():
        assert k[key] == pytest.approx(val, rel=1e-9)
