"""
Regression Test Suite for Hebbian Sandbox
DataForge 2026 — Synaptic Plasticity as Short-Term Memory

Protects the frozen mathematical model (hebbian_sparse.py) and API contracts (app.py).
Expected values are grounded in the verified frozen implementation.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from hebbian_sparse import SparseHebbianMemory, run_sparse_experiment
from app import app

client = TestClient(app)


# ==============================================================================
# 1. FROZEN MODEL DETERMINISM & HEADLINE MATHEMATICS
# ==============================================================================

def test_deterministic_seed_single_trial():
    """Verify single trial with fixed seed produces deterministic, perfect recall at delay=0."""
    mem = SparseHebbianMemory(200, sparsity=0.05, seed=42)
    pattern = mem.random_pattern()
    mem.encode(pattern)
    active_indices = np.where(pattern == 1)[0]
    cue, known_mask = mem.make_cue(active_indices, 0.4)
    recalled = mem.recall(cue, known_mask)
    accuracy = mem.overlap_accuracy(recalled, pattern, mem.k)

    assert accuracy == 1.0
    assert np.sum(recalled) == mem.k
    assert mem.k == 10


def test_headline_no_interference_experiment():
    """Verify baseline experiment at delay=0 with no distractors gives exactly 1.0."""
    res = run_sparse_experiment(n_distractors=0, delay_steps_list=(0,), trials=50, seed=0)
    assert res[0] == 1.0


def test_headline_decay_experiment():
    """Verify memory decays over delay steps due to scalar shrink and synaptic noise."""
    res = run_sparse_experiment(n_distractors=0)
    assert res[0] == 1.0
    # Frozen model exact headline value at delay=20 is 0.667 (0.6669999999999999)
    assert res[20] == pytest.approx(0.667, abs=0.001)
    assert res[20] < res[0]


def test_headline_interference_experiment():
    """Verify distractors cause synaptic crosstalk, degrading recall beyond pure decay."""
    res_decay = run_sparse_experiment(n_distractors=0)
    res_interfere = run_sparse_experiment(n_distractors=20)

    # Frozen model exact headline value for 20 distractors at delay=20 is 0.629
    assert res_interfere[20] == pytest.approx(0.629, abs=0.001)
    # Interference must degrade accuracy strictly lower than pure decay
    assert res_interfere[20] < res_decay[20]


def test_headline_model_outcomes_protection():
    """Explicitly protect all three headline scientific model outcomes against regressions:
    - delay=0, distractors=0 -> 1.0
    - delay=20, distractors=0 -> 0.667
    - delay=20, distractors=20 -> 0.629
    """
    res_decay = run_sparse_experiment(n_distractors=0)
    res_interfere = run_sparse_experiment(n_distractors=20)

    assert res_decay[0] == 1.0, "Baseline (delay=0, dist=0) must be 1.0"
    assert res_decay[20] == pytest.approx(0.667, abs=0.002), "Decay outcome (delay=20, dist=0) must be ~0.667"
    assert res_interfere[20] == pytest.approx(0.629, abs=0.002), "Interference outcome (delay=20, dist=20) must be ~0.629"




# ==============================================================================
# 2. API ENDPOINT CONTRACTS & SCHEMA VALIDATION
# ==============================================================================

def test_api_health():
    """Verify /health returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_trial_endpoint_valid():
    """Verify /trial returns all expected keys and valid ranges."""
    payload = {
        "n": 200,
        "sparsity": 0.05,
        "delay": 0,
        "decay_rate": 0.03,
        "noise_std": 0.22,
        "n_distractors": 0,
        "reveal_fraction": 0.4,
        "seed": 42
    }
    response = client.post("/trial", json=payload)
    assert response.status_code == 200
    data = response.json()

    expected_keys = {
        "pattern", "cue", "known_mask", "raw_activation",
        "recalled", "weight_matrix", "accuracy", "k_active_units"
    }
    assert expected_keys.issubset(data.keys())
    assert data["k_active_units"] == 10
    assert data["accuracy"] == 1.0
    assert len(data["pattern"]) == 200
    assert len(data["recalled"]) == 200


def test_api_history_endpoint_valid():
    """Verify /history subsamples to at most 15 animation frames."""
    payload = {
        "n": 200,
        "sparsity": 0.05,
        "delay": 20,
        "decay_rate": 0.03,
        "noise_std": 0.22,
        "n_distractors": 5,
        "reveal_fraction": 0.4,
        "seed": 123
    }
    response = client.post("/history", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "history" in data
    assert len(data["history"]) <= 15
    assert 0.0 <= data["accuracy"] <= 1.0


def test_api_validation_boundaries():
    """Verify Pydantic input validation blocks illegal parameter combinations."""
    # n below minimum 20
    assert client.post("/trial", json={"n": 10}).status_code == 422

    # sparsity above maximum 0.5
    assert client.post("/trial", json={"sparsity": 0.6}).status_code == 422

    # negative delay
    assert client.post("/trial", json={"delay": -1}).status_code == 422

    # negative distractors
    assert client.post("/trial", json={"n_distractors": -5}).status_code == 422

    # reveal fraction below minimum 0.05
    assert client.post("/trial", json={"reveal_fraction": 0.01}).status_code == 422
