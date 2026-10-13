"""Checkpoints must record the architecture, split and RNG they were made with (FIX_PLAN P3.4)."""

import random

import numpy as np
import pytest
import torch

import src.train as T
from src.config import CONFIG, override_config
from src.model import apply_checkpoint_arch


@pytest.fixture(autouse=True)
def _restore_flags():
    with override_config(BASELINE_MODE=CONFIG["BASELINE_MODE"], DCT_ENABLED=CONFIG["DCT_ENABLED"]):
        yield


# ── The arch block ───────────────────────────────────────────────


def test_arch_block_records_the_training_switches():
    with override_config(BASELINE_MODE=True, DCT_ENABLED=True):
        arch = T._arch_block()
    assert arch["baseline_mode"] is True and arch["freq_mode"] == "dct"
    assert arch["sample_count"] == CONFIG["SAMPLE_COUNT"]


def test_arch_block_is_preferred_over_the_legacy_config_dump():
    """A new checkpoint's explicit arch wins even if the CONFIG dump disagrees."""
    CONFIG["BASELINE_MODE"] = False
    label = apply_checkpoint_arch(
        {
            "arch": {"baseline_mode": True, "freq_mode": "fft"},
            "config": {"CONFIG": {"BASELINE_MODE": False, "DCT_ENABLED": True}},
        }
    )
    assert CONFIG["BASELINE_MODE"] is True and CONFIG["DCT_ENABLED"] is False
    assert label.startswith("BASELINE")


def test_legacy_checkpoint_without_arch_still_loads():
    """Every FYP-1 checkpoint predates the arch block and must keep working."""
    CONFIG["BASELINE_MODE"] = False
    apply_checkpoint_arch({"config": {"CONFIG": {"BASELINE_MODE": True, "DCT_ENABLED": False}}})
    assert CONFIG["BASELINE_MODE"] is True


# ── Split provenance ─────────────────────────────────────────────


def test_split_provenance_hashes_the_split_file():
    prov = T._split_provenance()
    assert prov["file"] and prov["sha256"] and len(prov["sha256"]) == 64


def test_split_provenance_survives_a_missing_file():
    with override_config(SPLIT_FILE="does/not/exist.json"):
        assert T._split_provenance()["sha256"] is None


# ── RNG state ────────────────────────────────────────────────────


def test_rng_round_trip_restores_all_streams():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    state = T._rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    assert T._restore_rng(state) is True
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected


def test_rng_state_survives_a_torch_save_round_trip(tmp_path):
    """torch.save turns the nested tuples into lists; restoring must still work."""
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    path = tmp_path / "rng.pt"
    torch.save({"rng": T._rng_state()}, path)
    expected = (random.random(), float(np.random.rand()))

    random.seed(0)
    np.random.seed(0)
    loaded = torch.load(path, weights_only=False)["rng"]
    assert T._restore_rng(loaded) is True
    assert (random.random(), float(np.random.rand())) == expected


def test_missing_rng_block_is_not_an_error():
    assert T._restore_rng(None) is False


# ── Threshold ────────────────────────────────────────────────────


def test_youden_threshold_separates_a_clean_split():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert 0.3 < T._youden_threshold(labels, probs) <= 0.7


def test_youden_threshold_falls_back_on_one_class():
    assert T._youden_threshold(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4])) == 0.5


# ── The threshold actually reaches the checkpoint (audit-fyp2 §13.3) ──


def test_save_ckpt_stores_the_computed_threshold(tmp_path):
    """The training loop used to rebuild a four-key metrics dict at the save
    sites, dropping `threshold` and `balanced_acc`, so every checkpoint recorded
    the 0.5 fallback. Save a realistic validation dict and check it survives."""
    import torch.nn as nn

    model = nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    va = {
        "loss": 0.4,
        "acc": 0.8845,
        "f1": 0.9306,
        "auc": 0.9084,
        "gate": 0.626,
        "threshold": 0.9442,
        "balanced_acc": 0.7964,
    }
    metrics = {**va, "accuracy": va["acc"], "epoch": 14}

    path = tmp_path / "ck.pt"
    T._save_ckpt(model, opt, 14, {}, metrics, str(path))
    ck = torch.load(path, weights_only=False)

    assert ck["threshold"] == pytest.approx(0.9442)
    assert ck["threshold"] != 0.5, "threshold fell back to the 0.5 default"
    assert ck["metrics"]["balanced_acc"] == pytest.approx(0.7964)


def test_epoch_metrics_carries_every_required_field():
    """The call site that builds the checkpoint's metrics block is where the
    fields were being dropped, so test it directly."""
    va = {
        "loss": 0.4,
        "acc": 0.88,
        "f1": 0.93,
        "auc": 0.91,
        "gate": 0.63,
        "threshold": 0.9442,
        "balanced_acc": 0.7964,
    }
    m = T._epoch_metrics(va, 14)
    for key in T.CKPT_METRIC_KEYS:
        assert key in m, f"{key} dropped on the way to the checkpoint"
    assert m["accuracy"] == pytest.approx(0.88) and m["epoch"] == 14


def test_epoch_metrics_refuses_an_incomplete_validation_result():
    """The old four-key dict must now fail loudly instead of silently storing
    the 0.5 threshold fallback (audit-fyp2 13.3)."""
    with pytest.raises(KeyError):
        T._epoch_metrics({"acc": 0.88, "f1": 0.93, "auc": 0.91}, 14)


def test_validate_returns_threshold_and_balanced_accuracy():
    """_validate() is the source of both fields; if it stops returning them,
    _epoch_metrics raises rather than falling back."""
    import inspect

    src = inspect.getsource(T._validate)
    assert '"threshold"' in src and '"balanced_acc"' in src
