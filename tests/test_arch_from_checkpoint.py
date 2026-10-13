"""Eval loaders must run a checkpoint in the mode it was trained in (audit-fyp2 F2)."""

import pytest

from src.config import CONFIG, override_config
from src.model import apply_checkpoint_arch


@pytest.fixture(autouse=True)
def _restore_flags():
    with override_config(BASELINE_MODE=CONFIG["BASELINE_MODE"], DCT_ENABLED=CONFIG["DCT_ENABLED"]):
        yield


def test_baseline_checkpoint_sets_baseline_mode():
    CONFIG["BASELINE_MODE"] = False
    label = apply_checkpoint_arch(
        {"config": {"CONFIG": {"BASELINE_MODE": True, "DCT_ENABLED": False}}}
    )
    assert CONFIG["BASELINE_MODE"] is True
    assert label.startswith("BASELINE")


def test_full_checkpoint_clears_baseline_mode():
    CONFIG["BASELINE_MODE"] = True
    apply_checkpoint_arch({"config": {"CONFIG": {"BASELINE_MODE": False, "DCT_ENABLED": True}}})
    assert CONFIG["BASELINE_MODE"] is False
    assert CONFIG["DCT_ENABLED"] is True


def test_checkpoint_without_config_leaves_flags_alone():
    CONFIG["BASELINE_MODE"] = False
    apply_checkpoint_arch({"model_state": {}})
    assert CONFIG["BASELINE_MODE"] is False
