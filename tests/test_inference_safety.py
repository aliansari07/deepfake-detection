"""Inference must be deterministic and must never run on random weights (audit-fyp2 F3, §4.6b)."""

import pytest
from conftest import stub_face_detector

from src.config import CONFIG


def test_inference_augmentation_disabled_by_default():
    assert CONFIG["AUGMENT_ENABLED"] is False


def test_missing_checkpoint_raises():
    stub_face_detector()
    from src.inference import _load_model

    with pytest.raises(FileNotFoundError):
        _load_model("checkpoints/does-not-exist.pt")
