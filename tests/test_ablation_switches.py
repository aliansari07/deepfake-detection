"""Ablation switches must change only what they name and leave the full model alone (P3.2)."""

import pytest
import torch

from src.config import CONFIG, override_config
from src.model import DeepfakeDetector, apply_checkpoint_arch, arch_label


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return DeepfakeDetector().eval()


def _frames(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, CONFIG["SAMPLE_COUNT"], 3, 224, 224, generator=g)


def _run(m, frames, **flags):
    with override_config(**flags), torch.no_grad():
        return m(frames)


def test_defaults_are_the_full_model(model):
    """The switches default to on, so the full path must be bit-identical to
    the pre-ablation computation."""
    frames = _frames()
    a, ga = _run(model, frames)
    b, gb = _run(model, frames, FREQ_ENABLED=True, TRANSFORMER_ENABLED=True)
    assert torch.equal(a, b) and torch.equal(ga, gb)


def test_no_freq_keeps_the_transformer_and_drops_the_gate(model):
    """A1: the run BASELINE_MODE cannot express."""
    logit, gate = _run(model, _frames(), FREQ_ENABLED=False)
    assert logit.shape == (1, 1)
    assert gate is None  # no gate without a frequency branch


def test_no_freq_changes_the_prediction(model):
    frames = _frames(2)
    full, _ = _run(model, frames)
    ablated, _ = _run(model, frames, FREQ_ENABLED=False)
    assert not torch.allclose(full, ablated)


def test_no_transformer_keeps_the_gate(model):
    logit, gate = _run(model, _frames(3), TRANSFORMER_ENABLED=False)
    assert logit.shape == (1, 1)
    assert gate is not None and gate.shape == (1, CONFIG["SAMPLE_COUNT"])


def test_baseline_mode_still_wins_over_the_new_switches(model):
    """A2 uses BASELINE_MODE so it reproduces FYP-1's B6 exactly; the newer
    switches must not interfere with that path."""
    frames = _frames(4)
    a, ga = _run(model, frames, BASELINE_MODE=True)
    b, gb = _run(model, frames, BASELINE_MODE=True, FREQ_ENABLED=True, TRANSFORMER_ENABLED=True)
    assert torch.equal(a, b) and ga is None and gb is None


def test_ablation_switches_respect_the_padding_mask(model):
    """Padded content must stay ignorable with the frequency branch off."""
    frames = _frames(5)
    mask = torch.zeros(1, CONFIG["SAMPLE_COUNT"], dtype=torch.bool)
    mask[:, 6:] = True
    zeros = frames.clone()
    zeros[:, 6:] = 0
    noise = frames.clone()
    noise[:, 6:] = torch.rand_like(noise[:, 6:])
    with override_config(FREQ_ENABLED=False), torch.no_grad():
        a, _ = model(zeros, mask)
        b, _ = model(noise, mask)
    assert torch.allclose(a, b, atol=1e-5)


# ── Round-tripping through a checkpoint ──────────────────────────


def test_checkpoint_restores_the_ablation_switches():
    with override_config(FREQ_ENABLED=True, TRANSFORMER_ENABLED=True):
        label = apply_checkpoint_arch(
            {
                "arch": {
                    "baseline_mode": False,
                    "freq_mode": "fft",
                    "freq_enabled": False,
                    "transformer_enabled": True,
                }
            }
        )
        assert CONFIG["FREQ_ENABLED"] is False
        assert CONFIG["TRANSFORMER_ENABLED"] is True
        assert label.startswith("ABLATED") and "freq=OFF" in label


def test_legacy_checkpoint_forces_both_branches_on():
    """A FYP-1 checkpoint has no switches; it must not inherit an ablation left
    set by whatever ran earlier in the same process."""
    with override_config(FREQ_ENABLED=False, TRANSFORMER_ENABLED=False):
        label = apply_checkpoint_arch(
            {"config": {"CONFIG": {"BASELINE_MODE": False, "DCT_ENABLED": False}}}
        )
        assert CONFIG["FREQ_ENABLED"] is True
        assert CONFIG["TRANSFORMER_ENABLED"] is True
        assert label == "FULL | freq=FFT"


def test_arch_label_names_each_configuration():
    with override_config(BASELINE_MODE=True):
        assert arch_label().startswith("BASELINE")
    with override_config(BASELINE_MODE=False, DCT_ENABLED=True):
        assert arch_label() == "FULL | freq=DCT"
    with override_config(BASELINE_MODE=False, TRANSFORMER_ENABLED=False):
        assert "transformer=OFF" in arch_label()
