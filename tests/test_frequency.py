"""Frequency input: legacy path frozen, per_image path normalised and comparable."""

import os

import pytest
import torch

from src.config import CONFIG, override_config
from src.model import frequency_transform

REF_PATH = os.path.join(os.path.dirname(__file__), "assets", "freq_legacy_ref.pt")


@pytest.fixture(scope="module")
def ref():
    """Outputs captured from the pre-P3.1 code; the legacy path must still match."""
    if not os.path.isfile(REF_PATH):
        pytest.skip("reference tensor not generated")
    return torch.load(REF_PATH, weights_only=True)


def _x(seed=0, b=3):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, 3, 64, 64, generator=g)


# ── The legacy path must not move ────────────────────────────────


def test_legacy_fft_matches_the_reference(ref):
    with override_config(FREQ_NORM="none", DCT_ENABLED=False):
        out = frequency_transform(ref["input"])
    assert torch.equal(out, ref["fft"])


def test_legacy_dct_matches_the_reference(ref):
    with override_config(FREQ_NORM="none", DCT_ENABLED=True):
        out = frequency_transform(ref["input"])
    assert torch.equal(out, ref["dct"])


def test_none_is_the_default():
    """A FYP-1 checkpoint that says nothing about normalisation gets the legacy path."""
    assert CONFIG["FREQ_NORM"] == "none"


def test_legacy_dct_has_no_log(ref):
    """The FYP-1 DCT path returns a raw magnitude: non-negative, unlike the log-FFT.
    This asymmetry is the finding; the test pins it so it cannot be 'fixed' silently
    in the legacy path."""
    assert ref["dct"].min() >= 0.0
    assert ref["fft"].min() < 0.0


# ── The per_image path ───────────────────────────────────────────


@pytest.mark.parametrize("mode", ["fft", "dct"])
def test_per_image_is_standardised(mode):
    with override_config(FREQ_NORM="per_image", DCT_ENABLED=mode == "dct"):
        out = frequency_transform(_x(1))
    assert torch.isfinite(out).all()
    per_image_mean = out.mean(dim=(-2, -1))
    per_image_std = out.std(dim=(-2, -1))
    assert torch.allclose(per_image_mean, torch.zeros_like(per_image_mean), atol=1e-4)
    assert torch.allclose(per_image_std, torch.ones_like(per_image_std), atol=1e-3)


def test_per_image_makes_the_two_modes_comparable():
    """The point of P3.1: FFT and DCT inputs land on the same scale, so a trained
    comparison between them measures the transform rather than the magnitude."""
    x = _x(2)
    with override_config(FREQ_NORM="per_image", DCT_ENABLED=False):
        fft = frequency_transform(x)
    with override_config(FREQ_NORM="per_image", DCT_ENABLED=True):
        dct = frequency_transform(x)
    assert abs(fft.mean().item() - dct.mean().item()) < 1e-3
    assert abs(fft.std().item() - dct.std().item()) < 1e-2

    # Without normalisation they are orders of magnitude apart.
    with override_config(FREQ_NORM="none", DCT_ENABLED=False):
        fft_raw = frequency_transform(x)
    with override_config(FREQ_NORM="none", DCT_ENABLED=True):
        dct_raw = frequency_transform(x)
    assert abs(fft_raw.mean().item() - dct_raw.mean().item()) > 1.0


def test_constant_image_dct_is_dc_only():
    """A flat image has all its DCT energy at the (0,0) coefficient."""
    x = torch.full((1, 3, 32, 32), 0.5)
    with override_config(FREQ_NORM="none", DCT_ENABLED=True):
        out = frequency_transform(x)[0, 0]
    assert out[0, 0] > 0
    off_dc = out.clone()
    off_dc[0, 0] = 0
    assert off_dc.abs().max() < 1e-4


def test_explicit_arguments_override_the_config():
    """Callers can ask for a mode directly without mutating global state."""
    x = _x(3)
    with override_config(FREQ_NORM="none", DCT_ENABLED=False):
        explicit = frequency_transform(x, mode="dct", norm="per_image")
        implicit = frequency_transform(x)
    assert not torch.equal(explicit, implicit)
    assert abs(explicit.mean().item()) < 1e-4  # standardised


def test_output_shape_is_single_channel():
    for norm in ("none", "per_image"):
        with override_config(FREQ_NORM=norm):
            assert frequency_transform(_x(4, b=2)).shape == (2, 1, 64, 64)
