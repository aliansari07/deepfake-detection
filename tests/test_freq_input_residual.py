"""R5: the frequency branch can look at a high-frequency spatial residual instead of a spectrum.

The point of R5 is that a log-FFT magnitude map is the wrong input for a translation-equivariant,
globally-pooled, ImageNet-pretrained CNN (cause C1, audit-fyp2.md §16 and §20.3). Its entire
justification is R1's measurement of the alternative representation — residual probe 0.674 against
the spectrum's 0.561 (§15.1) — so the operator here must be the operator R1 probed, not merely
something similar. `test_residual_matches_the_probe_operator` is the test that keeps that true.
"""

import math

import cv2
import numpy as np
import pytest
import torch

from src.config import CONFIG, override_config
from src.model import (
    _RESIDUAL_KSIZE,
    _RESIDUAL_SIGMA,
    _gaussian_kernel_1d,
    apply_checkpoint_arch,
    arch_label,
    frequency_transform,
    highpass_residual,
)


@pytest.fixture(autouse=True)
def _restore_flags():
    with override_config(
        FREQ_INPUT=CONFIG.get("FREQ_INPUT", "spectrum"),
        FREQ_NORM=CONFIG["FREQ_NORM"],
        DCT_ENABLED=CONFIG["DCT_ENABLED"],
        BASELINE_MODE=CONFIG["BASELINE_MODE"],
        FREQ_ENABLED=CONFIG["FREQ_ENABLED"],
        TRANSFORMER_ENABLED=CONFIG["TRANSFORMER_ENABLED"],
    ):
        yield


def _x(seed=0, b=3, n=64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, 3, n, n, generator=g)


def _gray(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def _smooth(n=64):
    """A low-frequency image, which is spectrally what a face crop looks like.

    One cycle across the frame: a 5x5 sigma-1 Gaussian barely touches it, so almost nothing
    survives the high-pass. Random noise is the opposite case and misleads about scale.
    """
    y, x = torch.meshgrid(torch.linspace(0, 1, n), torch.linspace(0, 1, n), indexing="ij")
    img = 0.5 + 0.25 * torch.sin(2 * math.pi * x) + 0.2 * torch.cos(2 * math.pi * y)
    return img.clamp(0, 1).view(1, 1, n, n).expand(1, 3, n, n).contiguous()


# ── The legacy default must not move ─────────────────────────────


def test_spectrum_is_the_default():
    """Every checkpoint trained before R5 is a spectrum model; the default must keep them valid."""
    assert CONFIG.get("FREQ_INPUT", "spectrum") == "spectrum"


def test_default_config_leaves_the_spectrum_path_identical():
    x = _x(1)
    with override_config(FREQ_NORM="per_image", DCT_ENABLED=False):
        explicit = frequency_transform(x, freq_input="spectrum")
        implicit = frequency_transform(x)
    assert torch.equal(explicit, implicit)


# ── The operator is R1's operator ────────────────────────────────


def test_gaussian_kernel_matches_cv2():
    k_cv = cv2.getGaussianKernel(_RESIDUAL_KSIZE, _RESIDUAL_SIGMA).ravel()
    k_ours = _gaussian_kernel_1d(
        _RESIDUAL_KSIZE, _RESIDUAL_SIGMA, torch.device("cpu"), torch.float64
    ).numpy()
    assert np.allclose(k_cv, k_ours, atol=1e-12)


def test_residual_matches_the_probe_operator():
    """The guard on R5's evidential basis.

    tools/freq_probe.py's feat_residual computes `g - cv2.GaussianBlur(g, (5,5), 1.0)`, and its
    0.674 probe AUC is why R5 is being built at all. If this test fails, the experiment is no
    longer testing the representation that was measured.
    """
    rng = np.random.default_rng(0)
    g = rng.random((64, 64)).astype(np.float32)
    expected = g - cv2.GaussianBlur(g, (_RESIDUAL_KSIZE, _RESIDUAL_KSIZE), _RESIDUAL_SIGMA)
    got = highpass_residual(torch.from_numpy(g).view(1, 1, 64, 64)).numpy()[0, 0]
    assert np.abs(expected - got).max() < 1e-6


# ── It behaves like a high-pass filter ───────────────────────────


def test_constant_image_has_no_residual():
    """A flat field has no high-frequency content, and reflect padding must not invent any."""
    out = highpass_residual(torch.full((1, 1, 32, 32), 0.42))
    assert out.abs().max() < 1e-6


def test_residual_suppresses_a_smooth_gradient():
    """A linear ramp is almost all low frequency, so the residual should be near zero."""
    ramp = torch.linspace(0, 1, 64).view(1, 1, 1, 64).expand(1, 1, 64, 64).contiguous()
    assert highpass_residual(ramp).abs().mean() < 1e-3


def test_residual_keeps_fine_detail():
    """A one-pixel checkerboard is pure high frequency and must survive."""
    board = torch.zeros(1, 1, 32, 32)
    board[..., ::2, ::2] = 1.0
    board[..., 1::2, 1::2] = 1.0
    assert highpass_residual(board).abs().mean() > 0.1


# ── Wiring through frequency_transform ───────────────────────────


def test_residual_shape_is_single_channel():
    with override_config(FREQ_INPUT="residual", FREQ_NORM="per_image"):
        assert frequency_transform(_x(2, b=2)).shape == (2, 1, 64, 64)


def test_residual_differs_from_spectrum():
    x = _x(3)
    with override_config(FREQ_NORM="per_image"):
        spec = frequency_transform(x, freq_input="spectrum")
        res = frequency_transform(x, freq_input="residual")
    assert not torch.allclose(spec, res)


def test_freq_mode_is_irrelevant_to_the_residual():
    """No Fourier transform is taken, so fft and dct must give the same thing."""
    x = _x(4)
    with override_config(FREQ_INPUT="residual", FREQ_NORM="per_image", DCT_ENABLED=False):
        as_fft = frequency_transform(x)
    with override_config(FREQ_INPUT="residual", FREQ_NORM="per_image", DCT_ENABLED=True):
        as_dct = frequency_transform(x)
    assert torch.equal(as_fft, as_dct)


def test_residual_per_image_is_standardised():
    with override_config(FREQ_INPUT="residual", FREQ_NORM="per_image"):
        out = frequency_transform(_x(5))[0, 0]
    assert abs(out.mean().item()) < 1e-5
    assert abs(out.std().item() - 1.0) < 1e-3


def test_raw_residual_of_a_natural_image_is_badly_scaled():
    """Documents the scaling trap that train.py guards against.

    Natural images, faces included, are dominated by low frequencies, so their high-pass residual
    is a small fraction of the pixel range. Training on it un-normalised would starve the branch
    and look identical to R5's hypothesis being false, which is why train.py selects per_image
    unless told otherwise.

    White noise is the exception and is not what a face crop looks like: being entirely high
    frequency, its residual survives the filter nearly intact (std ~0.17 rather than ~0.001). The
    smooth input used here is the representative case.
    """
    smooth = _smooth()
    with override_config(FREQ_INPUT="residual", FREQ_NORM="none"):
        raw = frequency_transform(smooth)
    with override_config(FREQ_INPUT="residual", FREQ_NORM="per_image"):
        normed = frequency_transform(smooth)
    assert raw.std() < 0.02, "a low-frequency image should leave almost no residual"
    assert normed.std() > 10 * raw.std(), "per_image must materially rescale it"


def test_white_noise_residual_survives_the_filter():
    """The counterpart to the test above, so the exception is on the record rather than a surprise.

    A high-pass filter is supposed to keep high-frequency content. Uniform noise is all of it.
    """
    with override_config(FREQ_INPUT="residual", FREQ_NORM="none"):
        noisy = frequency_transform(_x(6))
    assert noisy.std() > 0.1


def test_residual_is_differentiable():
    """The branch is trained through this, so gradients must reach the input."""
    x = _x(7, b=1).requires_grad_(True)
    with override_config(FREQ_INPUT="residual", FREQ_NORM="per_image"):
        frequency_transform(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_residual_applied_to_luminance_not_raw_channels():
    """frequency_transform greys first, so the residual of the grey must be what comes out."""
    x = _x(8, b=1)
    with override_config(FREQ_INPUT="residual", FREQ_NORM="none"):
        got = frequency_transform(x)
    assert torch.allclose(got, highpass_residual(_gray(x).float()), atol=1e-6)


# ── Checkpoints describe which input they were trained on ───────


def test_arch_block_round_trips_residual():
    CONFIG["FREQ_INPUT"] = "spectrum"
    apply_checkpoint_arch(
        {
            "arch": {
                "baseline_mode": False,
                "freq_enabled": True,
                "transformer_enabled": True,
                "freq_mode": "fft",
                "freq_norm": "per_image",
                "freq_input": "residual",
            }
        }
    )
    assert CONFIG["FREQ_INPUT"] == "residual"


def test_pre_r5_checkpoint_is_read_as_spectrum():
    """A checkpoint whose arch block predates the switch must not inherit an R5 session's setting.

    Without this, evaluating M1 inside an R5 process would feed its frequency backbone a residual
    it was never trained on and report a wrong number with no warning.
    """
    CONFIG["FREQ_INPUT"] = "residual"
    apply_checkpoint_arch(
        {
            "arch": {
                "baseline_mode": False,
                "freq_enabled": True,
                "transformer_enabled": True,
                "freq_mode": "fft",
                "freq_norm": "per_image",
            }
        }
    )
    assert CONFIG["FREQ_INPUT"] == "spectrum"


def test_legacy_fyp1_checkpoint_is_read_as_spectrum():
    CONFIG["FREQ_INPUT"] = "residual"
    apply_checkpoint_arch({"config": {"CONFIG": {"BASELINE_MODE": False, "DCT_ENABLED": False}}})
    assert CONFIG["FREQ_INPUT"] == "spectrum"


def test_arch_label_names_the_residual_and_not_a_transform():
    """A label saying FFT on a run that takes no Fourier transform would be a false claim."""
    with override_config(
        BASELINE_MODE=False,
        FREQ_ENABLED=True,
        TRANSFORMER_ENABLED=True,
        FREQ_INPUT="residual",
        FREQ_NORM="per_image",
        DCT_ENABLED=True,
    ):
        label = arch_label()
    assert "RESIDUAL" in label
    assert "DCT" not in label and "FFT" not in label
