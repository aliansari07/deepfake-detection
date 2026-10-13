"""Padded frames must not influence the logit; no mask must equal the legacy path (P2.5)."""

import pytest
import torch

from src.config import CONFIG, override_config
from src.model import DeepfakeDetector


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = DeepfakeDetector().eval()
    return m


def _inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    n = CONFIG["SAMPLE_COUNT"]
    frames = torch.rand(1, n, 3, 224, 224, generator=g)
    mask = torch.zeros(1, n, dtype=torch.bool)
    mask[:, 7:] = True
    return frames, mask


@pytest.mark.parametrize("baseline", [False, True])
def test_padded_content_is_ignored(model, baseline):
    frames, mask = _inputs()
    zeros = frames.clone()
    zeros[:, 7:] = 0
    noise = frames.clone()
    noise[:, 7:] = torch.rand_like(noise[:, 7:])
    with override_config(BASELINE_MODE=baseline), torch.no_grad():
        a, ga = model(zeros, mask)
        b, gb = model(noise, mask)
    assert torch.allclose(a, b, atol=1e-5)
    if not baseline:
        assert torch.allclose(ga.mean(1), gb.mean(1), atol=1e-6)


def test_all_false_mask_equals_no_mask(model):
    frames, _ = _inputs(2)
    with override_config(BASELINE_MODE=False), torch.no_grad():
        a, ga = model(frames)
        b, gb = model(frames, torch.zeros(1, CONFIG["SAMPLE_COUNT"], dtype=torch.bool))
    assert torch.equal(a, b) and torch.equal(ga, gb)
