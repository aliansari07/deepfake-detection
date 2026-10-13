"""Augmentation must vary per worker and be constant within a clip (audit-fyp2 §5.2/§5.3)."""

import torch

from src.augmentation import TRANSFORMS, CompressionAwareAugmentation
from src.config import TRAIN_CONFIG
from src.dataset import worker_init_fn


def _clip(k=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(k, 3, 32, 32, generator=g)


def test_same_seed_reproduces_parameters():
    a = CompressionAwareAugmentation(seed=42)._sample_params()
    b = CompressionAwareAugmentation(seed=42)._sample_params()
    assert a == b


def test_workers_draw_different_parameter_streams():
    """Four workers with the shared TRAIN_CONFIG seed would augment identically;
    reseed() from the worker's own seed is what makes the streams differ."""
    shared = [
        CompressionAwareAugmentation(seed=TRAIN_CONFIG["SEED"])._sample_params() for _ in range(4)
    ]
    assert all(p == shared[0] for p in shared)  # the bug, if left unfixed

    reseeded = [
        CompressionAwareAugmentation().reseed(1000 + wid)._sample_params() for wid in range(4)
    ]
    assert len({repr(p) for p in reseeded}) == 4


def test_reseed_is_reproducible():
    a = CompressionAwareAugmentation(seed=1).reseed(77)._sample_params()
    b = CompressionAwareAugmentation(seed=2).reseed(77)._sample_params()
    assert a == b


def test_video_shares_one_parameter_set():
    aug = CompressionAwareAugmentation(seed=3)
    out, params = aug.augment_video(_clip(), return_params=True)
    assert out.shape == (6, 3, 32, 32)
    assert set(params["apply"]) == set(TRANSFORMS)
    assert 30 <= params["jpeg_q"] <= 95 and params["blur_k"] in (3, 5, 7)
    # One clip draws exactly one parameter set, so the next draw differs from
    # what a per-frame sampler would have produced by frame 2.
    assert aug._sample_params() != params


def test_deterministic_transforms_are_identical_across_frames():
    """With noise off, the same parameters applied to identical frames must give
    identical output — the per-video contract."""
    aug = CompressionAwareAugmentation(seed=5)
    p = aug._sample_params()
    p["apply"]["noise"] = False
    frame = torch.rand(3, 32, 32, generator=torch.Generator().manual_seed(9))
    a = aug._apply(frame, p)
    b = aug._apply(frame, p)
    assert torch.equal(a, b)


def test_empty_clip_is_handled():
    out = CompressionAwareAugmentation(seed=0).augment_video(torch.empty(0, 3, 32, 32))
    assert out.shape[0] == 0


def test_worker_init_fn_without_workers_is_a_noop():
    """Called outside a worker process (num_workers=0), there is nothing to reseed."""
    worker_init_fn(0)


def test_per_frame_api_still_works():
    """The inference TTA path and the demo still use per-frame sampling."""
    aug = CompressionAwareAugmentation(seed=11)
    out, logs = aug.augment_batch(torch.rand(3, 3, 32, 32))
    assert out.shape == (3, 3, 32, 32) and len(logs) == 3
    assert aug(torch.rand(3, 32, 32)).shape == (3, 32, 32)
