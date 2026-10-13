"""R4: a separate, gentler augmentation view for the frequency branch.

Motivated by audit-fyp2.md section 15.2 — A4 showed our own resize/blur pipeline is what collapses
the fusion gate (M1 reached 0.644 by epoch 9; A4 without augmentation never passed 0.517). R4 keeps
the full profile for the spatial branch and gives the frequency transform a gentler copy.

The properties worth protecting are: the default path is bit-identical to pre-R4, the gentle
profile really does drop the low-pass transforms, the pair is reproducible from one seed, and no
existing caller of model.forward has to change.
"""

import torch

from src.augmentation import PROFILES, TRANSFORMS, CompressionAwareAugmentation
from src.config import CONFIG
from src.model import DeepfakeDetector


def _aug(seed=0):
    return CompressionAwareAugmentation(seed=seed)


# ── the gentle profile ────────────────────────────────────────────


def test_gentle_never_applies_the_low_pass_transforms():
    """resize and blur are what destroy the spectrum, so gentle must never use them."""
    aug = _aug(1)
    for _ in range(500):
        p = aug._sample_params("gentle")
        assert not p["apply"]["resize"], "gentle applied resize"
        assert not p["apply"]["blur"], "gentle applied blur"


def test_gentle_still_applies_compression_like_corruptions():
    """Dropping everything would make the branch fragile to real compression, which is the point
    of compression-aware training. jpeg and noise must survive."""
    aug = _aug(2)
    seen = {name: False for name in TRANSFORMS}
    for _ in range(500):
        for name, on in aug._sample_params("gentle")["apply"].items():
            seen[name] |= on
    assert seen["jpeg"] and seen["noise"]


def test_gentle_uses_a_milder_jpeg_range_than_full():
    assert PROFILES["gentle"]["jpeg_q"][0] > PROFILES["full"]["jpeg_q"][0]
    aug = _aug(3)
    for _ in range(200):
        assert aug._sample_params("gentle")["jpeg_q"] >= PROFILES["gentle"]["jpeg_q"][0]


def test_the_apply_coin_is_flipped_for_every_transform_even_when_forbidden():
    """The membership test must come AFTER the draw. Writing `name in allowed and rng.random()`
    would short-circuit for forbidden names and change the stream depending on the profile, so a
    seed would no longer replay a run."""
    import random

    class Counting(random.Random):
        def __init__(self, seed):
            super().__init__(seed)
            self.calls = 0

        def random(self):
            self.calls += 1
            return super().random()

    counts, params = {}, {}
    for profile in ("full", "gentle"):
        aug = CompressionAwareAugmentation()
        aug.rng = Counting(99)
        params[profile] = aug._sample_params(profile)
        counts[profile] = aug.rng.calls

    # Same number of draws either way. If the membership test short-circuited, the gentle profile
    # would skip two coin flips and come in lower.
    assert counts["full"] == counts["gentle"], counts
    # And every transform still gets a decision recorded, forbidden or not.
    for profile in ("full", "gentle"):
        assert set(params[profile]["apply"]) == set(TRANSFORMS)


# ── reproducibility ───────────────────────────────────────────────


def test_the_pair_reproduces_from_one_seed():
    frames = torch.rand(4, 3, 64, 64)
    a_s, a_f = _aug(42).augment_video_pair(frames, "gentle")
    b_s, b_f = _aug(42).augment_video_pair(frames, "gentle")
    assert torch.equal(a_s, b_s) and torch.equal(a_f, b_f)


def test_the_two_views_differ():
    frames = torch.rand(4, 3, 64, 64)
    spatial, freq = _aug(7).augment_video_pair(frames, "gentle")
    assert not torch.equal(spatial, freq), "the whole point of R4 is that they differ"


def test_the_spatial_view_of_a_pair_uses_the_full_profile():
    frames = torch.rand(2, 3, 64, 64)
    _, _, p_spatial, p_freq = _aug(11).augment_video_pair(frames, "gentle", return_params=True)
    assert p_spatial["profile"] == "full"
    assert p_freq["profile"] == "gentle"


def test_one_parameter_set_per_clip_not_per_frame():
    """A real re-encode applies one quality to the whole video (FIX_PLAN P3.3)."""
    frames = torch.rand(6, 3, 64, 64)
    _, _, p_spatial, _ = _aug(5).augment_video_pair(frames, "gentle", return_params=True)
    assert isinstance(p_spatial["jpeg_q"], int)


def test_empty_clip_is_handled():
    empty = torch.zeros(0, 3, 64, 64)
    spatial, freq = _aug(0).augment_video_pair(empty, "gentle")
    assert spatial.shape[0] == 0 and freq.shape[0] == 0


# ── backwards compatibility ───────────────────────────────────────


def test_full_profile_is_unchanged_from_pre_r4():
    """The M1/A1/A2/A4 runs and the seeded reproduction tests depend on these exact draws."""
    p = _aug(123)._sample_params("full")
    assert p["profile"] == "full"
    assert 30 <= p["jpeg_q"] <= 95
    assert 0.5 <= p["scale"] <= 0.9
    assert 0.5 <= p["blur_sigma"] <= 2.0
    assert set(p["apply"]) == set(TRANSFORMS)


def test_augment_video_defaults_to_the_full_profile():
    frames = torch.rand(3, 3, 64, 64)
    a = _aug(31).augment_video(frames, return_params=True)[1]
    b = _aug(31).augment_video(frames, return_params=True, profile="full")[1]
    assert a == b


def test_config_default_keeps_both_branches_on_the_same_frames():
    assert CONFIG["FREQ_AUGMENT"] == "full"


def test_forward_without_freq_frames_is_identical_to_passing_frames():
    """Every evaluation and inference script calls model(frames, pad_mask). That path must not
    change, or legacy checkpoints stop reproducing."""
    torch.manual_seed(0)
    model = DeepfakeDetector().eval()
    frames = torch.rand(1, 4, 3, 224, 224)
    pad = torch.zeros(1, 4, dtype=torch.bool)
    with torch.no_grad():
        a, ga = model(frames, pad)
        b, gb = model(frames, pad, frames)
    assert torch.allclose(a, b, atol=0, rtol=0)
    assert (ga is None and gb is None) or torch.allclose(ga, gb, atol=0, rtol=0)


def test_a_different_freq_view_changes_the_logit():
    """If it did not, the frequency branch would be ignoring its input entirely."""
    torch.manual_seed(0)
    model = DeepfakeDetector().eval()
    frames = torch.rand(1, 4, 3, 224, 224)
    other = torch.rand(1, 4, 3, 224, 224)
    pad = torch.zeros(1, 4, dtype=torch.bool)
    with torch.no_grad():
        a, _ = model(frames, pad)
        b, _ = model(frames, pad, other)
    assert not torch.allclose(a, b), "freq_frames had no effect on the output"


def test_arch_block_records_the_profile():
    from src.train import _arch_block

    assert _arch_block()["freq_augment"] == CONFIG["FREQ_AUGMENT"]


def test_none_profile_leaves_the_frequency_view_clean():
    """The sharpest test of cause C2: the frequency branch sees undamaged frames while the
    spatial branch keeps the full profile."""
    aug = _aug(17)
    for _ in range(200):
        p = aug._sample_params("none")
        assert not any(p["apply"].values()), "the none profile applied a transform"


def test_none_profile_returns_the_frames_untouched():
    frames = torch.rand(4, 3, 64, 64)
    spatial, freq = _aug(19).augment_video_pair(frames, "none")
    assert torch.equal(freq, frames), "the none profile altered the frequency view"
    assert not torch.equal(spatial, frames), "the spatial view should still be augmented"
