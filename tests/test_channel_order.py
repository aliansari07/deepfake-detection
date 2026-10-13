"""The RGB inference path (PLAN task F1).

Two data paths disagree about colour. `src/face_cache.py` converts crops to RGB before caching, so
every model trained on the v2 cache learned RGB. `src/preprocess.py` crops straight out of an
OpenCV frame and never converts, so live inference produces BGR.

Feeding a v2 model BGR does not raise; it returns a confident, wrong probability. That is why this
is tested rather than trusted, and why the legacy fallback must stay BGR so T4-20 is untouched.
"""

import torch

from src.model import checkpoint_channel_order, to_channel_order

# ── what a checkpoint declares ────────────────────────────────────


def test_a_v2_checkpoint_declares_rgb():
    ckpt = {"arch": {"baseline_mode": False, "channel_order": "RGB"}}
    assert checkpoint_channel_order(ckpt) == "RGB"


def test_a_legacy_checkpoint_falls_back_to_bgr():
    """FYP-1 checkpoints have no arch block and trained on the v1 BGR cache. If this ever returned
    RGB, T4-20 would silently stop reproducing AUC 0.9912."""
    assert checkpoint_channel_order({"config": {"CONFIG": {}}}) == "BGR"
    assert checkpoint_channel_order({}) == "BGR"


def test_an_arch_block_without_the_field_falls_back_to_bgr():
    assert checkpoint_channel_order({"arch": {"baseline_mode": True}}) == "BGR"


def test_the_declared_order_is_case_insensitive():
    assert checkpoint_channel_order({"arch": {"channel_order": "rgb"}}) == "RGB"


# ── the conversion itself ─────────────────────────────────────────


def test_conversion_swaps_red_and_blue():
    frames = torch.zeros(2, 3, 4, 4)
    frames[:, 0] = 1.0  # channel 0 hot
    out = to_channel_order(frames, "BGR", "RGB")
    assert out[:, 2].eq(1.0).all(), "channel 0 should have moved to channel 2"
    assert out[:, 0].eq(0.0).all()


def test_conversion_leaves_the_green_channel_alone():
    frames = torch.rand(3, 3, 8, 8)
    out = to_channel_order(frames, "BGR", "RGB")
    assert torch.equal(out[:, 1], frames[:, 1])


def test_matching_orders_are_a_no_op_and_return_the_same_object():
    frames = torch.rand(2, 3, 4, 4)
    assert to_channel_order(frames, "BGR", "BGR") is frames
    assert to_channel_order(frames, "RGB", "rgb") is frames


def test_conversion_is_its_own_inverse():
    frames = torch.rand(2, 3, 5, 5)
    there = to_channel_order(frames, "BGR", "RGB")
    back = to_channel_order(there, "RGB", "BGR")
    assert torch.equal(back, frames)


def test_conversion_works_on_an_unbatched_clip():
    frames = torch.rand(10, 3, 224, 224)
    assert to_channel_order(frames, "BGR", "RGB").shape == frames.shape


def test_conversion_works_on_a_batched_clip():
    frames = torch.rand(2, 10, 3, 224, 224)
    out = to_channel_order(frames, "BGR", "RGB")
    assert out.shape == frames.shape
    assert torch.equal(out[:, :, 1], frames[:, :, 1])


# ── the wiring ────────────────────────────────────────────────────


def test_inference_converts_only_when_the_checkpoint_asks_for_it():
    """probe_video reads model.channel_order; an unset attribute must mean BGR, so any caller
    that builds a model by hand keeps the pre-F1 behaviour."""
    import inspect

    from src import inference

    src = inspect.getsource(inference.probe_video)
    assert 'getattr(model, "channel_order", "BGR")' in src, (
        "the default must be BGR or legacy callers change behaviour silently"
    )
    assert "to_channel_order(raw_tensor" in src


def test_load_model_records_the_order_on_the_model():
    import inspect

    from src import inference

    assert "model.channel_order = checkpoint_channel_order(ckpt)" in inspect.getsource(
        inference._load_model
    )
