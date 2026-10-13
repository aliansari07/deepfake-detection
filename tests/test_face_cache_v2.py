"""Cache v2 loading: RGB uint8 in, zero padding + mask out, no repeated frames (P2.2/P2.3)."""

import json
import os

import torch

from src.config import CONFIG
from src.dataset import DeepfakeDataset
from src.face_cache import CACHE_VERSION, load_frames, to_sequence


def _write_v2(path, k):
    frames = torch.zeros((k, 3, 224, 224), dtype=torch.uint8)
    frames[:, 0] = 255  # pure red in RGB order
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "version": CACHE_VERSION,
            "channel_order": "RGB",
            "dtype": "uint8",
            "frames": frames,
            "n_detected": k,
            "frame_idx": list(range(k)),
        },
        path,
    )


def test_to_sequence_zero_pads_and_masks():
    frames = torch.rand(7, 3, 4, 4)
    out, mask = to_sequence(frames, 10)
    assert out.shape == (10, 3, 4, 4)
    assert mask.tolist() == [False] * 7 + [True] * 3
    assert torch.equal(out[:7], frames) and out[7:].abs().sum() == 0


def test_to_sequence_truncates_without_mask():
    out, mask = to_sequence(torch.rand(12, 3, 4, 4), 10)
    assert out.shape[0] == 10 and not mask.any()


def test_dataset_reads_v2_split(tmp_path):
    root = tmp_path
    _write_v2(str(root / "cache_v2" / "original" / "000.pt"), 10)
    _write_v2(str(root / "cache_v2" / "Deepfakes" / "000_003.pt"), 6)
    split = root / "split.json"
    split.write_text(
        json.dumps(
            {
                "cache_dir": "cache_v2",
                "cache_version": 2,
                "val": ["original/000.pt", "Deepfakes/000_003.pt"],
            }
        )
    )

    ds = DeepfakeDataset(str(root), split="val", split_file=str(split))
    assert ds.channel_order == "RGB"

    frames, label, mask = ds[1]
    assert label.item() == 1
    assert frames.shape == (CONFIG["SAMPLE_COUNT"], 3, 224, 224)
    assert mask.sum().item() == CONFIG["SAMPLE_COUNT"] - 6
    assert frames[:6, 0].min() == 1.0 and frames[:6, 1:].max() == 0.0  # red stays in channel 0
    assert frames[6:].abs().sum() == 0  # zero padding, not repeats

    frames, label, mask = ds[0]
    assert label.item() == 0 and not mask.any()


def test_load_frames_rejects_legacy_tensor(tmp_path):
    p = tmp_path / "legacy.pt"
    torch.save(torch.rand(10, 3, 224, 224), p)
    try:
        load_frames(str(p))
    except RuntimeError:
        return
    raise AssertionError("legacy tensor accepted as cache v2")


def test_legacy_split_returns_empty_mask():
    split = CONFIG["SPLIT_FILE"]
    if not os.path.isdir(os.path.join(CONFIG["DATASET_ROOT"], "cache")) or "legacy" not in split:
        return
    ds = DeepfakeDataset(CONFIG["DATASET_ROOT"], split="val")
    frames, _, mask = ds[0]
    assert ds.channel_order == "BGR" and frames.dtype == torch.float32 and not mask.any()
