"""The FYP-2 split must be identity- and source-clip-disjoint (P2.1/P2.3)."""

import json
import os

import pytest

from src.config import CONFIG
from src.dataset import DeepfakeDataset
from tools import build_split as bs

SPLIT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "splits", "ffpp_official_v1.json"
)


def test_official_pair_lists_are_disjoint():
    pairs = bs.load_pairs()
    ids = {s: {i for p in pairs[s] for i in p} for s in bs.SPLITS}
    assert {s: len(v) for s, v in ids.items()} == {"train": 720, "val": 140, "test": 140}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]


def test_candidates_cover_both_directions_of_every_pair():
    cand = bs.candidates(bs.load_pairs())
    assert {s: len(v) for s, v in cand.items()} == {
        "train": 720 + 3600,
        "val": 140 + 700,
        "test": 140 + 700,
    }
    assert "Deepfakes/071_054" in cand["train"] and "Deepfakes/054_071" in cand["train"]


def test_verify_rejects_identity_leak():
    spec = {
        "train": ["original/000.pt", "Deepfakes/000_003.pt"],
        "val": ["FaceSwap/003_000.pt"],
        "test": ["original/999.pt"],
    }
    with pytest.raises(AssertionError):
        bs.verify(spec)


@pytest.fixture(scope="module")
def spec():
    if not os.path.isfile(SPLIT_FILE):
        pytest.skip("splits/ffpp_official_v1.json not built yet")
    with open(SPLIT_FILE, encoding="utf-8") as f:
        return json.load(f)


def test_committed_split_is_disjoint_and_excludes_dfd(spec):
    bs.verify(spec)
    for s in bs.SPLITS:
        assert not any(k.startswith("DeepFakeDetection/") for k in spec[s])
    assert spec["cache_version"] == 2 and spec["cache_dir"] == "cache_v2"


def test_committed_split_sizes_are_close_to_official(spec):
    expected = {"train": 4320, "val": 840, "test": 840}
    for s, n in expected.items():
        assert len(spec[s]) + len(spec["excluded"][s]) == n
        assert len(spec["excluded"][s]) <= 0.01 * n  # < 1% lost to extraction


def test_test_split_is_locked(spec):
    with pytest.raises(PermissionError):
        DeepfakeDataset(CONFIG["DATASET_ROOT"], split="test", split_file=SPLIT_FILE)


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(CONFIG["DATASET_ROOT"], "cache_v2")),
    reason="cache v2 not present on this machine",
)
def test_every_listed_file_matches_manifest(spec):
    root = os.path.join(CONFIG["DATASET_ROOT"], "cache_v2")
    with open(os.path.join(root, "manifest.json"), encoding="utf-8") as f:
        files = json.load(f)["files"]
    for s in bs.SPLITS:
        for rel in spec[s]:
            e = files[rel[:-3]]
            assert os.path.getsize(os.path.join(root, *rel.split("/"))) == e["size"]
