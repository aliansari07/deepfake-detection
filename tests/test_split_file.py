"""The committed legacy split must match T4-20 and never overlap (audit-fyp2 F1 / §5.1)."""

import json
import os

import pytest

from src.config import CONFIG
from src.dataset import DeepfakeDataset


@pytest.fixture(scope="module")
def spec():
    with open(CONFIG["SPLIT_FILE"], encoding="utf-8") as f:
        return json.load(f)


def test_counts_match_t4_20(spec):
    def real(xs):
        return sum(x.startswith("original/") for x in xs)

    assert (len(spec["train"]), len(spec["val"])) == (5560, 1391)
    assert (real(spec["train"]), real(spec["val"])) == (793, 199)


def test_no_file_in_both_splits(spec):
    assert not set(spec["train"]) & set(spec["val"])
    assert len(set(spec["val"])) == len(spec["val"])


def test_late_files_excluded(spec):
    assert len(spec["excluded_late_files"]) == 49
    assert not set(spec["excluded_late_files"]) & (set(spec["train"]) | set(spec["val"]))


def test_missing_split_file_refuses_to_shuffle(tmp_path):
    with pytest.raises(FileNotFoundError):
        DeepfakeDataset._load_split(str(tmp_path), str(tmp_path / "nope.json"), "val")


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(CONFIG["DATASET_ROOT"], "cache")),
    reason="face cache not present on this machine",
)
def test_loader_returns_split_in_stored_order(spec):
    pairs = DeepfakeDataset._load_split(CONFIG["DATASET_ROOT"], CONFIG["SPLIT_FILE"], "val")
    rels = [
        os.path.relpath(p, os.path.join(CONFIG["DATASET_ROOT"], "cache")).replace(os.sep, "/")
        for p, _ in pairs
    ]
    assert rels == spec["val"]
    assert sum(1 for _, label in pairs if label == 0) == 199
