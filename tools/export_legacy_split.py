#!/usr/bin/env python
"""Freeze the train/val split that T4-20 (and every FYP-1 result)
was produced with into splits/legacy_t4-20.json.

DeepfakeDataset used to shuffle the cache file list at runtime,
so adding files to the cache silently re-drew the split. The 49
videos recovered by the bbox-clamp fix were cached on
2026-09-17 10:05-10:15, after all experiments; excluding them and
replaying the old seeded shuffle reproduces the original split
(5,560 / 1,391, 199 real / 1,192 fake).

NOTE: this split is file-level and identity-overlapping. It exists
only to reproduce FYP-1 results — see audit-fyp2.md F1.

Usage:
  python tools/export_legacy_split.py            # writes splits/legacy_t4-20.json
  python tools/export_legacy_split.py --check    # verify existing file only
"""

import argparse
import datetime
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import CONFIG, TRAIN_CONFIG  # noqa: E402

FAKE_SUBDIRS = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
    "DeepFakeDetection",
    "FaceShifter",
]
LATE_CUTOFF = datetime.datetime(2026, 9, 17, 10, 0, 0)
OUT = os.path.join(ROOT, "splits", "legacy_t4-20.json")


def _rel(path, cache_root):
    return os.path.relpath(path, cache_root).replace(os.sep, "/")


def build():
    """Reconstruct the FYP-1 (T4-20) train/val split from the legacy cache."""
    cache_root = os.path.join(CONFIG["DATASET_ROOT"], "cache")
    late = []

    def find(sub, label):
        out = []
        for p in sorted(glob.glob(os.path.join(cache_root, sub, "*.pt"))):
            if datetime.datetime.fromtimestamp(os.path.getmtime(p)) >= LATE_CUTOFF:
                late.append(_rel(p, cache_root))
                continue
            out.append((p, label))
        return out

    real = find("original", 0)
    fake = []
    for d in FAKE_SUBDIRS:
        fake += find(d, 1)

    def do_split(items, ratio, rng):
        items = list(items)
        rng.shuffle(items)
        n = int(len(items) * ratio)
        return items[:n], items[n:]

    # Exact replay of the old DeepfakeDataset.__init__: each split instance
    # re-created the rng, split reals then fakes, then shuffled its own pairs.
    order = {}
    for split in ("train", "val"):
        rng = random.Random(TRAIN_CONFIG["SEED"])
        real_tr, real_va = do_split(real, TRAIN_CONFIG["TRAIN_RATIO"], rng)
        fake_tr, fake_va = do_split(fake, TRAIN_CONFIG["TRAIN_RATIO"], rng)
        pairs = (real_tr + fake_tr) if split == "train" else (real_va + fake_va)
        rng.shuffle(pairs)
        order[split] = [_rel(p, cache_root) for p, _ in pairs]

    return {
        "name": "legacy_t4-20",
        "description": (
            "File-level seeded split used by T4-20 and all FYP-1 results. "
            "IDENTITY-OVERLAPPING (audit-fyp2.md F1) — reproduction only."
        ),
        "created": datetime.date.today().isoformat(),
        "seed": TRAIN_CONFIG["SEED"],
        "train_ratio": TRAIN_CONFIG["TRAIN_RATIO"],
        "cache_dir": "cache",
        "label_rule": "original/* = 0 (REAL), everything else = 1 (FAKE)",
        "excluded_late_files": sorted(late),
        "train": order["train"],
        "val": order["val"],
        "test": [],
    }


def check(spec):
    """Sanity-check the class balance and sizes of a split spec."""
    tr, va = spec["train"], spec["val"]

    def n_real(xs):
        return sum(1 for x in xs if x.startswith("original/"))

    ok = (len(tr), len(va), n_real(tr), n_real(va)) == (5560, 1391, 793, 199)
    overlap = set(tr) & set(va)
    print(
        f"train {len(tr)} ({n_real(tr)} real) | val {len(va)} ({n_real(va)} real) | "
        f"overlap {len(overlap)} | late excluded {len(spec['excluded_late_files'])}"
    )
    return ok and not overlap


def main():
    """Write the legacy split file, or verify it with ``--check``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        with open(OUT, encoding="utf-8") as f:
            spec = json.load(f)
    else:
        spec = build()
    if not check(spec):
        print("ERROR: split does not match the T4-20 counts (5,560 / 1,391).")
        sys.exit(1)
    if not args.check:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=1)
        print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
