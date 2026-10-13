"""Celeb-DF v2 video lists (official test list and the FYP-1 custom sample).
No face-detector import, so extraction workers and tools can use it.
"""

import os
import sys

import numpy as np

from src.config import CONFIG
from src.logger import log

CELEB_ROOT = os.path.join(os.path.dirname(CONFIG["DATASET_ROOT"]), "Celeb-DF-v2")

REAL_DIRS = ["Celeb-real", "YouTube-real"]
FAKE_DIRS = ["Celeb-synthesis"]


def _list_videos(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".mp4", ".avi", ".mov"))
    )


def build_sample(n_real, n_fake, seed, force_all=False):
    """Return (videos, labels) — all real videos + a seeded fake subsample."""
    real_vids = []
    for d in REAL_DIRS:
        real_vids += _list_videos(os.path.join(CELEB_ROOT, d))
    fake_vids = _list_videos(os.path.join(CELEB_ROOT, FAKE_DIRS[0]))

    rng = np.random.default_rng(seed)
    real = sorted(real_vids)
    fake = sorted(fake_vids)

    if not force_all:
        if n_real is not None and len(real) > n_real:
            real = sorted(real[i] for i in rng.choice(len(real), n_real, replace=False))
        if n_fake is not None and len(fake) > n_fake:
            fake = sorted(fake[i] for i in rng.choice(len(fake), n_fake, replace=False))

    videos = real + fake
    labels = [0] * len(real) + [1] * len(fake)

    n_r = len(real)
    n_f = len(fake)
    if n_r == 0 or n_f == 0:
        print(f"ERROR: no videos under {CELEB_ROOT} (real={n_r}, fake={n_f})")
        sys.exit(1)

    log("CROSS", f"Sample: {n_r} real / {n_f} fake (seed={seed}) from {CELEB_ROOT}")
    return videos, labels


def build_official():
    """The standard Celeb-DF v2 test set. In the list file 1 = REAL, 0 = FAKE,
    the opposite of this project's labels (0 = REAL, 1 = FAKE)."""
    list_path = os.path.join(CELEB_ROOT, "List_of_testing_videos.txt")
    if not os.path.isfile(list_path):
        print(f"ERROR: official test list not found — {list_path}")
        sys.exit(1)
    videos, labels = [], []
    with open(list_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            flag, rel = line.split()
            videos.append(os.path.join(CELEB_ROOT, *rel.replace("\\", "/").split("/")))
            labels.append(0 if flag == "1" else 1)
    missing = [v for v in videos if not os.path.exists(v)]
    if missing:
        print(f"ERROR: {len(missing)} official test videos missing (first: {missing[0]})")
        sys.exit(1)
    log("CROSS", f"Official test list: {labels.count(0)} real / {labels.count(1)} fake")
    return videos, labels


def rel_key(video_path):
    """'Celeb-real/id0_0000' style key (no extension) for cache paths."""
    rel = os.path.relpath(video_path, CELEB_ROOT).replace(os.sep, "/")
    return os.path.splitext(rel)[0]
