#!/usr/bin/env python
"""Show / resume the pre-extraction cache for leftover videos only.

  python resume_preextract.py --show    # preview leftovers (no extraction)
  python resume_preextract.py           # extract ALL remaining videos
  python resume_preextract.py --folder original   # one class only

preextract.py is already resume-safe (skips valid caches), so this
is a thin wrapper that reports what's left, then defers to it.
"""

import argparse
import os
import subprocess
import sys

from src.config import CONFIG


def _cache_root():
    return os.path.join(CONFIG["DATASET_ROOT"], "cache")


def _classes(folder=None):
    root = CONFIG["DATASET_ROOT"]
    if folder:
        return [folder] if os.path.isdir(os.path.join(root, folder)) else []
    return sorted(
        d
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d not in ("cache", "csv", "archive.zip")
    )


def _videos(cls):
    folder = os.path.join(CONFIG["DATASET_ROOT"], cls)
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.endswith(".mp4"))


def _cache_path(cls, vid):
    return os.path.join(_cache_root(), cls, f"{os.path.splitext(vid)[0]}.pt")


def _valid_cache(path):
    if not os.path.exists(path):
        return False
    try:
        import torch

        t = torch.load(path, map_location="cpu", weights_only=True)
        return isinstance(t, torch.Tensor) and t.dim() == 4 and t.shape[1:] == (3, 224, 224)
    except Exception:
        return False


def main():
    """Show, or extract, the videos that still lack a cache file."""
    parser = argparse.ArgumentParser(description="Resume leftover face extraction")
    parser.add_argument("--folder", help="Restrict to one class")
    parser.add_argument("--show", action="store_true", help="Only show leftovers, don't extract")
    args = parser.parse_args()

    classes = _classes(args.folder)
    if not classes:
        print("No class folders found. Check DATASET_ROOT in src/config.py.")
        sys.exit(1)

    grand_left = grand_done = grand_total = 0
    print("\n=== CACHE STATUS ===")
    for cls in classes:
        videos = _videos(cls)
        done = sum(1 for v in videos if _valid_cache(_cache_path(cls, v)))
        left = len(videos) - done
        grand_done += done
        grand_total += len(videos)
        grand_left += left
        mark = "DONE" if left == 0 else ("PARTIAL" if done else "NOT STARTED")
        print(f"  {cls:<20} {done:>4}/{len(videos):<4} done  {left:>4} left  [{mark}]")
    print(f"\n  TOTAL: {grand_done}/{grand_total} cached, {grand_left} left to extract\n")

    if args.show:
        print("(--show: nothing extracted)")
        return

    if grand_left == 0:
        print("Nothing left to extract.")
        return

    print("Running preextract.py (auto-skips finished videos)...\n")
    cmd = [sys.executable, "preextract.py"]
    if args.folder:
        cmd += ["--folder", args.folder]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
