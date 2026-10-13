#!/usr/bin/env python
r"""Video-level compression robustness (PLAN task C1)

Re-encodes whole videos with H.264 at escalating CRF, re-detects faces from
the re-encoded file, and measures how each model degrades. This is the honest
version of compress_test.py, which re-compresses already-extracted 224x224
crops as JPEG: that misses what real compression does to a video, because the
face detector never sees the degraded frames and temporal artifacts are absent.

WHICH SPLIT: validation, not test.
  The binder specified the 840 test videos. That predates the test-access
  discipline added in task D1: this study needs one read per model per CRF
  level, so on test it would cost 2 models x 4 levels = 8 reads and destroy
  the once-per-model guarantee for a result that is about RELATIVE degradation.
  The slope of the curve is what the S3 claim rests on, and val measures that
  as well as test would. Deviation recorded in audit-fyp2.md.

Usage:
  python compress_video_test.py --checkpoints checkpoints/M1-20.best.pt \
      checkpoints/A4-20.best.pt --limit 200
  python compress_video_test.py --crf 23 28 35 40 --limit 0     # everything
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from src.config import CONFIG, DEVICE
from src.dataset import DeepfakeDataset
from src.face_cache import to_sequence
from src.logger import section
from src.metrics import DEFAULT_RESAMPLES, DEFAULT_SEED, auc_ci
from src.model import DeepfakeDetector, apply_checkpoint_arch

CACHE_ROOT = os.path.join("checkpoints", "crf_cache")


def ffmpeg_exe():
    """Return the path of the ffmpeg binary bundled with ``imageio-ffmpeg``."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def cache_path(crf, video_path):
    """One face tensor per (crf, video), so an interrupted run resumes."""
    root = os.path.join(CONFIG["DATASET_ROOT"], "")
    rel = video_path.replace(root, "").replace(".mp4", ".pt")
    return os.path.join(CACHE_ROOT, f"crf{crf}", rel)


def reencode(src, dst, crf):
    """H.264 at the given CRF. -preset veryfast keeps 3,000 re-encodes tractable."""
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        src,
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "veryfast",
        "-an",
        dst,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def faces_at_crf(video_path, crf):
    """Face tensor for this video re-encoded at `crf`, cached on disk.

    Returns (frames (T,3,224,224), pad_mask) or (None, None) when no face
    survives the compression — itself a meaningful failure at high CRF.
    """
    cp = cache_path(crf, video_path)
    if os.path.exists(cp):
        rec = torch.load(cp, map_location="cpu", weights_only=True)
        if rec["frames"].shape[0] == 0:
            return None, None
        return to_sequence(rec["frames"].float() / 255.0, CONFIG["SAMPLE_COUNT"])

    from src.preprocess import extract_faces  # heavy: only when needed

    os.makedirs(os.path.dirname(cp), exist_ok=True)
    tmp = os.path.join(
        tempfile.gettempdir(), f"crf{crf}_{os.getpid()}_{os.path.basename(video_path)}"
    )
    try:
        reencode(video_path, tmp, crf)
        frames, _ = extract_faces(tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    if frames is None:
        frames = torch.zeros(0, 3, 224, 224)
    torch.save({"frames": (frames * 255.0).round().clamp(0, 255).to(torch.uint8)}, cp)
    if frames.shape[0] == 0:
        return None, None
    return to_sequence(frames, CONFIG["SAMPLE_COUNT"])


def load_model(path):
    """Load a checkpoint, apply its architecture and return ``(model, checkpoint)``."""
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    apply_checkpoint_arch(ck)
    m = DeepfakeDetector().to(DEVICE)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m, ck


def main():
    """Re-encode validation videos at each CRF and report AUC with bootstrap CIs."""
    ap = argparse.ArgumentParser(description="Video-level CRF robustness (PLAN C1)")
    ap.add_argument(
        "--checkpoints",
        nargs="+",
        default=["checkpoints/M1-20.best.pt", "checkpoints/A4-20.best.pt"],
    )
    ap.add_argument("--crf", nargs="+", type=int, default=[23, 28, 35, 40])
    ap.add_argument(
        "--limit",
        type=int,
        default=200,
        help="videos per class to use; 0 means all. The curve's SLOPE is the "
        "result, and a stratified subsample measures it at a fraction of "
        "the face-detection cost",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_RESAMPLES)
    ap.add_argument("--out", default="results/crf_curves.json")
    args = ap.parse_args()

    ds = DeepfakeDataset(CONFIG["DATASET_ROOT"], split="val")
    items = [
        (p.replace(os.sep + "cache_v2" + os.sep, os.sep).replace(".pt", ".mp4"), l)
        for p, l in ds.samples
    ]
    items = [(v, l) for v, l in items if os.path.exists(v)]

    if args.limit:
        rng = np.random.default_rng(args.seed)
        by = {0: [i for i in items if i[1] == 0], 1: [i for i in items if i[1] == 1]}
        picked = []
        for _label, group in by.items():
            take = min(args.limit, len(group))
            picked += [group[i] for i in rng.choice(len(group), take, replace=False)]
        items = picked

    n_real = sum(1 for _, l in items if l == 0)
    section("VIDEO-LEVEL CRF ROBUSTNESS (task C1)")
    print("  Split      : VAL  (not test — see the header of this file)")
    print(f"  Videos     : {len(items)}  ({n_real} real / {len(items) - n_real} fake)")
    print(f"  CRF levels : {args.crf}")
    print(f"  Models     : {', '.join(os.path.basename(c) for c in args.checkpoints)}")
    print(f"  Re-encodes : {len(items) * len(args.crf):,}\n")

    report = {
        "split": "val",
        "n": len(items),
        "n_real": n_real,
        "crf_levels": args.crf,
        "limit": args.limit,
        "seed": args.seed,
        "models": {},
    }

    for ckpt in args.checkpoints:
        name = os.path.basename(ckpt).split(".")[0]
        model, ck = load_model(ckpt)
        report["models"][name] = {
            "checkpoint": ckpt,
            "arch": ck.get("arch", {}),
            "threshold": ck.get("threshold"),
            "curve": {},
        }
        print(f"  {name}")
        for crf in args.crf:
            t0 = time.time()
            probs, labels, missing = [], [], 0
            for video, label in items:
                frames, pad = faces_at_crf(video, crf)
                if frames is None:
                    missing += 1
                    continue
                with torch.no_grad():
                    logit, _ = model(frames.unsqueeze(0).to(DEVICE), pad.unsqueeze(0).to(DEVICE))
                probs.append(float(torch.sigmoid(logit).item()))
                labels.append(label)
            probs, labels = np.array(probs), np.array(labels)
            ci = auc_ci(labels, probs, resamples=args.bootstrap, seed=DEFAULT_SEED)
            report["models"][name]["curve"][str(crf)] = {
                "auc": ci.point,
                "ci_low": ci.lo,
                "ci_high": ci.hi,
                "n_scored": int(len(labels)),
                "n_no_face": missing,
                "seconds": round(time.time() - t0, 1),
            }
            print(
                f"    CRF {crf:>2}  AUC {ci.format()}   "
                f"n={len(labels)}  no-face={missing}  [{time.time() - t0:.0f}s]"
            )

    # The claim is about the SLOPE: a robust model should lose less as CRF rises.
    print("\n  degradation from the mildest to the harshest CRF")
    lo, hi = str(min(args.crf)), str(max(args.crf))
    for name, m in report["models"].items():
        if lo in m["curve"] and hi in m["curve"]:
            drop = m["curve"][lo]["auc"] - m["curve"][hi]["auc"]
            m["degradation"] = {"from_crf": int(lo), "to_crf": int(hi), "auc_drop": drop}
            print(
                f"    {name:<10} CRF {lo} {m['curve'][lo]['auc']:.4f}  ->  "
                f"CRF {hi} {m['curve'][hi]['auc']:.4f}   drop {drop:+.4f}"
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
