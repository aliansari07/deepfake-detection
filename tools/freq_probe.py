"""R1 — how much real/fake signal is actually in the frequency domain?

`audit-fyp2.md` section 14.3 measured the frequency branch as contributing nothing, and offered four
explanations. Before spending GPU time on R3-R5 we need to know whether the information is
there at all: no architecture can recover a signal the C23 encoder already destroyed.

This fits simple linear probes to hand-built frequency features and reports validation AUC. A
linear probe is a floor, not a ceiling -- it says what is trivially extractable. But if even the
richest feature set sits at chance, the branch cannot be rescued by changing the model, and the
only remaining route is raw or c0 source video.

Three feature sets, in increasing power:

  radial     the radial average of log|FFT|. This is what section 14.3 used, and it averages over
             angle -- which smears out exactly the discrete spectral peaks that upsampling
             artifacts produce. Kept so the earlier number stays reproducible.
  grid       the full 2D log|FFT| downsampled to an 8x8 grid of band means, so orientation and
             isolated peaks survive.
  residual   statistics of the high-frequency spatial residual (frame minus its Gaussian blur):
             energy, kurtosis and the horizontal/vertical co-occurrence that SRM-style forensics
             features are built from.

Usage:
    python tools/freq_probe.py                      # all three, 400 videos per class
    python tools/freq_probe.py --per-class 150      # quicker
    python tools/freq_probe.py --features grid
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CONFIG  # noqa: E402
from src.dataset import DeepfakeDataset  # noqa: E402
from src.face_cache import load_frames  # noqa: E402

NBINS = 64
GRID = 8


def _spectrum(gray):
    return torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(gray))) + 1e-8).numpy()


def feat_radial(gray):
    """Radial average of log|FFT| -- the section 14.3 feature, kept for continuity."""
    s = _spectrum(gray)
    h, w = s.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    idx = np.clip((r / r.max() * (NBINS - 1)).astype(int), 0, NBINS - 1)
    return np.array([s[idx == b].mean() for b in range(NBINS)])


def feat_grid(gray):
    """Full 2D spectrum as GRID x GRID band means -- keeps orientation and isolated peaks."""
    s = _spectrum(gray)
    h, w = s.shape
    ys = np.array_split(np.arange(h), GRID)
    xs = np.array_split(np.arange(w), GRID)
    return np.array([s[np.ix_(yb, xb)].mean() for yb in ys for xb in xs])


def feat_residual(gray):
    """Statistics of the high-frequency spatial residual (frame minus its blur).

    This is the input R5 proposes to feed the branch instead of a spectrum, so its probe score is
    a direct preview of whether R5 is worth building.
    """
    g = gray.numpy().astype(np.float32)
    res = g - cv2.GaussianBlur(g, (5, 5), 1.0)
    dx, dy = np.diff(res, axis=1).ravel(), np.diff(res, axis=0).ravel()
    out = [
        res.std(),
        np.abs(res).mean(),
        float(((res - res.mean()) ** 4).mean() / max(res.var() ** 2, 1e-12)),
        dx.std(),
        dy.std(),
        float(np.corrcoef(dx[:-1], dx[1:])[0, 1]) if dx.size > 2 else 0.0,
    ]
    for q in (50, 75, 90, 95, 99):
        out.append(float(np.percentile(np.abs(res), q)))
    hist, _ = np.histogram(np.clip(res, -16, 16), bins=16, range=(-16, 16), density=True)
    return np.nan_to_num(np.concatenate([out, hist]))


FEATURES = {"radial": feat_radial, "grid": feat_grid, "residual": feat_residual}


def video_features(path, fn, n_frames):
    """Average the feature function ``fn`` over up to ``n_frames`` frames of a video."""
    frames, _ = load_frames(path)
    k = min(n_frames, frames.shape[0])
    if k == 0:
        return None
    step = max(1, frames.shape[0] // k)
    vecs = []
    for i in range(0, step * k, step):
        f = frames[i]
        vecs.append(fn(0.299 * f[0] + 0.587 * f[1] + 0.114 * f[2]))
    return np.mean(vecs, axis=0)


def build(split, fn, per_class, n_frames, seed=0):
    """Build a balanced feature matrix and labels for one split."""
    ds = DeepfakeDataset(CONFIG["DATASET_ROOT"], split=split)
    rng = np.random.default_rng(seed)
    by = {0: [], 1: []}
    for p, l in ds.samples:
        by[l].append(p)
    X, y, groups = [], [], []
    for label, paths in by.items():
        for i in rng.choice(len(paths), min(per_class, len(paths)), replace=False):
            v = video_features(paths[i], fn, n_frames)
            if v is not None:
                X.append(v)
                y.append(label)
                groups.append(os.path.basename(os.path.dirname(paths[i])))
    return np.array(X), np.array(y), np.array(groups)


def main():
    """Fit linear probes on frequency features and report validation AUC."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--features", choices=list(FEATURES) + ["all"], default="all")
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--out", default="results/freq_probe.json")
    args = ap.parse_args()

    names = list(FEATURES) if args.features == "all" else [args.features]
    report = {"per_class": args.per_class, "frames": args.frames, "sets": {}}

    for name in names:
        fn = FEATURES[name]
        print(f"\n=== {name} " + "=" * (58 - len(name)))
        Xtr, ytr, _ = build("train", fn, args.per_class, args.frames)
        Xva, yva, gva = build("val", fn, args.per_class, args.frames)
        print(f"  train {Xtr.shape}  val {Xva.shape}")

        clf = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")
        )
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xva)[:, 1]
        overall = float(roc_auc_score(yva, p))
        print(f"  overall val AUC  {overall:.4f}   (0.50 = carries nothing)")

        per_method = {}
        real = yva == 0
        for g in sorted(set(gva[yva == 1])):
            m = real | (gva == g)
            per_method[g] = float(roc_auc_score(yva[m], p[m]))
            print(f"    {g:<18} {per_method[g]:.4f}   (n={int((gva == g).sum())})")
        report["sets"][name] = {
            "overall_auc": overall,
            "per_method": per_method,
            "n_features": int(Xtr.shape[1]),
        }

    best = max(report["sets"], key=lambda k: report["sets"][k]["overall_auc"])
    print(f"\nbest feature set: {best} at AUC {report['sets'][best]['overall_auc']:.4f}")
    print("A spatial CNN reaches 0.91-0.94 on these same videos (audit-fyp2 section 14.1).")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
