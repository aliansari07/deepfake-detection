"""Paired comparison of trained models (PLAN task D2).

This replaces the throwaway scripts that produced the interval tables in `audit-fyp2.md` sections
14 and 15. Those numbers could not be regenerated, which meant no claim resting on them was
defensible. Every cross-model claim in the thesis should come out of this tool.

Two modes:

  --checkpoints A.pt B.pt ...      score each checkpoint on the split, then compare
  --runs dir1 dir2 ...             reuse probs.npy/labels.npy written by evaluate.py

The first model on the command line is the reference; every other is compared against it. Each
model is reported at its own validation-calibrated Youden threshold, because a model whose
probabilities are crushed against 1.0 (M1 needs 0.944) is not comparable with one that is
calibrated (A2 needs 0.464) at any shared cut.

Both claim rules are printed for every pair, because decision D6 is open: the paired rule (the
difference interval excludes zero) and the stricter marginal rule PLAN.md currently words as "the
confidence intervals do not overlap". See `audit-fyp2.md` section 14.4.

Usage:
    python tools/compare_runs.py --checkpoints checkpoints/A2-20.best.pt \\
        checkpoints/M1-20.best.pt checkpoints/A1-20.best.pt --out results/comparison.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CONFIG, DEVICE  # noqa: E402
from src.dataset import DeepfakeDataset  # noqa: E402
from src.metrics import DEFAULT_RESAMPLES, DEFAULT_SEED, compare, summarise  # noqa: E402
from src.model import DeepfakeDetector, apply_checkpoint_arch, arch_label  # noqa: E402


def youden(y_true, probs):
    """Threshold that maximises Youden's J (TPR - FPR); 0.5 if undefined."""
    fpr, tpr, thr = roc_curve(y_true, probs)
    t = float(thr[int(np.argmax(tpr - fpr))])
    return t if np.isfinite(t) else 0.5


def score_checkpoint(path, dl):
    """Load a checkpoint and return its probabilities on ``dl``."""
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    apply_checkpoint_arch(ck)
    model = DeepfakeDetector().to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for frames, y, pad in dl:
            logit, _ = model(frames.to(DEVICE), pad.to(DEVICE))
            probs.append(torch.sigmoid(logit).float().cpu().numpy().ravel())
            labels.append(y.numpy().ravel())
    return np.concatenate(probs), np.concatenate(labels), arch_label(), ck.get("epoch")


def main():
    """Compare two or more runs with bootstrap and paired confidence intervals."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--checkpoints", nargs="+", help="checkpoint paths; the first is the reference"
    )
    src.add_argument(
        "--runs", nargs="+", help="evaluate.py output dirs holding probs.npy/labels.npy"
    )
    ap.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
        help="the test split is reachable only through evaluate.py --final (task D1)",
    )
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_RESAMPLES)
    ap.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default="results/comparison.json")
    args = ap.parse_args()

    models, labels, groups = {}, None, None

    if args.checkpoints:
        ds = DeepfakeDataset(CONFIG["DATASET_ROOT"], split=args.split)
        dl = DataLoader(ds, batch_size=8, num_workers=0)
        groups = np.array([os.path.basename(os.path.dirname(p)) for p, _ in ds.samples])
        for path in args.checkpoints:
            name = os.path.basename(path).split(".")[0]
            probs, labels, arch, epoch = score_checkpoint(path, dl)
            models[name] = {"probs": probs, "arch": arch, "epoch": epoch, "source": path}
            print(f"  scored {name:<12} {arch}  (epoch {epoch})")
    else:
        for d in args.runs:
            name = os.path.basename(os.path.normpath(d))
            probs = np.load(os.path.join(d, "probs.npy"))
            labels = np.load(os.path.join(d, "labels.npy"))
            entry = {"probs": probs, "arch": "(from saved run)", "epoch": None, "source": d}
            # Reuse the threshold the run itself used. Re-deriving a Youden point from the data
            # being scored fits the operating point ON the evaluation set — harmless on val, but
            # on the test split it reports a number tuned on the split it claims to measure.
            mp = os.path.join(d, "metrics.json")
            if os.path.exists(mp):
                with open(mp, encoding="utf-8") as fh:
                    saved = json.load(fh)
                if "threshold" in saved:
                    entry["threshold"] = float(saved["threshold"])
                    entry["threshold_source"] = f"as evaluated ({saved.get('split', '?')} run)"
                entry["split"] = saved.get("split")
                entry["arch"] = saved.get("mode", entry["arch"])
            note = (
                f"  thr {entry['threshold']:.4f} ({entry['threshold_source']})"
                if "threshold" in entry
                else "  no threshold recorded by that run — one will be fitted"
            )
            print(f"  loaded {name:<28} n={probs.size}{note}")
            models[name] = entry

    names = list(models)
    fitted_here = []
    for n, m in models.items():
        if "threshold" not in m:
            m["threshold"] = youden(labels, m["probs"])
            m["threshold_source"] = "FITTED ON THIS DATA"
            fitted_here.append(n)
    if fitted_here:
        print("\n  WARNING: no stored threshold for " + ", ".join(fitted_here) + ".")
        print(
            "  A Youden point was fitted on the very data being scored. That is acceptable on\n"
            "  val, but on the test split it reports an operating point tuned on the test set.\n"
            "  Quote AUC (threshold-free) from those rows, not accuracy."
        )

    n_fake = int(labels.sum())
    split_label = (
        args.split if args.checkpoints else (models[names[0]].get("split") or "unknown (saved run)")
    )
    print(
        f"\n  split: {split_label}, n={labels.size} ({labels.size - n_fake} real / {n_fake} fake)"
    )

    print(f"\n{'':<24}" + "".join(f"{n:>16}" for n in names))
    print(
        f"{'operating threshold':<24}" + "".join(f"{models[n]['threshold']:>16.4f}" for n in names)
    )
    for label, fn in (
        ("AUC", lambda n: roc_auc_score(labels, models[n]["probs"])),
        (
            "accuracy @ own thr",
            lambda n: accuracy_score(
                labels, (models[n]["probs"] > models[n]["threshold"]).astype(int)
            ),
        ),
        (
            "balanced acc @ own thr",
            lambda n: balanced_accuracy_score(
                labels, (models[n]["probs"] > models[n]["threshold"]).astype(int)
            ),
        ),
        (
            "REAL recall",
            lambda n: (
                (models[n]["probs"] > models[n]["threshold"]).astype(int)[labels == 0] == 0
            ).mean(),
        ),
        (
            "FAKE recall",
            lambda n: (
                (models[n]["probs"] > models[n]["threshold"]).astype(int)[labels == 1] == 1
            ).mean(),
        ),
    ):
        print(f"{label:<24}" + "".join(f"{fn(n):>16.4f}" for n in names))

    if groups is not None:
        print("\nper-manipulation accuracy at each model's operating threshold")
        print(f"{'':<24}" + "".join(f"{n:>16}" for n in names))
        for g in sorted(set(groups)):
            m = groups == g
            want = 0 if g == "original" else 1
            print(
                f"{g:<24}"
                + "".join(
                    f"{((models[n]['probs'] > models[n]['threshold']).astype(int)[m] == want).mean():>16.4f}"  # noqa: E501
                    for n in names
                )
            )

    report = {
        "split": split_label,
        "n": int(labels.size),
        "bootstrap": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "models": {},
        "comparisons": [],
    }

    print(f"\nbootstrap intervals ({args.bootstrap} resamples, seed {args.bootstrap_seed})")
    for n in names:
        m = models[n]
        s = summarise(
            labels,
            m["probs"],
            threshold=m["threshold"],
            resamples=args.bootstrap,
            seed=args.bootstrap_seed,
        )
        report["models"][n] = {
            "arch": m["arch"],
            "epoch": m["epoch"],
            "source": m["source"],
            "threshold": m["threshold"],
            "threshold_source": m.get("threshold_source", "fitted"),
            **s,
        }
        a = s["auc"]
        print(
            f"  {n:<14} AUC {a['point']:.4f} [{a['ci_low']:.4f}, {a['ci_high']:.4f}]   "
            f"bal {s['balanced_accuracy']['point']:.4f}"
        )

    ref = names[0]
    print(f"\npaired comparisons against {ref}")
    for n in names[1:]:
        c = compare(
            labels,
            models[n]["probs"],
            models[ref]["probs"],
            name_a=n,
            name_b=ref,
            resamples=args.bootstrap,
            seed=args.bootstrap_seed,
        )
        report["comparisons"].append(c.as_dict())
        print("\n" + c.verdict())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
