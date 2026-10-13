#!/usr/bin/env python
"""Evaluation of a trained checkpoint.

Usage:
  python evaluate.py                          # best_model.pt on val
  python evaluate.py --checkpoint <ckpt.pt>   # explicit
  python evaluate.py --threshold 0.5          # overridden threshold
  python evaluate.py --checkpoint <ckpt.pt> --split test --final
                                              # THE final evaluation (PLAN D1)

The test split is read ONCE PER MODEL. --split test needs --final as well, and
every unlock is appended to results/TEST_ACCESS_LOG.md keyed by the checkpoint's
sha256, so renaming a file does not buy a second look. A test run defaults to the
threshold the checkpoint calibrated on validation, never the 0.5 fallback.

Metrics (report §7.3): accuracy, ROC-AUC, F1, per-class accuracy, confusion
matrix, bootstrap intervals. Saved to checkpoints/eval_<name>[_test]/.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.config import CONFIG, DEVICE, TRAIN_CONFIG
from src.dataset import DeepfakeDataset
from src.logger import section
from src.metrics import DEFAULT_RESAMPLES, DEFAULT_SEED, summarise
from src.model import DeepfakeDetector, apply_checkpoint_arch, arch_label, count_params
from src.test_access import (
    LOG_PATH,
    check_access,
    record_access,
    resolve_test_threshold,
    sha256_file,
)


def main():
    """Evaluate a checkpoint on the FF++ validation (or locked test) split."""
    parser = argparse.ArgumentParser(description="Evaluate DeepfakeDetector v7.0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_RESAMPLES,
        help="bootstrap resamples for the confidence intervals; 0 disables them (PLAN D2)",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="which split to evaluate. 'test' is the final, once-per-model "
        "evaluation and requires --final (PLAN D1)",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="confirm a final test-split evaluation. Required with --split test",
    )
    parser.add_argument(
        "--allow-repeat",
        action="store_true",
        help="permit a second test-split read of a checkpoint already in "
        "results/TEST_ACCESS_LOG.md; it is recorded as a repeat",
    )
    parser.add_argument(
        "--repeat-reason", default="", help="why a repeat read is justified; goes in the access log"
    )
    args = parser.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = os.path.join(CONFIG["CHECKPOINT_DIR"], "best_model.pt")
    if not os.path.exists(ckpt):
        print(f"ERROR: checkpoint not found — {ckpt}")
        sys.exit(1)

    # ── Test-split gate (PLAN D1) ────────────────────────────────────────────
    # Two separate locks: --final says the operator meant it, and the access log says this
    # checkpoint has not already had its one look.
    is_final = args.split == "test"
    ckpt_sha = None
    if is_final:
        if not args.final:
            print(
                "ERROR: --split test is the final evaluation and needs --final as well.\n"
                "       It may be run once per model and every run is appended to\n"
                "       results/TEST_ACCESS_LOG.md. Use --split val for anything iterative."
            )
            sys.exit(1)
        ckpt_sha = sha256_file(ckpt)
        try:
            access_kind = check_access(ckpt_sha, allow_repeat=args.allow_repeat)
        except PermissionError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        if access_kind == "repeat" and not args.repeat_reason:
            print('ERROR: --allow-repeat also needs --repeat-reason "..." for the log.')
            sys.exit(1)

    c = torch.load(ckpt, map_location=DEVICE, weights_only=True)

    if is_final:
        try:
            threshold, thr_source = resolve_test_threshold(
                c, args.threshold if args.threshold != 0.5 else None
            )
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        threshold, thr_source = args.threshold, "command line / default"

    section("FINAL TEST EVALUATION" if is_final else "EVALUATION")
    print(f"  Checkpoint : {ckpt}")
    print(f"  Split      : {args.split.upper()}" + ("   ← once per model" if is_final else ""))
    print(f"  Threshold  : {threshold:.4f}  ({thr_source})")
    print(f"  Device     : {DEVICE}")
    if is_final:
        print(f"  Access     : {access_kind}  (sha {ckpt_sha[:16]})")

    model = DeepfakeDetector().to(DEVICE)
    model.load_state_dict(c["model_state"])
    model.eval()
    print(f"  Mode       : {apply_checkpoint_arch(c)}")
    print(f"  Params     : {count_params(model):,}")

    test_ds = DeepfakeDataset(
        CONFIG["DATASET_ROOT"], split=args.split, augment=False, allow_test=is_final
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=TRAIN_CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    section("RUNNING INFERENCE")
    y_true, y_probs = [], []
    t0 = time.time()
    with torch.no_grad():
        for frames, labels, pad_mask in test_loader:
            frames = frames.to(DEVICE)
            logits, _ = model(frames, pad_mask.to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            y_probs.append(probs)
            y_true.append(labels.numpy().ravel())

    y_true = np.concatenate(y_true)
    y_probs = np.concatenate(y_probs)
    y_pred = (y_probs >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_probs)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    el = time.time() - t0
    n_fake = int(y_true.sum())
    n_real = int(len(y_true) - n_fake)

    # Per-class accuracy
    acc_real = (cm[0, 0]) / max(n_real, 1)
    acc_fake = (cm[1, 1]) / max(n_fake, 1)
    # With ~1:6 real:fake, plain accuracy is dominated by fakes ("all FAKE"
    # already scores ~0.857), so report the class-balanced figure too.
    balanced_acc = (acc_real + acc_fake) / 2
    majority_acc = max(n_real, n_fake) / max(len(y_true), 1)

    # Per-manipulation breakdown (loader is unshuffled, so order = samples).
    classes = np.array([os.path.basename(os.path.dirname(p)) for p, _ in test_ds.samples])
    per_class = {}
    for cls in sorted(set(classes), key=lambda c: (c != "original", c)):
        mask = classes == cls
        entry = {"n": int(mask.sum()), "acc": float((y_pred[mask] == y_true[mask]).mean())}
        if cls != "original" and n_real:
            both = mask | (classes == "original")
            entry["auc_vs_real"] = float(roc_auc_score(y_true[both], y_probs[both]))
        per_class[cls] = entry

    section("RESULTS")
    print(f"\n  Samples  : {len(y_true):,} (REAL {n_real:,} / FAKE {n_fake:,})  [{el:.0f}s]")
    print(f"  Accuracy : {acc:.4f}     (REAL {acc_real:.4f} | FAKE {acc_fake:.4f})")
    print(f"  Balanced : {balanced_acc:.4f}     (majority-class baseline {majority_acc:.4f})")
    print(f"  AUC      : {auc:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  Precision: REAL {p[0]:.4f} | FAKE {p[1]:.4f}")
    print(f"  Recall   : REAL {r[0]:.4f} | FAKE {r[1]:.4f}")
    print(f"  Confusion: TN {tn} FP {fp} | FN {fn} TP {tp}\n")
    print(f"  {'Class':<18} {'n':>5} {'acc':>7} {'AUC vs real':>12}")
    for cls, e in per_class.items():
        auc_txt = f"{e['auc_vs_real']:.4f}" if "auc_vs_real" in e else "—"
        print(f"  {cls:<18} {e['n']:>5} {e['acc']:>7.4f} {auc_txt:>12}")
    print()

    name = os.path.splitext(os.path.basename(ckpt))[0]
    out_dir = os.path.join(
        os.path.dirname(ckpt), f"eval_{name}_test" if is_final else f"eval_{name}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Confusion matrix plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["REAL", "FAKE"])
        ax.set_yticklabels(["REAL", "FAKE"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"AUC {auc:.3f} | Acc {acc:.3f}")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        plt.colorbar(im)
        fig.tight_layout()
        cm_path = os.path.join(out_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"  Confusion matrix → {cm_path}")
    except Exception as e:
        print(f"  (plot skipped: {e})")

    metrics = {
        "checkpoint": ckpt,
        "threshold": threshold,
        "n": len(y_true),
        "n_real": n_real,
        "n_fake": n_fake,
        "split_file": os.path.basename(test_ds.split_file),
        "split": args.split,
        "mode": arch_label(),
        "accuracy": acc,
        "acc_real": acc_real,
        "acc_fake": acc_fake,
        "balanced_accuracy": balanced_acc,
        "majority_class_accuracy": majority_acc,
        "auc": auc,
        "f1": f1,
        "per_class": per_class,
        "precision": {"real": float(p[0]), "fake": float(p[1])},
        "recall": {"real": float(r[0]), "fake": float(r[1])},
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "inference_time_s": el,
    }

    if args.bootstrap:
        metrics["intervals"] = summarise(
            y_true, y_probs, threshold=threshold, resamples=args.bootstrap, seed=args.bootstrap_seed
        )
        iv = metrics["intervals"]
        print(f"  Bootstrap {args.bootstrap} resamples, seed {args.bootstrap_seed}:")
        for key in ("auc", "accuracy", "balanced_accuracy", "f1"):
            e = iv[key]
            print(f"    {key:<18} {e['point']:.4f} [{e['ci_low']:.4f}, {e['ci_high']:.4f}]")
        print()
    # Raw scores, so a later paired comparison uses the same videos in the same order
    # (src/metrics.py::compare, PLAN D2). Without these, every cross-model claim would have to
    # re-run inference and could not be reproduced.
    np.save(os.path.join(out_dir, "probs.npy"), y_probs)
    np.save(os.path.join(out_dir, "labels.npy"), y_true)

    mp = os.path.join(out_dir, "metrics.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics → {mp}")

    # The access log is written last and only after the numbers exist, so a crashed run does not
    # spend this model's single look at the test split (PLAN D1).
    if is_final:
        metrics["test_access"] = {
            "sha256": ckpt_sha,
            "kind": access_kind,
            "reason": args.repeat_reason or None,
        }
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        record_access(
            model=name,
            checkpoint_path=ckpt,
            checkpoint_sha=ckpt_sha,
            git_commit=c.get("git_commit"),
            split_file=test_ds.split_file,
            threshold=threshold,
            n=len(y_true),
            auc=auc,
            balanced_acc=balanced_acc,
            kind=access_kind,
            reason=args.repeat_reason,
        )
        print(f"  Access log → {LOG_PATH}  (recorded as '{access_kind}')")
        print(
            "\n  This model has now used its one look at the test split. Report this number as\n"
            "  final; do not tune anything against it."
        )

    return 0 if auc >= 0.85 else 2


if __name__ == "__main__":
    sys.exit(main())
