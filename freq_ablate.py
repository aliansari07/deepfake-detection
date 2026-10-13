#!/usr/bin/env python
"""Does the frequency branch contribute?.

Ablates the gated fusion ON THE SAME checkpoint (T4-20.best.pt)
over the identical FF++ holdout used by evaluate.py:

  full    : learned gate  →  F = g·s + (1−g)·f      (baseline)
  spatial : gate forced 1 →  F = s                  (freq removed)
  freq    : gate forced 0 →  F = f                  (spatial removed)

If full ≈ spatial, the frequency branch adds nothing and the
gate is dead weight; if full > spatial, it contributes.

Usage:
  python freq_ablate.py [--checkpoint checkpoints/T4-20.best.pt]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from src.config import CONFIG, DEVICE, TRAIN_CONFIG
from src.dataset import DeepfakeDataset
from src.logger import log, section
from src.model import DeepfakeDetector, apply_checkpoint_arch, force_gate

MODES = ["full", "spatial", "freq"]


def run_mode(model, loader, mode):
    """Evaluate ``model`` on ``loader`` with the gate forced into ``mode``."""
    y_true, y_probs = [], []
    t0 = time.time()
    with torch.no_grad():
        for frames, labels, pad_mask in loader:
            frames = frames.to(DEVICE)
            logits, _ = model(frames, pad_mask.to(DEVICE))
            y_probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
            y_true.append(labels.numpy().ravel())
    y_true = np.concatenate(y_true)
    y_probs = np.concatenate(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)
    el = time.time() - t0

    auc = roc_auc_score(y_true, y_probs)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).ravel()
    n = len(y_true)
    n_fake = int(y_true.sum())
    acc_real = 1.0 - y_pred[y_true == 0].mean()
    acc_fake = y_pred[y_true == 1].mean()
    print(
        f"  {mode:<7} Acc {acc:.4f}  AUC {auc:.4f}  F1 {f1:.4f}  "
        f"[REAL {acc_real:.4f} | FAKE {acc_fake:.4f}]  "
        f"TN {cm[0]} FP {cm[1]} FN {cm[2]} TP {cm[3]}  [{el:.0f}s]"
    )
    return {
        "mode": mode,
        "n": n,
        "n_real": n - n_fake,
        "n_fake": n_fake,
        "accuracy": float(acc),
        "auc": float(auc),
        "f1": float(f1),
        "acc_real": float(acc_real),
        "acc_fake": float(acc_fake),
        "confusion": {"tn": int(cm[0]), "fp": int(cm[1]), "fn": int(cm[2]), "tp": int(cm[3])},
        "inference_time_s": el,
    }


def main():
    """Run the frequency-branch ablation for every requested gate mode."""
    parser = argparse.ArgumentParser(description="Frequency-branch ablation")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--modes", nargs="+", default=MODES, choices=MODES, help=f"default: {' '.join(MODES)}"
    )
    args = parser.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = os.path.join(CONFIG["CHECKPOINT_DIR"], "best_model.pt")
    if not os.path.exists(ckpt):
        print(f"ERROR: checkpoint not found — {ckpt}")
        sys.exit(1)

    section("FREQUENCY-BRANCH ABLATION [SAME CHECKPOINT]")
    print(f"  Checkpoint : {ckpt}")
    print(f"  Device     : {DEVICE}")
    print(f"  Modes      : {' '.join(args.modes)}")
    print("  Split      : val (FF++ holdout, same as evaluate.py)")

    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    print(f"  Mode       : {apply_checkpoint_arch(state)}")
    if CONFIG["BASELINE_MODE"]:
        print(
            "ERROR: baseline checkpoint has no frequency branch / gate — "
            "forcing the gate is meaningless."
        )
        sys.exit(1)

    test_ds = DeepfakeDataset(CONFIG["DATASET_ROOT"], split="val", augment=False)
    loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=TRAIN_CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    section("RESULTS (Δ vs full)")
    results = {}
    for mode in args.modes:
        m = DeepfakeDetector().to(DEVICE)
        m.load_state_dict(state["model_state"])
        m.eval()
        if mode != "full":
            m = force_gate(m, 1.0 if mode == "spatial" else 0.0)
        results[mode] = run_mode(m, loader, mode)

    full = results["full"]
    print("\n  Δ from full model:")
    for mode in args.modes:
        if mode == "full":
            continue
        d_acc = results[mode]["accuracy"] - full["accuracy"]
        d_auc = results[mode]["auc"] - full["auc"]
        d_f1 = results[mode]["f1"] - full["f1"]
        print(f"  {mode:<7} Δacc {d_acc:+.4f}  Δauc {d_auc:+.4f}  Δf1 {d_f1:+.4f}")

    verdict = (
        "FREQ BRANCH CONTRIBUTES ✓"
        if full["auc"] > results.get("spatial", full)["auc"] + 1e-6
        else "FREQ BRANCH ≈ NO-OP — spatial alone matches"
    )
    print(f"\n  {verdict}")
    log(
        "ABLATE",
        "Gate forced constant on identical weights — all other state identical to full model.",
    )

    out_dir = os.path.join(os.path.dirname(ckpt), "eval_freq_ablation")
    os.makedirs(out_dir, exist_ok=True)
    mp = os.path.join(out_dir, "metrics.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": ckpt, "results": results}, f, indent=2)
    print(f"\n  Metrics → {mp}")


if __name__ == "__main__":
    main()
