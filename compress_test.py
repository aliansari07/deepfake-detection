#!/usr/bin/env python
"""Compression-robustness test.

Re-compresses the FF++ holdout faces at escalating JPEG quality
levels (simulating social-media pipelines — the exact distortion
the model was trained against) and measures metric degradation.

  clean : no transform                     (≈ original H.264)
  q85   : JPEG quality 85  (≈ CRF 18, light)
  q65   : JPEG quality 65  (≈ CRF 25)
  q45   : JPEG quality 45  (≈ CRF 30, social-media-grade)
  q25   : JPEG quality 25  (≈ CRF 35+, heavy)
  aug   : stochastic CompressionAwareAugmentation (eval-time)

Usage:
  python compress_test.py [--checkpoint checkpoints/T4-20.best.pt]
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from src.config import CONFIG, DEVICE, TRAIN_CONFIG
from src.dataset import DeepfakeDataset
from src.logger import log, section
from src.model import DeepfakeDetector, apply_checkpoint_arch

LEVELS = [("clean", None), ("q85", 85), ("q65", 65), ("q45", 45), ("q25", 25), ("aug", "aug")]


def jpeg_transform(t, quality):
    """JPEG-encode and decode one ``(C, H, W)`` float RGB frame at ``quality``."""
    bgr = (t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return t
    rgb = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def transform_batch(frames, level, augmentor):
    """frames: (B, N, 3, 224, 224) float [0,1] → transformed, on CPU."""
    if level is None:
        return frames
    if level == "aug":
        out = []
        for i in range(frames.shape[0]):
            aug, _ = augmentor.augment_batch(frames[i])
            out.append(aug)
        return torch.stack(out)
    out = []
    for i in range(frames.shape[0]):
        f2 = torch.stack([jpeg_transform(frames[i, j], level) for j in range(frames.shape[1])])
        out.append(f2)
    return torch.stack(out)


def run_level(model, loader, name, level, augmentor):
    """Evaluate ``model`` on ``loader`` after applying one compression level."""
    y_true, y_probs = [], []
    t0 = time.time()
    with torch.no_grad():
        for frames, labels, pad_mask in loader:
            frames = transform_batch(frames, level, augmentor)
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
    print(
        f"  {name:<6} Acc {acc:.4f}  AUC {auc:.4f}  F1 {f1:.4f}  "
        f"TN {cm[0]} FP {cm[1]} FN {cm[2]} TP {cm[3]}  [{el:.0f}s]"
    )
    return {
        "level": name,
        "n": len(y_true),
        "accuracy": float(acc),
        "auc": float(auc),
        "f1": float(f1),
        "confusion": {"tn": int(cm[0]), "fp": int(cm[1]), "fn": int(cm[2]), "tp": int(cm[3])},
        "inference_time_s": el,
    }


def main():
    """Run the compression-robustness test across the requested levels."""
    parser = argparse.ArgumentParser(description="Compression-robustness test")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--levels", nargs="+", default=[l for l, _ in LEVELS], choices=[l for l, _ in LEVELS]
    )
    args = parser.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = os.path.join(CONFIG["CHECKPOINT_DIR"], "best_model.pt")
    if not os.path.exists(ckpt):
        print(f"ERROR: checkpoint not found — {ckpt}")
        sys.exit(1)

    section("COMPRESSION-ROBUSTNESS TEST [holdout, n≈1,391]")
    print(f"  Checkpoint : {ckpt}")
    print(f"  Device     : {DEVICE}")
    print(f"  Levels     : {' '.join(args.levels)}")

    model = DeepfakeDetector().to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state["model_state"])
    model.eval()
    print(f"  Mode       : {apply_checkpoint_arch(state)}")

    test_ds = DeepfakeDataset(CONFIG["DATASET_ROOT"], split="val", augment=False)
    loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=TRAIN_CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    from src.augmentation import CompressionAwareAugmentation

    augmentor = CompressionAwareAugmentation(seed=42)

    section("RESULTS")
    results = {}
    for name, level in LEVELS:
        if name not in args.levels:
            continue
        results[name] = run_level(model, loader, name, level, augmentor)

    clean = results["clean"]
    print("\n  Δ vs clean:")
    for name in results:
        if name == "clean":
            continue
        print(
            f"  {name:<6} Δacc {results[name]['accuracy'] - clean['accuracy']:+.4f}  "
            f"Δauc {results[name]['auc'] - clean['auc']:+.4f}  "
            f"Δf1  {results[name]['f1'] - clean['f1']:+.4f}"
        )

    worst = min(
        (r for k, r in results.items() if k != "clean"), key=lambda r: r["accuracy"], default=clean
    )
    worst_name = [k for k, r in results.items() if r is worst][0] if worst is not clean else "clean"
    print(
        f"\n  Worst level: {worst_name} — Acc {worst['accuracy']:.4f} "
        f"(drop {clean['accuracy'] - worst['accuracy']:.4f})"
    )
    log("COMPRESS", "Same holdout as evaluate.py — clean row is the reference.")

    out_dir = os.path.join(os.path.dirname(ckpt), "eval_compress")
    os.makedirs(out_dir, exist_ok=True)
    mp = os.path.join(out_dir, "metrics.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": ckpt, "results": results, "worst": worst_name}, f, indent=2)
    print(f"\n  Metrics → {mp}")


if __name__ == "__main__":
    main()
