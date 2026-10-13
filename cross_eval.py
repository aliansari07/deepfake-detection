#!/usr/bin/env python
"""Cross-dataset generalisation on Celeb-DF v2.

Trains the model on FaceForensics++ but tests it on an UNSEEN
dataset: Celeb-DF v2 (Celeb-real + YouTube-real vs Celeb-synthesis).
This is the ">75% accuracy on cross-dataset testing" criterion.

Protocols:
  official (default) : the 518 videos in List_of_testing_videos.txt
                       (178 real / 340 fake) — the standard Celeb-DF v2
                       benchmark; results are comparable with literature.
  sample             : custom seeded sample from ALL videos (FYP-1 used
                       890 real / 890 fake). Not comparable with literature.

Usage:
  python cross_eval.py                        # official 518-video test list
  python cross_eval.py --protocol sample      # balanced: all real (890) + 890 matched fakes
  python cross_eval.py --protocol sample --n-fake 300          # 890 real vs 300 fake
  python cross_eval.py --protocol sample --n-real 100 --n-fake 100   # smoke test
  python cross_eval.py --protocol sample --all  # every available video (no caps)
  python cross_eval.py --verbose              # verbose per-video logging
  Models trained on a cache-v2 split (splits/ffpp_official_v1.json) read RGB faces
  from datasets/Celeb-DF-v2/cache_v2 automatically; extract them first with
  python preextract.py --cache-version 2 --dataset celebdf [--celebdf-sample]

The run is RESUME-SAFE: every prediction is appended to
checkpoints/eval_celebdf/predictions.jsonl as it completes, so an
interrupted run skips already-processed videos on restart.

Outputs:
  (official protocol writes to eval_celebdf_official[_<tag>]/ instead)
  checkpoints/eval_celebdf/predictions.jsonl  (incremental)
  checkpoints/eval_celebdf/sample.json        (sample provenance)
  checkpoints/eval_celebdf/metrics.json
  checkpoints/eval_celebdf/confusion_matrix.png
"""

import argparse
import hashlib
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

from src.celebdf import CELEB_ROOT, build_official, build_sample, rel_key
from src.config import CONFIG, DEVICE
from src.logger import detail, section
from src.metrics import DEFAULT_RESAMPLES, DEFAULT_SEED, summarise
from src.model import DeepfakeDetector, apply_checkpoint_arch, force_gate
from src.preprocess import extract_faces


def load_progress(prog_path):
    """Load previously saved predictions; return {abs_path: record}."""
    records = {}
    if os.path.exists(prog_path):
        with open(prog_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records[rec["path"]] = rec
    return records


def append_record(prog_path, rec):
    """Append one JSON record to the resumable progress file."""
    with open(prog_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def get_frames(vpath, cache_dir, use_cache):
    """extract_faces, optionally cached as uint8 .pt keyed by md5(vpath)."""
    key = hashlib.md5(vpath.encode("utf-8")).hexdigest()
    fp = os.path.join(cache_dir, f"{key}.pt")
    if use_cache and os.path.exists(fp):
        frames = torch.load(fp, map_location="cpu", weights_only=True)
        return frames.float() / 255.0, {"cached": True}
    frames, meta = extract_faces(vpath)
    if frames is not None and use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save((frames * 255.0).round().clamp(0, 255).to(torch.uint8), fp)
    return frames, meta


def get_frames_v2(vpath):
    """Faces from cache v2 (RGB uint8, preextract.py --cache-version 2 --dataset celebdf).
    Returns (frames (T,3,224,224), pad_mask (T,)) or (None, None) when no face."""
    from src.face_cache import load_frames, to_sequence

    fp = os.path.join(CELEB_ROOT, "cache_v2", *rel_key(vpath).split("/")) + ".pt"
    if not os.path.exists(fp):
        print(
            f"ERROR: {fp} missing. Run: python preextract.py --cache-version 2 "
            f"--dataset celebdf [--celebdf-sample]"
        )
        sys.exit(1)
    frames, _ = load_frames(fp)
    if frames.shape[0] == 0:
        return None, None
    return to_sequence(frames, CONFIG["SAMPLE_COUNT"])


def checkpoint_cache_version(state):
    """Cache version of the split the checkpoint was trained on (1 if unknown)."""
    split_file = state.get("config", {}).get("CONFIG", {}).get("SPLIT_FILE")
    if split_file and os.path.isfile(split_file):
        with open(split_file, encoding="utf-8") as f:
            return int(json.load(f).get("cache_version", 1))
    return 1


def main():
    """Evaluate a checkpoint on the Celeb-DF v2 test list."""
    parser = argparse.ArgumentParser(description="Cross-dataset eval on Celeb-DF v2")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint to evaluate (default: checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--protocol",
        default="official",
        choices=["official", "sample"],
        help="official = 518-video test list (default); sample = custom seeded sample",
    )
    parser.add_argument(
        "--faces-cache-dir",
        default=None,
        help="Face cache directory (default: <out_dir>/faces_cache); "
        "point several runs at one dir to extract faces once",
    )
    parser.add_argument(
        "--n-real", type=int, default=None, help="Cap on real videos (default: all real available)"
    )
    parser.add_argument(
        "--n-fake", type=int, default=None, help="Cap on fake videos (default: match real count)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Evaluate every available video (no caps)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augment", action="store_true", help="Apply compression-aware augmentation at eval time"
    )
    parser.add_argument(
        "--force", action="store_true", help="Ignore and overwrite existing progress file"
    )
    parser.add_argument(
        "--tag", default="", help="Output tag → eval_celebdf_<tag> (keeps runs separate)"
    )
    parser.add_argument(
        "--gate",
        default="learned",
        choices=["learned", "spatial", "freq"],
        help="Force fusion gate constant for ablation",
    )
    parser.add_argument(
        "--cache-faces", action="store_true", help="Cache extracted faces as uint8 .pt for reuse"
    )
    parser.add_argument(
        "--face-cache-v2",
        action="store_true",
        help="Read RGB faces from datasets/Celeb-DF-v2/cache_v2 (automatic "
        "for checkpoints trained on a cache-v2 split)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_RESAMPLES,
        help="bootstrap resamples for the confidence intervals; 0 disables them (PLAN D2)",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--verbose", action="store_true", help="Log every video + face-extraction detail"
    )
    args = parser.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = os.path.join(CONFIG["CHECKPOINT_DIR"], "best_model.pt")
    if not os.path.exists(ckpt):
        print(f"ERROR: checkpoint not found — {ckpt}")
        sys.exit(1)

    # ---- Sampling (default: balanced — all real 890 vs 890 matched fakes) ----
    n_real = args.n_real
    n_fake = args.n_fake
    if args.all:
        n_real = n_fake = None
    elif not n_real and not n_fake:
        n_real, n_fake = None, 890
    elif n_real and not n_fake:
        n_fake = n_real

    if args.protocol == "official":
        videos, labels = build_official()
        n_real = n_fake = None
    else:
        videos, labels = build_sample(n_real, n_fake, args.seed, force_all=args.all)

    if not args.verbose:
        import src.logger as logger_mod

        logger_mod.mute(True)

    out_dir = os.path.join(
        os.path.dirname(ckpt),
        "eval_celebdf"
        + ("_official" if args.protocol == "official" else "")
        + (f"_{args.tag}" if args.tag else ""),
    )
    os.makedirs(out_dir, exist_ok=True)
    prog_path = os.path.join(out_dir, "predictions.jsonl")
    sample_path = os.path.join(out_dir, "sample.json")
    faces_cache = args.faces_cache_dir or os.path.join(out_dir, "faces_cache")

    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "Celeb-DF-v2",
                "protocol": args.protocol,
                "checkpoint": ckpt,
                "seed": args.seed,
                "gate": args.gate,
                "n_real_requested": n_real,
                "n_fake_requested": n_fake,
                "all": args.all,
                "sample": videos,
                "labels": labels,
            },
            f,
            indent=2,
        )

    section("CROSS-DATASET EVALUATION [Celeb-DF v2]")
    print(f"  Checkpoint : {ckpt}")
    print(f"  Protocol   : {args.protocol} ({len(videos)} videos)")
    print(f"  Device     : {DEVICE}")
    print(f"  Gate       : {args.gate}      ", end="")
    print(f"Augment : {'ON' if args.augment else 'OFF'}")
    print(
        f"  Face cache : {'ON' if args.cache_faces else 'OFF'}   "
        f"Verbose : {'ON' if args.verbose else 'OFF'}"
    )
    print(f"  Output dir : {out_dir}")

    model = DeepfakeDetector().to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state["model_state"])
    model.eval()
    print(f"  Mode       : {apply_checkpoint_arch(state)}")
    trained_v2 = checkpoint_cache_version(state) >= 2
    if args.face_cache_v2 and not trained_v2:
        print(
            "ERROR: --face-cache-v2 gives RGB faces, but this checkpoint was trained on "
            "the legacy BGR cache."
        )
        sys.exit(1)
    use_v2 = trained_v2
    print(
        f"  Faces      : {'cache v2 (RGB, masked padding)' if use_v2 else 'legacy extraction (BGR)'}"  # noqa: E501
    )
    if CONFIG["BASELINE_MODE"] and args.gate != "learned":
        print("ERROR: --gate needs a checkpoint with a frequency branch (not a baseline).")
        sys.exit(1)
    if args.gate == "spatial":
        model = force_gate(model, 1.0)
    elif args.gate == "freq":
        model = force_gate(model, 0.0)

    augmentor = None
    if args.augment:
        from src.augmentation import CompressionAwareAugmentation

        augmentor = CompressionAwareAugmentation(seed=args.seed)

    # ---- Resume: adopt previously saved predictions ----
    done = load_progress(prog_path) if not args.force else {}
    pending = []
    for vpath, label in zip(videos, labels):
        if vpath in done:
            continue
        pending.append((vpath, label))
    print(
        f"  Progress   : {len(done):,} already done / "
        f"{len(pending):,} to process / {len(videos):,} total"
    )

    section("RUNNING INFERENCE")
    index = {v: i for i, v in enumerate(videos)}
    no_face = [v for v, r in done.items() if r.get("no_face") and v in index]
    t0 = time.time()

    for i, (vpath, label) in enumerate(pending, 1):
        if use_v2:
            frames, pad_mask = get_frames_v2(vpath)
        else:
            frames, _ = get_frames(vpath, faces_cache, args.cache_faces)
            pad_mask = None
        rec = {"path": vpath, "label": int(label)}
        if frames is None:
            rec["no_face"] = True
            append_record(prog_path, rec)
            done[vpath] = rec
            continue
        if augmentor is not None:
            frames, _ = augmentor.augment_batch(frames)

        with torch.no_grad():
            logit, _ = model(frames.to(DEVICE), None if pad_mask is None else pad_mask.to(DEVICE))
        prob = torch.sigmoid(logit).item()
        rec["prob"] = prob
        append_record(prog_path, rec)
        done[vpath] = rec

        if args.verbose:
            detail(
                f"[{i}/{len(pending)}] {os.path.basename(vpath)[:48]:<48} "
                f"label={label} prob={prob:.3f}"
            )

        if i % 100 == 0 or i == len(pending):
            el = time.time() - t0
            print(f"  [{i}/{len(pending)}] processed — {el:.0f}s", flush=True)

    preds = [r for r in done.values() if "prob" in r and r["path"] in index]
    if not preds:
        print("ERROR: no videos produced predictions.")
        sys.exit(1)

    preds.sort(key=lambda r: index[r["path"]])
    y_true = np.array([r["label"] for r in preds])
    y_probs = np.array([r["prob"] for r in preds])
    y_pred = (y_probs >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_probs)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    el = time.time() - t0

    section("RESULTS [Celeb-DF v2]")
    print(
        f"\n  Samples    : {len(y_true):,} "
        f"(REAL {int((y_true == 0).sum()):,} / "
        f"FAKE {int((y_true == 1).sum()):,})  [{el:.0f}s]"
    )
    acc_real = (y_pred[y_true == 0] == 0).mean()
    acc_fake = (y_pred[y_true == 1] == 1).mean()
    print(f"  Accuracy   : {acc:.4f}     (REAL {acc_real:.4f} | FAKE {acc_fake:.4f})")
    print(f"  AUC        : {auc:.4f}")
    print(f"  F1         : {f1:.4f}")
    print(f"  Precision  : REAL {p[0]:.4f} | FAKE {p[1]:.4f}")
    print(f"  Recall     : REAL {r[0]:.4f} | FAKE {r[1]:.4f}")
    print(f"  Confusion  : TN {tn} FP {fp} | FN {fn} TP {tp}")
    if no_face:
        print(f"  No-face    : {len(no_face)} videos skipped")
    print(f"  Resumed    : {len(no_face)} previously-saved no-face entries")

    # Per-class breakdown
    section("PER-CLASS BREAKDOWN")
    for split, mask in (("REAL", y_true == 0), ("FAKE", y_true == 1)):
        acc_seg = (y_pred[mask] == y_true[mask]).mean()
        print(f"  {split:<5} n={int(mask.sum()):>4}  acc={acc_seg:.4f}")

    metrics = {
        "dataset": "Celeb-DF-v2",
        "protocol": args.protocol,
        "checkpoint": ckpt,
        "seed": args.seed,
        "gate": args.gate,
        "augment": args.augment,
        "sample_requested": {
            "n_real": n_real,
            "n_fake": n_fake,
            "all": args.all,
        },
        "n": int(len(y_true)),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
        "n_no_face": len(no_face),
        "n_resumed": len(no_face),
        "threshold": 0.5,
        "split": "celebdf-official",
        "accuracy": float(acc),
        "acc_real": float(acc_real),
        "acc_fake": float(acc_fake),
        "auc": float(auc),
        "f1": float(f1),
        "precision": {"real": float(p[0]), "fake": float(p[1])},
        "recall": {"real": float(r[0]), "fake": float(r[1])},
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "inference_time_s": el,
    }

    # Confidence intervals (PLAN D2). Celeb-DF is the cross-dataset claim S2, so an interval is
    # not optional here: 73.2% against a 75% target is only meaningful with its uncertainty.
    if getattr(args, "bootstrap", DEFAULT_RESAMPLES):
        metrics["intervals"] = summarise(
            y_true, y_probs, threshold=0.5, resamples=args.bootstrap, seed=args.bootstrap_seed
        )
        iv = metrics["intervals"]
        print(f"\n  Bootstrap {args.bootstrap} resamples, seed {args.bootstrap_seed}:")
        for key in ("auc", "accuracy", "balanced_accuracy", "f1"):
            e = iv[key]
            print(f"    {key:<18} {e['point']:.4f} [{e['ci_low']:.4f}, {e['ci_high']:.4f}]")

    np.save(os.path.join(out_dir, "probs.npy"), y_probs)
    np.save(os.path.join(out_dir, "labels.npy"), y_true)
    mp = os.path.join(out_dir, "metrics.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics → {mp}")

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
        ax.set_title(f"Celeb-DF v2  |  AUC {auc:.3f} | Acc {acc:.3f}")
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

    print(f"\n  {'PASS' if acc >= 0.75 else 'BELOW'} — criterion is >75% cross-dataset accuracy.")


if __name__ == "__main__":
    main()
