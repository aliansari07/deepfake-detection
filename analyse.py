#!/usr/bin/env python
"""Ablation / analysis tools.

Usage:
  python analyse.py gate        --checkpoint <ckpt> [--split val]   # whole split, cached faces
  python analyse.py gate        --checkpoint <ckpt> --videos v... --labels 0 1 ...
  python analyse.py gate-stress --checkpoint <ckpt> [--split val] [--n N]  # Dv3, controlled
  python analyse.py dct         --video <path>
  python analyse.py baseline

Commands:
  gate         [B3]  gate-weight distribution over real vs fake videos (observational)
  gate-stress  [Dv3] blank one stream at a time, measure whether the gate responds (controlled)
  dct          [B4]  FFT vs DCT frequency-map comparison
  baseline     [B6]  train the spatial-only baseline (ablation)
"""

import argparse
import json
import os
import sys

import torch

from src.config import CONFIG, DEVICE
from src.logger import section


def cmd_gate(args):
    """Report the fusion-gate weight distribution over real vs fake videos."""
    ckpt = args.checkpoint or os.path.join(CONFIG["CHECKPOINT_DIR"], "best_model.pt")
    if not os.path.exists(ckpt):
        print(f"ERROR: checkpoint not found — {ckpt}")
        sys.exit(1)

    from src.model import DeepfakeDetector, apply_checkpoint_arch

    model = DeepfakeDetector().to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state["model_state"])
    print(f"  Mode : {apply_checkpoint_arch(state)}")

    if args.videos:
        # Explicit videos: faces are re-extracted from the .mp4 files.
        from src.analysis import analyse_fusion_weights

        labels = args.labels if args.labels is not None else [0, 1] * (len(args.videos) // 2)
        print(
            "  NOTE: explicit --videos may include training videos; "
            "use the default (whole split) for reportable numbers."
        )
        results = analyse_fusion_weights(model, args.videos, labels, checkpoint_path=None)
        if results:
            real = [r for r in results if r["label"] == 0]
            fake = [r for r in results if r["label"] == 1]
            if real and fake:
                print(
                    f"\n  Summary — spatial gate: REAL "
                    f"{sum(r['spatial_weight'] for r in real) / len(real):.3f} vs FAKE "
                    f"{sum(r['spatial_weight'] for r in fake) / len(fake):.3f}"
                )
        return

    # Default: every video in the split, from the cached faces.
    from src.analysis import analyse_gate_split

    result = analyse_gate_split(model, split=args.split)
    if result:
        name = os.path.splitext(os.path.basename(ckpt))[0]
        out = os.path.join(os.path.dirname(ckpt), f"eval_{name}", f"gate_{args.split}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": ckpt, **result}, f, indent=1)
        print(f"\n  Gate stats → {out}")


def cmd_gate_stress(args):
    """Blank one input stream at a time and measure whether the gate responds."""
    ckpt = args.checkpoint or os.path.join(CONFIG["CHECKPOINT_DIR"], "best_model.pt")
    if not os.path.exists(ckpt):
        print(f"ERROR: checkpoint not found — {ckpt}")
        sys.exit(1)

    from src.model import DeepfakeDetector, apply_checkpoint_arch

    model = DeepfakeDetector().to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state["model_state"])
    print(f"  Mode : {apply_checkpoint_arch(state)}")

    from src.analysis import analyse_gate_stress

    result = analyse_gate_stress(model, split=args.split, n=args.n, seed=args.seed)
    if result:
        name = os.path.splitext(os.path.basename(ckpt))[0]
        out = os.path.join(os.path.dirname(ckpt), f"eval_{name}", f"gate_stress_{args.split}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": ckpt, **result}, f, indent=1)
        print(f"\n  Gate stress stats → {out}")


def cmd_dct(args):
    """Compare FFT and DCT frequency maps for a single video."""
    from src.analysis import compare_dct_fft

    if not os.path.exists(args.video):
        print(f"ERROR: video not found — {args.video}")
        sys.exit(1)
    compare_dct_fft(args.video)


def cmd_baseline(args):
    """Train the spatial-only baseline used in the ablation study."""
    from src.analysis import run_baseline_training

    run_baseline_training()


def main():
    """Parse the sub-command and dispatch to the matching ``cmd_*`` handler."""
    parser = argparse.ArgumentParser(
        description="DeepfakeDetector analysis / ablation tools", prog="analyse.py"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gate = sub.add_parser("gate", help="[B3] fusion gate analysis")
    p_gate.add_argument("--checkpoint", default=None)
    p_gate.add_argument(
        "--split",
        default="val",
        help="Split from CONFIG['SPLIT_FILE'] used when --videos is not given",
    )
    p_gate.add_argument("--videos", nargs="+", default=None)
    p_gate.add_argument("--labels", nargs="+", type=int, default=None)
    p_gate.set_defaults(func=cmd_gate)

    p_stress = sub.add_parser("gate-stress", help="[Dv3] controlled gate stress test")
    p_stress.add_argument("--checkpoint", default=None)
    p_stress.add_argument("--split", default="val")
    p_stress.add_argument("--n", type=int, default=None, help="Subset size (default: whole split)")
    p_stress.add_argument("--seed", type=int, default=42)
    p_stress.set_defaults(func=cmd_gate_stress)

    p_dct = sub.add_parser("dct", help="[B4] DCT vs FFT comparison")
    p_dct.add_argument("--video", required=True)
    p_dct.set_defaults(func=cmd_dct)

    p_base = sub.add_parser("baseline", help="[B6] train spatial-only baseline")
    p_base.set_defaults(func=cmd_baseline)

    args = parser.parse_args()
    section("ANALYSIS TOOLS")
    args.func(args)


if __name__ == "__main__":
    main()
