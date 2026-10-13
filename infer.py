#!/usr/bin/env python
"""CLI entry point for inference.

Usage:
  python infer.py --video <path>            # full pipeline + Grad-CAM
  python infer.py --video <path> --checkpoint <ckpt.pt>

Verdict hostname (matches report): frames -> faces -> detector ->
detail heatmaps flags.
"""

import argparse
import os
import sys

import torch

from src.config import CONFIG, DEVICE
from src.inference import run_pipeline
from src.logger import section


def main():
    """Run the full detection pipeline on one video and print the verdict."""
    parser = argparse.ArgumentParser(description="Run DeepfakeDetector v7.0")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument(
        "--checkpoint", default=None, help="Checkpoint to load (defaults to best_model.pt)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: video not found — {args.video}")
        sys.exit(1)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(CONFIG["CHECKPOINT_DIR"], "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"ERROR: no best_model.pt at {ckpt_path}. Provide --checkpoint or train first.")
            sys.exit(1)

    section("SYSTEM")
    print(f"  Device  : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")

    result = run_pipeline(args.video, ckpt_path)

    print("")
    print("=" * 60)
    print(f"  Video      : {args.video}")
    print(f"  Verdict    : {result['verdict']}")
    print(f"  Confidence : {result['confidence'] * 100:.1f}% ({result['confidence']:.4f})")
    print(f"  Faces      : {result.get('faces_detected', 0)}")
    if "spatial_weight" in result:
        print(
            f"  Gate       : spatial {result['spatial_weight']:.3f} / "
            f"freq {1 - result['spatial_weight']:.3f}"
        )
    if result.get("gradcam_image"):
        print(f"  Grad-CAM   : {result['gradcam_image']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
