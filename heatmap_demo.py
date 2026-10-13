#!/usr/bin/env python
"""Tiny Grad-CAM demo: pick a few frames from ANY video, show the
spatial-stream heatmap the detector would produce.

Usage:
  python heatmap_demo.py --video <path> [--frames 2] [--checkpoint <ckpt.pt>]

NOTES
  - No checkpoint? Weights are RANDOM — heatmap is just to eyeball
    the overlay look, not a real prediction.
  - Saves to gradcam/<video>_gradcam.png and opens it if possible.
"""

import argparse
import os

import torch

from src.config import DEVICE
from src.model import DeepfakeDetector, apply_checkpoint_arch
from src.preprocess import extract_faces


def main():
    """Render Grad-CAM heatmaps for a few evenly spaced frames of a video."""
    parser = argparse.ArgumentParser(description="Grad-CAM heatmap demo on any video")
    parser.add_argument("--video", required=True, help="Path to any .mp4")
    parser.add_argument(
        "--frames", type=int, default=2, help="How many evenly-spaced frames to heatmap (default 2)"
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Optional trained .pt. Defaults to random weights."
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: video not found — {args.video}")
        raise SystemExit(1)

    # ── Model ────────────────────────────────────────────────
    print(f"Building DeepfakeDetector on {DEVICE} ...")
    model = DeepfakeDetector().to(DEVICE)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {args.checkpoint} [{apply_checkpoint_arch(ckpt)}]")
    else:
        print("WARNING: no checkpoint — RANDOM weights (demo look only)")

    model.eval()

    # ── Faces ────────────────────────────────────────────────
    print(f"Extracting {args.frames} face crop(s) from {os.path.basename(args.video)} ...")
    raw_frames, meta = extract_faces(args.video, sample_count=args.frames)
    if raw_frames is None:
        print("ERROR: no faces detected in video.")
        raise SystemExit(1)

    n = raw_frames.shape[0]
    print(f"Got {n} face crop(s). Generating Grad-CAM ...")

    # ── Heatmap ──────────────────────────────────────────────
    from src.gradcam import generate_gradcam

    # input_tensor must require grad so the backward pass records hooks
    input_tensor = raw_frames.to(DEVICE).clone().requires_grad_(True)
    video_name = os.path.splitext(os.path.basename(args.video))[0]

    out_path = generate_gradcam(
        model=model,
        input_tensor=input_tensor,
        raw_frames=raw_frames,
        video_name=video_name,
    )
    print(f"\nSaved heatmap → {out_path}")

    # open it on the default viewer
    try:
        os.startfile(out_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
