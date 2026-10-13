#!/usr/bin/env python
"""Entry point for training.

Usage:
  python train.py                # default run (DD_RUN_ID or "T4") + config epochs
  python train.py T4-20          # run "T4" for 20 epochs (id-keyed logs/ckpts/plots)
  python train.py T3-3           # any <name>-<epochs> combo
  python train.py --resume T4-20 # continue from T4-20.latest.pt (or latest.pt)

Ablation matrix (FIX_PLAN P3.2 / P4.1) — every flag is recorded in the
checkpoint's "arch" block, so evaluation rebuilds the right architecture
without being told:
  python train.py M1-20                               # full model
  python train.py A1-20 --no-freq                     # spatial + transformer
  python train.py A2-20 --no-freq --no-transformer    # spatial only (B6)
  python train.py A3-20 --freq-mode dct               # log-DCT instead of FFT
  python train.py A4-20 --no-augment                  # S3 control
"""

import argparse
import os

import torch

from src.config import CONFIG, DEVICE, TRAIN_CONFIG
from src.logger import section
from src.model import arch_label
from src.train import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepfakeDetector v7.0")
    parser.add_argument(
        "run_id",
        nargs="?",
        default=None,
        help="Run id like 'T4-20' (base name + epochs); empty = default run with config EPOCHS",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from the run's latest checkpoint"
    )

    ablation = parser.add_argument_group("ablations (recorded in the checkpoint)")
    ablation.add_argument(
        "--no-freq", action="store_true", help="Drop the frequency branch and the fusion gate"
    )
    ablation.add_argument(
        "--no-transformer",
        action="store_true",
        help="Drop the temporal Transformer (mean-pool instead)",
    )
    ablation.add_argument(
        "--freq-mode",
        choices=["fft", "dct"],
        default=None,
        help="Frequency transform (default: fft)",
    )
    ablation.add_argument(
        "--freq-augment",
        choices=["full", "gentle", "none"],
        default=None,
        help="augmentation profile for the FREQUENCY branch's view of each "
        "clip (R4). 'full' = both branches see the same frames "
        "(pre-R4 default); 'gentle' drops resize and blur, which R2 "
        "measured as the cause of the gate collapse",
    )
    ablation.add_argument(
        "--freq-norm",
        choices=["none", "per_image"],
        default=None,
        help="Frequency input normalisation. 'none' is the legacy "
        "FYP-1 behaviour (un-normalised, DCT without a log); "
        "'per_image' logs both modes then standardises each "
        "image. Use per_image for FYP-2 runs.",
    )
    ablation.add_argument(
        "--freq-input",
        choices=["spectrum", "residual"],
        default=None,
        help="WHAT the frequency branch looks at (R5). 'spectrum' is the "
        "log-magnitude FFT/DCT map and the default. 'residual' feeds the "
        "high-frequency spatial residual instead, which R1 measured as the "
        "far more separable representation (0.674 vs 0.561). With "
        "'residual' the --freq-mode choice no longer applies, and "
        "--freq-norm defaults to per_image because a raw residual is "
        "badly scaled for a CNN.",
    )
    ablation.add_argument(
        "--no-augment", action="store_true", help="Train without compression-aware augmentation"
    )
    ablation.add_argument(
        "--seed", type=int, default=None, help=f"Random seed (default: {TRAIN_CONFIG['SEED']})"
    )
    args = parser.parse_args()

    # Both off at once is exactly the FYP-1 B6 path, which the model implements
    # as the single BASELINE_MODE switch — use it so A2 reproduces B6 bit for
    # bit. Either one alone uses the independent switches.
    if args.no_freq and args.no_transformer:
        CONFIG["BASELINE_MODE"] = True
    else:
        CONFIG["FREQ_ENABLED"] = not args.no_freq
        CONFIG["TRANSFORMER_ENABLED"] = not args.no_transformer
    if args.freq_mode is not None:
        CONFIG["DCT_ENABLED"] = args.freq_mode == "dct"
    if args.freq_norm is not None:
        CONFIG["FREQ_NORM"] = args.freq_norm
    if args.freq_input is not None:
        CONFIG["FREQ_INPUT"] = args.freq_input
        if args.freq_input == "residual":
            # A raw residual is a few hundredths of the pixel range, so an unnormalised run would
            # starve the branch and look exactly like R5's hypothesis being false. Default to
            # per_image rather than letting the config's legacy "none" produce a false negative.
            if args.freq_norm is None:
                CONFIG["FREQ_NORM"] = "per_image"
                print(
                    "  NOTE    : --freq-input residual selected --freq-norm per_image "
                    "(a raw residual is badly scaled; pass --freq-norm explicitly to override)"
                )
            elif args.freq_norm == "none":
                print(
                    "  WARNING : --freq-input residual with --freq-norm none. The residual is "
                    "~0.01-0.05 of the pixel range unnormalised, so a null result from this "
                    "run would be uninterpretable. per_image is strongly recommended."
                )
            if args.freq_mode is not None:
                print(
                    f"  WARNING : --freq-mode {args.freq_mode} is ignored with "
                    "--freq-input residual (no Fourier transform is taken)."
                )
    if args.freq_augment is not None:
        CONFIG["FREQ_AUGMENT"] = args.freq_augment
    if args.no_augment:
        TRAIN_CONFIG["TRAIN_AUGMENT"] = False
    if args.seed is not None:
        TRAIN_CONFIG["SEED"] = args.seed

    section("SYSTEM")
    print(f"  Device  : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Dataset : {CONFIG['DATASET_ROOT']}")
    print(f"  Split   : {os.path.basename(CONFIG['SPLIT_FILE'])}")
    print(f"  Ckpts   : {CONFIG['CHECKPOINT_DIR']}")
    print(f"  Arch    : {arch_label()}")
    print(f"  FreqIn  : {CONFIG.get('FREQ_INPUT', 'spectrum')}")
    print(f"  FreqNorm: {CONFIG['FREQ_NORM']}")
    if CONFIG.get("FREQ_INPUT", "spectrum") == "residual":
        print("  DCT     : n/a (residual input takes no Fourier transform)")
    else:
        print(f"  DCT     : {'ON' if CONFIG['DCT_ENABLED'] else 'OFF (FFT)'}")
    print(f"  Baseline: {'ON' if CONFIG['BASELINE_MODE'] else 'OFF'}")
    print(f"  Augment : {'ON' if TRAIN_CONFIG['TRAIN_AUGMENT'] else 'OFF'}")
    print(f"  Seed    : {TRAIN_CONFIG['SEED']}")

    # T4-20 style: base name + epochs from the single positional arg.
    run_id = args.run_id
    if run_id and run_id.count("-") == 1 and run_id.split("-")[1].isdigit():
        os.environ["DD_RUN_ID"] = run_id.split("-")[0]
        TRAIN_CONFIG["EPOCHS"] = int(run_id.split("-")[1])
    elif run_id:
        os.environ["DD_RUN_ID"] = run_id

    trained_model, history = train_model(resume=args.resume, run_id=args.run_id)
