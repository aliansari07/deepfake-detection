"""ALL paths and hyperparameters for DeepfakeDetector v7.0.
Every module reads CONFIG / TRAIN_CONFIG from here.
Edit this file first when changing anything.
"""

import os
from contextlib import contextmanager

import torch

# ── Project paths ───────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_ROOT = os.path.join(_PROJECT_ROOT, "datasets", "FaceForensics++_C23")

CONFIG = {
    # Paths
    "DATASET_ROOT": _DATASET_ROOT,
    "CHECKPOINT_DIR": os.path.join(_PROJECT_ROOT, "checkpoints"),
    "GRADCAM_DIR": os.path.join(_PROJECT_ROOT, "gradcam"),
    # Committed train/val(/test) split. legacy_t4-20.json reproduces every
    # FYP-1 result; it is identity-overlapping (see audit-fyp2.md F1).
    # splits/ffpp_official_v1.json is the identity-disjoint FYP-2 split
    # (face cache v2, RGB); select it with DD_SPLIT_FILE for new runs.
    "SPLIT_FILE": os.environ.get("DD_SPLIT_FILE")
    or os.path.join(_PROJECT_ROOT, "splits", "legacy_t4-20.json"),
    # Logging detail: "pipeline" | "debug"
    "LOG_MODE": "pipeline",
    # ── Face extraction ──────────────────────────────────────
    "SAMPLE_COUNT": 10,  # N evenly-spaced frames per video
    "FACE_SIZE": (224, 224),  # final crop resolution
    "DETECTION_THRESHOLD": 0.5,  # min InsightFace det_score
    "DET_SIZE": (640, 640),  # insightface preview size
    # ── Architecture / pipeline flags ────────────────────────
    "DCT_ENABLED": False,  # True = DCT instead of FFT [B4]
    "BASELINE_MODE": False,  # True = spatial CNN only, no freq/transformer [B6]
    # Independent ablation switches (FIX_PLAN P3.2). BASELINE_MODE stays the
    # legacy combined switch and takes precedence when True, so FYP-1 baseline
    # checkpoints keep their exact forward pass. These two allow the A1 run
    # (no frequency branch, Transformer kept), which BASELINE_MODE cannot express.
    "FREQ_ENABLED": True,  # False = no frequency branch, no gate [A1]
    "TRANSFORMER_ENABLED": True,  # False = mean-pool instead of the Transformer
    "GRADCAM_ENABLED": True,  # False = skip heatmap generation
    # Inference-time augmentation only (src/inference.py). Keep False: random
    # distortions make verdicts non-deterministic. Training augmentation is
    # always on via build_dataloaders() and does not read this flag.
    "AUGMENT_ENABLED": False,
    # Confidence gate for "UNCERTAIN" verdicts [NFR-08]. A verdict is
    # UNCERTAIN when confidence = max(p, 1-p) is below this threshold.
    "CONFIDENCE_THRESHOLD": 0.6,
    # Frequency transform: log(|FFT| + eps)
    "FREQ_EPS": 1e-6,
    # "none"      = legacy, un-normalised, and DCT without a log (FYP-1 behaviour,
    #               kept so every FYP-1 checkpoint reproduces exactly).
    # "per_image" = log for both modes, then zero mean / unit variance per image.
    #               Use for FYP-2 runs: `train.py --freq-norm per_image` (P3.1).
    "FREQ_NORM": "none",
    # What the frequency branch actually looks at (R5, PLAN_BINDER.md Tab 5, audit §20.3).
    # "spectrum" = the log-magnitude FFT/DCT map. The FYP-1 and FYP-2 behaviour, and the default,
    #              so every existing checkpoint reproduces bit for bit.
    # "residual" = the high-frequency spatial residual, frame minus its Gaussian blur. Attacks
    #              cause C1: in a Fourier magnitude image absolute position *is* the meaning
    #              (radius = frequency, angle = orientation), so a translation-equivariant,
    #              globally-pooled, ImageNet-pretrained CNN has the wrong inductive bias and no
    #              transferable prior. A residual map is an ordinary image, where translation
    #              equivariance is correct and ImageNet edge filters do transfer.
    #              R1 measured the representations directly: residual probe 0.674 vs spectrum
    #              0.561, on fewer than half the features (audit §15.1).
    # Use `train.py --freq-input residual` (R5). Residual needs FREQ_NORM="per_image" to be
    # sanely scaled; train.py selects that automatically unless told otherwise.
    "FREQ_INPUT": "spectrum",
    # Augmentation profile for the FREQUENCY branch's view of each clip (R4,
    # PLAN_BINDER.md Tab 5). "full" means both branches see the same augmented frames, which is
    # the pre-R4 behaviour and the default. "gentle" gives the frequency transform a lightly
    # augmented copy instead, because R2 measured that resize/blur is what collapses the fusion
    # gate (audit-fyp2.md section 15.2). The spatial branch always keeps the full profile.
    "FREQ_AUGMENT": "full",
}

# ── Training configuration (T4-20 run) ──────────────────────────
TRAIN_CONFIG = {
    "BATCH_SIZE": 2,  # effective batch 8 via GRAD_ACCUM=4
    "LR": 1e-4,
    "MIN_LR": 1e-6,
    "EPOCHS": 20,
    "WARMUP_EPOCHS": 2,
    "DROPOUT": 0.3,
    "GRAD_CLIP": 1.0,
    "LABEL_SMOOTH": 0.1,
    "EARLY_STOP_PATIENCE": 10,
    "GRAD_ACCUM": 4,
    "EMA": False,
    "SEED": 42,
    "TRAIN_RATIO": 0.8,
    # Compression-aware augmentation during TRAINING only. The A4 ablation
    # (`train.py A4-20 --no-augment`) turns it off to give the S3 robustness
    # claim a control. Inference augmentation is CONFIG["AUGMENT_ENABLED"],
    # which is a separate flag and stays False.
    "TRAIN_AUGMENT": True,
    "NUM_WORKERS": 4,  # parallel .pt load + augment (GPU-starved at 0)
    # Speed: bfloat16 autocast (Ada), cudnn.benchmark, TF32 matmul
    "AMP_ENABLED": True,
    "CUDNN_BENCHMARK": True,
    "TF32_MATMUL": True,
    # Log a full progress line every N batches (loss/acc/auc/gate/lr/eta)
    "BATCH_LOG_INTERVAL": 30,
    # ── Architecture dims ────────────────────────────────────
    "D_MODEL": 512,
    "NHEAD": 8,
    "D_FF": 2048,
    "N_LAYERS": 2,
    "POSITIONAL_EMB": True,
}


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = _device()


@contextmanager
def override_config(**kwargs):
    """
    Temporarily override CONFIG / TRAIN_CONFIG keys inside a 'with' block.
    Used by the B4 (DCT/FFT) and B6 (baseline) ablation paths to mutate
    behaviour without touching global state.

    Usage:
        with override_config(DCT_ENABLED=True):
            freq_map = frequency_transform(frames)
    """
    saved = {}
    for key, value in kwargs.items():
        if key in CONFIG:
            saved[key] = (CONFIG, CONFIG[key])
            CONFIG[key] = value
        elif key in TRAIN_CONFIG:
            saved[key] = (TRAIN_CONFIG, TRAIN_CONFIG[key])
            TRAIN_CONFIG[key] = value
        else:
            raise KeyError(f"Unknown config key: {key}")

    try:
        yield
    finally:
        for key, (target, old) in saved.items():
            target[key] = old
