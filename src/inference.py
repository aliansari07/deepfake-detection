"""Full inference pipeline: load model → extract faces →
augment → forward pass → optional Grad-CAM → print verdict.
Run via:  python infer.py
Import:   from src.inference import run_pipeline, batch_infer
"""

import os
import time

import torch

from src.augmentation import CompressionAwareAugmentation
from src.config import CONFIG, DEVICE
from src.logger import debug, log, log_gpu, section
from src.model import (
    DeepfakeDetector,
    apply_checkpoint_arch,
    checkpoint_channel_order,
    to_channel_order,
)
from src.preprocess import extract_faces


def _load_model(checkpoint_path=None):
    """Build the detector on DEVICE, optionally load checkpoint weights.
    Returns the model in eval mode. Shared by run_pipeline and batch_infer
    so the model is only constructed once per process [6.2]."""
    section("MODEL")
    # Fail loudly (before building the model): a missing checkpoint must never
    # fall back to random weights.
    if checkpoint_path and not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    t_load_start = time.time()
    model = DeepfakeDetector().to(DEVICE)
    t_load_time = time.time() - t_load_start

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        mode = apply_checkpoint_arch(ckpt)
        val_auc = ckpt.get("metrics", {}).get("auc", "N/A")
        # PLAN F1: extract_faces yields BGR, but v2 checkpoints were trained on the RGB cache.
        # Carry the expected order on the model so probe_video can convert.
        model.channel_order = checkpoint_channel_order(ckpt)
        log(
            "MODEL",
            f"Checkpoint: {os.path.basename(checkpoint_path)} "
            f"(Val AUC: {val_auc}) [{mode}] expects {model.channel_order}",
        )
    else:
        log("MODEL", "WARNING: No checkpoint — results are RANDOM. Train first.")

    model.eval()
    params = sum(p.numel() for p in model.parameters())
    log("MODEL", f"Ready — {params:,} params, ~{params * 4 / 1024**2:.1f} MB, {t_load_time:.2f}s")
    log_gpu()
    return model


def probe_video(model: DeepfakeDetector, video_path: str, total_start: float | None = None) -> dict:
    """
    Run the per-video stages using an already-loaded model:
    extract faces → augment → forward pass → optional Grad-CAM → verdict.

    Args:
        model       : Loaded DeepfakeDetector in eval mode.
        video_path  : Path to .mp4 video.
        total_start : Optional monotonic start time; shown in the verdict
                      box "Time" field (batch mode passes a per-video start).

    Returns:
        dict: result, prob, s_weight, gradcam_path (empty dict if no faces)
    """
    # ── Face extraction ───────────────────────────────────────
    section("STAGE 1: FACE EXTRACTION")
    raw_tensor, meta = extract_faces(video_path)
    if raw_tensor is None:
        log("PREPROCESS", "ABORTED — no faces extracted.")
        return {}

    # extract_faces crops straight from an OpenCV frame, so it is BGR. Models trained on the v2
    # cache learned RGB (PLAN F1). Getting this wrong does not fail loudly — it returns a
    # confident wrong answer — so it is converted here, once, at the boundary.
    wanted = getattr(model, "channel_order", "BGR")
    if wanted != "BGR":
        raw_tensor = to_channel_order(raw_tensor, "BGR", wanted)
        log("PREPROCESS", f"Converted BGR → {wanted} for this checkpoint.")

    # ── Augmentation ──────────────────────────────────────────
    section("AUGMENTATION")
    aug_time = 0
    if CONFIG["AUGMENT_ENABLED"]:
        augmentor = CompressionAwareAugmentation()
        t0 = time.time()
        aug_tensor, aug_logs = augmentor.augment_batch(raw_tensor)
        aug_time = time.time() - t0
        pixel_diff = (aug_tensor - raw_tensor).abs().mean().item()
        log(
            "AUGMENT", f"{raw_tensor.shape[0]} frames in {aug_time:.3f}s (pixel Δ={pixel_diff:.4f})"
        )
        if CONFIG["LOG_MODE"] == "debug":
            for i, fl in enumerate(aug_logs):
                debug(f"Frame {i + 1}: {', '.join(fl)}")
        input_tensor = aug_tensor.to(DEVICE)
    else:
        log("AUGMENT", "DISABLED")
        input_tensor = raw_tensor.to(DEVICE)

    # ── Forward pass ──────────────────────────────────────────
    section("STAGES 2–4: MODEL FORWARD PASS")
    forward_start = time.time()
    with torch.no_grad():
        logit, gate_g = model(input_tensor)
        prob = torch.sigmoid(logit).item()
    forward_time = time.time() - forward_start

    s_weight = gate_g.mean().item() if gate_g is not None else 1.0
    f_weight = 1.0 - s_weight
    prediction = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if prediction == "FAKE" else 1.0 - prob
    # [NFR-08] flag low-confidence predictions instead of overconfident labels
    if confidence < CONFIG["CONFIDENCE_THRESHOLD"]:
        result = "UNCERTAIN"
    else:
        result = prediction
    n_faces = len(meta["frames"])

    # ── Grad-CAM ──────────────────────────────────────────────
    gradcam_path = None
    region_facts = {"available": False}
    if CONFIG["GRADCAM_ENABLED"]:
        section("GRAD-CAM EXPLAINABILITY")
        try:
            from src.gradcam import generate_gradcam

            grad_video = os.path.splitext(os.path.basename(video_path))[0]
            region_out = {}
            gradcam_path = generate_gradcam(
                model=model,
                input_tensor=input_tensor,
                raw_frames=raw_tensor,
                video_name=grad_video,
                meta={
                    "video_name": os.path.basename(video_path),
                    "verdict": result,
                    "prob": prob,
                    "confidence": confidence,
                    "logit": logit.item(),
                    "s_weight": s_weight,
                    # False for a spatial-only checkpoint, so the header does not draw a gate
                    # split for a model that has no gate (PLAN F2a).
                    "has_gate": gate_g is not None,
                    "faces": n_faces,
                    "faces_total": CONFIG["SAMPLE_COUNT"],
                },
                frames_meta=meta["frames"],
                region_out=region_out,
            )
            region_facts = region_out or region_facts
        except Exception as e:
            log("GRADCAM", f"WARNING: Grad-CAM failed — {e}")

    # ── AI explanation ────────────────────────────────────────
    # Built from the same numbers just computed above (and the grounded heatmap region, if
    # any) — never from a fresh model call, so the text can't drift from the verdict it explains.
    try:
        from src.explain import explain_verdict

        ai_explanation = explain_verdict(
            {
                "verdict": result,
                "confidence": confidence,
                "prob": prob,
                "logit": logit.item(),
                "has_gate": gate_g is not None,
                "spatial_weight": s_weight,
                "freq_weight": f_weight,
                "region": region_facts,
            }
        )
    except Exception as e:
        log("EXPLAIN", f"WARNING: explanation generation failed — {e}")
        ai_explanation = ""

    # ── Verdict ───────────────────────────────────────────────
    total_time = (time.time() - total_start) if total_start else forward_time

    section("VERDICT")
    print(f"\n  ┌{'─' * 50}┐")
    print(f"  │  RESULT  :  {result:<37}│")
    print(f"  │  Prob    :  {prob * 100:.2f}%{'':<35}│")
    print(f"  │  Logit   :  {logit.item():.4f}{'':<37}│")
    print(f"  │  Spatial :  {s_weight * 100:.1f}% | Freq: {f_weight * 100:.1f}%{'':<24}│")
    print(f"  │  Faces   :  {n_faces}/{CONFIG['SAMPLE_COUNT']}{'':<41}│")
    print(
        f"  │  Time    :  {total_time:.1f}s (extract={meta['extract_time']:.1f}s | "
        f"model={forward_time:.3f}s){'':>5}│"
    )
    if gradcam_path:
        print(f"  │  GradCAM :  {os.path.basename(gradcam_path):<37}│")
    print(f"  └{'─' * 50}┘\n")

    return {
        "result": result,
        "prob": prob,
        "s_weight": s_weight,
        "gradcam_path": gradcam_path,
        # website-facing keys (consumed by website/backend/main.py)
        "verdict": result,
        "confidence": confidence,  # distance from the 0.5 boundary
        "logit": logit.item(),
        "faces_detected": n_faces,
        "faces_total": CONFIG["SAMPLE_COUNT"],
        "spatial_weight": s_weight,
        "freq_weight": f_weight,
        "gradcam_image": gradcam_path,
        "analysis_time": total_time,
        "ai_explanation": ai_explanation,
    }


def run_pipeline(
    video_path: str,
    checkpoint_path: str | None = None,
    model: DeepfakeDetector | None = None,
) -> dict:
    """
    Full deepfake detection pipeline for a single video.

    Steps:
        1. Load model (optionally from checkpoint) unless one is passed in.
        2. Extract faces from video.
        3. Apply compression-aware augmentation.
        4. Run model forward pass.
        5. Generate Grad-CAM heatmaps (if enabled).
        6. Print verdict.

    Args:
        video_path      : Path to .mp4 video.
        checkpoint_path : Path to best_model.pt (or None for random weights).
        model           : Optional already-loaded model — reused instead of
                          loading the weights again (useful in a loop).

    Returns:
        dict: result, prob, s_weight, gradcam_path (empty dict if no faces)
    """
    total_start = time.time()

    print("=" * 65)
    print("  DEEPFAKE DETECTION PIPELINE v7.0")
    print("=" * 65)

    if model is None:
        model = _load_model(checkpoint_path)

    return probe_video(model, video_path, total_start=total_start)


def batch_infer(
    video_paths: list[str], checkpoint_path: str | None = None, verbose: bool = True
) -> list[dict]:
    """
    Batch inference: load the model ONCE, then run every video [6.2].

    Args:
        video_paths     : List of .mp4 paths.
        checkpoint_path : Path to best_model.pt (or None for random weights).
        verbose         : Print per-video verdict boxes (default True).

    Returns:
        list[dict]: one per input video (empty dict for videos with no faces).
    """
    print("=" * 65)
    print("  DEEPFAKE DETECTION PIPELINE v7.0 — BATCH MODE")
    print(f"  {len(video_paths)} video(s) — model loaded once")
    print("=" * 65)

    model = _load_model(checkpoint_path)

    results = []
    for i, vp in enumerate(video_paths, 1):
        log("BATCH", f"[{i}/{len(video_paths)}] {vp}")
        res = probe_video(model, vp, total_start=time.time())
        results.append(res)

    section("SUMMARY")
    n_ok = sum(1 for r in results if r)
    n_fake = sum(1 for r in results if r.get("result") == "FAKE")
    n_real = sum(1 for r in results if r.get("result") == "REAL")
    print(
        f"\n  {len(video_paths)} videos — {n_ok} processed "
        f"({len(results) - n_ok} no-face) | FAKE={n_fake} REAL={n_real}\n"
    )
    for vp, res in zip(video_paths, results):
        name = os.path.basename(vp)
        if res:
            print(f"  {name:<60s} {res['result']:>4}  ({res['prob'] * 100:5.1f}%)")
        else:
            print(f"  {name:<60s} {'N/A':>4}  (no faces)")

    return results
