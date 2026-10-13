"""Ablation study utilities:
- analyse_fusion_weights()  [B3]
- compare_dct_fft()         [B4]
- run_baseline_training()   [B6]
"""

import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt

from src.config import CONFIG, DEVICE, TRAIN_CONFIG, override_config
from src.logger import detail, log, section
from src.model import apply_checkpoint_arch, frequency_transform
from src.preprocess import extract_faces

# ── [B3] Fusion weight analysis ───────────────────────────────


def analyse_fusion_weights(model, video_paths, labels, checkpoint_path=None):
    """
    [B3] Analyse gate weight distribution across real vs fake videos.

    If the gate is working correctly:
      - Real videos → higher spatial weight (face looks natural)
      - Fake videos → higher frequency weight (GAN artifacts present)

    Args:
        model          : DeepfakeDetector
        video_paths    : List of .mp4 paths
        labels         : List of int (0=real, 1=fake)
        checkpoint_path: Optional .pt file to load weights from

    Returns:
        results (list of dicts)
    """
    section("FUSION WEIGHT ANALYSIS [B3]")

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        log("B3", f"Loaded: {checkpoint_path} [{apply_checkpoint_arch(ckpt)}]")

    model.eval()
    results = []

    for i, (vpath, label) in enumerate(zip(video_paths, labels)):
        try:
            frames, meta = extract_faces(vpath)
            if frames is None:
                continue
            frames = frames.to(DEVICE)
            with torch.no_grad():
                logit, gate_g = model(frames)
                prob = torch.sigmoid(logit).item()
            if gate_g is not None:
                sw = gate_g.mean().item()
                gs = gate_g.std().item()
            else:
                sw, gs = 1.0, 0.0

            results.append(
                {
                    "path": vpath,
                    "label": label,
                    "label_str": "REAL" if label == 0 else "FAKE",
                    "spatial_weight": sw,
                    "freq_weight": 1.0 - sw,
                    "gate_std": gs,
                    "prob": prob,
                    "pred": "FAKE" if prob > 0.5 else "REAL",
                }
            )
            detail(
                f"[{i + 1}/{len(video_paths)}] "
                f"{os.path.basename(vpath)[:30]} | "
                f"Label={results[-1]['label_str']} | "
                f"Spatial={sw:.3f} | Prob={prob:.3f}"
            )
        except Exception as e:
            log("B3", f"Error on {os.path.basename(vpath)}: {e}")

    if not results:
        log("B3", "No results collected.")
        return []

    real_w = [r["spatial_weight"] for r in results if r["label"] == 0]
    fake_w = [r["spatial_weight"] for r in results if r["label"] == 1]

    log("B3", "Spatial weight stats:")
    detail(f"REAL — {np.mean(real_w):.3f} ± {np.std(real_w):.3f} (n={len(real_w)})")
    detail(f"FAKE — {np.mean(fake_w):.3f} ± {np.std(fake_w):.3f} (n={len(fake_w)})")

    if abs(np.mean(real_w) - np.mean(fake_w)) > 0.05:
        detail("Gate IS differentiating between real and fake ✓")
    else:
        detail("Gate is NOT clearly differentiating yet.")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if real_w:
        axes[0].hist(real_w, bins=15, alpha=0.6, color="#2E5FA3", label="REAL", density=True)
    if fake_w:
        axes[0].hist(fake_w, bins=15, alpha=0.6, color="#E25B4A", label="FAKE", density=True)
    axes[0].axvline(x=0.5, color="black", linestyle="--", alpha=0.5)
    axes[0].set(
        xlabel="Spatial gate weight", ylabel="Density", title="[B3] Fusion gate distribution"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    all_sw = [r["spatial_weight"] for r in results]
    all_probs = [r["prob"] for r in results]
    all_colors = ["#2E5FA3" if r["label"] == 0 else "#E25B4A" for r in results]
    axes[1].scatter(
        all_sw, all_probs, c=all_colors, alpha=0.7, s=60, edgecolors="white", linewidths=0.5
    )
    axes[1].axhline(y=0.5, color="black", linestyle="--", alpha=0.4)
    axes[1].axvline(x=0.5, color="black", linestyle="--", alpha=0.4)
    axes[1].set(
        xlabel="Spatial gate weight", ylabel="Fake probability", title="Gate weight vs prediction"
    )
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CONFIG["CHECKPOINT_DIR"], "fusion_analysis_b3.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log("B3", f"Plot saved → {path}")
    return results


def analyse_gate_split(model, split="val", out_name="fusion_analysis_b3.png"):
    """
    [B3] Gate-weight distribution over a whole dataset split (cached faces),
    grouped by class and manipulation. Replaces the old 7-video default,
    which used 1 real vs 6 fake training videos (audit-fyp2.md §4.4).

    Returns: dict with per-video records and per-group statistics.
    """
    from torch.utils.data import DataLoader

    from src.dataset import DeepfakeDataset

    section(f"FUSION GATE ANALYSIS [B3] — {split} split")
    if CONFIG["BASELINE_MODE"]:
        log("B3", "Baseline checkpoint has no gate — nothing to analyse.")
        return {}

    ds = DeepfakeDataset(CONFIG["DATASET_ROOT"], split=split, augment=False)
    loader = DataLoader(ds, batch_size=TRAIN_CONFIG["BATCH_SIZE"], shuffle=False, num_workers=0)
    model.eval()

    gates, probs = [], []
    with torch.no_grad():
        for frames, _, pad_mask in loader:
            logit, gate_g = model(frames.to(DEVICE), pad_mask.to(DEVICE))
            gates += gate_g.mean(dim=1).float().cpu().tolist()  # per-video mean gate
            probs += torch.sigmoid(logit).float().cpu().ravel().tolist()

    classes = [os.path.basename(os.path.dirname(p)) for p, _ in ds.samples]
    labels = [l for _, l in ds.samples]
    records = [
        {
            "path": os.path.relpath(p, CONFIG["DATASET_ROOT"]).replace(os.sep, "/"),
            "class": c,
            "label": l,
            "spatial_weight": g,
            "prob": pr,
        }
        for (p, _), c, l, g, pr in zip(ds.samples, classes, labels, gates, probs)
    ]

    def stats(values):
        v = np.asarray(values)
        return {
            "n": int(v.size),
            "mean": float(v.mean()),
            "std": float(v.std()),
            "min": float(v.min()),
            "max": float(v.max()),
        }

    groups = {
        "REAL": [r["spatial_weight"] for r in records if r["label"] == 0],
        "FAKE": [r["spatial_weight"] for r in records if r["label"] == 1],
    }
    for c in sorted(set(classes)):
        if c != "original":
            groups[c] = [r["spatial_weight"] for r in records if r["class"] == c]
    summary = {k: stats(v) for k, v in groups.items()}
    summary["ALL"] = stats(gates)

    # Cliff's delta (effect size) real vs fake: P(real > fake) - P(real < fake)
    real = np.asarray(groups["REAL"])[:, None]
    fake = np.asarray(groups["FAKE"])[None, :]
    cliffs_delta = float((real > fake).mean() - (real < fake).mean())

    log("B3", f"Spatial gate over {len(records)} {split} videos:")
    for k, s in summary.items():
        detail(
            f"{k:<18} n={s['n']:>5}  mean={s['mean']:.4f}  std={s['std']:.4f}  "
            f"range=[{s['min']:.3f}, {s['max']:.3f}]"
        )
    detail(
        f"REAL vs FAKE: Δmean={summary['REAL']['mean'] - summary['FAKE']['mean']:+.4f}, "
        f"Cliff's delta={cliffs_delta:+.3f} "
        f"({'negligible' if abs(cliffs_delta) < 0.147 else 'non-negligible'} effect)"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    names = [k for k in groups if k not in ("REAL", "FAKE")]
    axes[0].boxplot(
        [groups["REAL"]] + [groups[k] for k in names], labels=["original"] + names, showfliers=False
    )
    axes[0].axhline(0.5, color="black", linestyle="--", alpha=0.4)
    axes[0].set(
        ylabel="Mean spatial gate weight per video",
        title=f"[B3] Gate by class ({split}, n={len(records)})",
    )
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(alpha=0.3)
    colors = ["#2E5FA3" if l == 0 else "#E25B4A" for l in labels]
    axes[1].scatter(gates, probs, c=colors, s=10, alpha=0.5)
    axes[1].axhline(0.5, color="black", linestyle="--", alpha=0.4)
    axes[1].set(
        xlabel="Spatial gate weight",
        ylabel="Fake probability",
        title="Gate weight vs prediction (blue=REAL, red=FAKE)",
    )
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(CONFIG["CHECKPOINT_DIR"], out_name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log("B3", f"Plot saved → {path}")

    return {
        "split": split,
        "split_file": os.path.basename(ds.split_file),
        "summary": summary,
        "cliffs_delta_real_vs_fake": cliffs_delta,
        "videos": records,
    }


# ── Dv3: gate stress test (controlled, audit §20.4) ────────────


def analyse_gate_stress(model, split="val", n=None, seed=42):
    """
    Dv3: does the fusion gate actually move when a stream is denied information?

    §20's gate study was observational — it used whatever signal each manipulation happened to
    carry, and found the gate allocating sensibly but barely moving (range 0.146). This is the
    controlled complement proposal deliverable 3 asks for: force each stream, in turn, to carry
    no information at all, and measure whether the gate responds.

    Each video is scored three times, so the comparison is paired (same videos, same model, same
    forward pass batching):
      baseline        : both streams see the real clip.
      spatial_blanked : the SPATIAL branch sees a flat grey clip; the frequency branch still sees
                        the real one (via `freq_frames`, R4's plumbing — no model change needed).
      freq_blanked     : the reverse. The frequency branch sees a flat grey clip (a constant image
                        has zero high-frequency content under either representation: its residual
                        is exactly zero, and its spectrum has no magnitude off the DC bin); the
                        spatial branch sees the real one.

    If the gate adapts, spatial_blanked should push g down (lean on frequency, the only informative
    stream left) and freq_blanked should push g up (lean on spatial). If it barely moves even under
    total denial, that is a stronger claim than §20's observational one: not "adapts weakly to
    content" but "does not meaningfully respond to the presence or absence of a stream at all".

    Returns: dict with per-video records and summary statistics.
    """
    from torch.utils.data import DataLoader, Subset

    from src.dataset import DeepfakeDataset

    section(f"GATE STRESS TEST [Dv3] — {split} split")
    if CONFIG["BASELINE_MODE"]:
        log("Dv3", "Baseline checkpoint has no gate — nothing to stress.")
        return {}

    ds = DeepfakeDataset(CONFIG["DATASET_ROOT"], split=split, augment=False)
    indices = list(range(len(ds)))
    if n is not None and n < len(indices):
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(indices, size=n, replace=False).tolist())
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=TRAIN_CONFIG["BATCH_SIZE"], shuffle=False, num_workers=0)
    model.eval()

    baseline, spatial_blanked, freq_blanked = [], [], []
    with torch.no_grad():
        for frames, _, pad_mask in loader:
            frames, pad_mask = frames.to(DEVICE), pad_mask.to(DEVICE)
            grey = torch.full_like(frames, 0.5)

            _, g0 = model(frames, pad_mask)
            _, g1 = model(grey, pad_mask, freq_frames=frames)
            _, g2 = model(frames, pad_mask, freq_frames=grey)

            baseline += g0.mean(dim=1).float().cpu().tolist()
            spatial_blanked += g1.mean(dim=1).float().cpu().tolist()
            freq_blanked += g2.mean(dim=1).float().cpu().tolist()

    samples = [ds.samples[i] for i in indices]
    classes = [os.path.basename(os.path.dirname(p)) for p, _ in samples]
    labels = [l for _, l in samples]
    records = [
        {
            "path": os.path.relpath(p, CONFIG["DATASET_ROOT"]).replace(os.sep, "/"),
            "class": c,
            "label": l,
            "baseline": b,
            "spatial_blanked": sb,
            "freq_blanked": fb,
        }
        for (p, _), c, l, b, sb, fb in zip(
            samples, classes, labels, baseline, spatial_blanked, freq_blanked
        )
    ]

    b_arr, sb_arr, fb_arr = map(np.asarray, (baseline, spatial_blanked, freq_blanked))
    d_spatial = sb_arr - b_arr  # expected negative: lean toward frequency
    d_freq = fb_arr - b_arr  # expected positive: lean toward spatial

    def stats(v):
        v = np.asarray(v)
        return {
            "n": int(v.size),
            "mean": float(v.mean()),
            "std": float(v.std()),
            "min": float(v.min()),
            "max": float(v.max()),
        }

    summary = {
        "baseline": stats(b_arr),
        "spatial_blanked": stats(sb_arr),
        "freq_blanked": stats(fb_arr),
        "delta_spatial_blanked": stats(d_spatial),
        "delta_freq_blanked": stats(d_freq),
        "frac_moved_toward_freq_when_spatial_blanked": float((d_spatial < 0).mean()),
        "frac_moved_toward_spatial_when_freq_blanked": float((d_freq > 0).mean()),
    }

    log(
        "Dv3",
        f"Gate stress over {len(records)} {split} videos "
        f"(seed {seed}{'' if n is None else f', n={n} subset'}):",
    )
    detail(
        f"baseline            mean={summary['baseline']['mean']:.4f}  "
        f"range=[{summary['baseline']['min']:.3f}, {summary['baseline']['max']:.3f}]"
    )
    detail(
        f"spatial blanked     mean={summary['spatial_blanked']['mean']:.4f}  "
        f"Δ={summary['delta_spatial_blanked']['mean']:+.4f}  "
        f"({summary['frac_moved_toward_freq_when_spatial_blanked'] * 100:.1f}% moved toward freq)"
    )
    detail(
        f"freq blanked        mean={summary['freq_blanked']['mean']:.4f}  "
        f"Δ={summary['delta_freq_blanked']['mean']:+.4f}  "
        f"({summary['frac_moved_toward_spatial_when_freq_blanked'] * 100:.1f}% moved toward spatial)"  # noqa: E501
    )

    return {
        "split": split,
        "split_file": os.path.basename(ds.split_file),
        "n": len(records),
        "summary": summary,
        "videos": records,
    }


# ── [B4] DCT vs FFT comparison ────────────────────────────────


def compare_dct_fft(video_path):
    """
    [B4] Side-by-side visual comparison of FFT and DCT frequency maps.

    Args:
        video_path : Path to any .mp4 file.
    """
    section("DCT vs FFT COMPARISON [B4]")
    frames, meta = extract_faces(video_path)
    if frames is None:
        log("B4", "ERROR: No faces extracted.")
        return

    frames_gpu = frames.to(DEVICE)
    results = {}

    for mode, flag in [("FFT", False), ("DCT", True)]:
        with override_config(DCT_ENABLED=flag):
            # freq_input is pinned, not inherited: this comparison is about two Fourier
            # transforms, and a residual input takes neither. Without the pin, a session whose
            # CONFIG says "residual" (which apply_checkpoint_arch sets for any R5 checkpoint)
            # would render two identical panels captioned FFT and DCT.
            freq_map = frequency_transform(frames_gpu, freq_input="spectrum")
        fm_np = freq_map.cpu().numpy()
        results[mode] = {
            "freq_map": freq_map,
            "mean": float(np.mean(fm_np)),
            "std": float(np.std(fm_np)),
            "min": float(np.min(fm_np)),
            "max": float(np.max(fm_np)),
        }
        log("B4", f"{mode}: mean={results[mode]['mean']:.3f} | std={results[mode]['std']:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    face_rgb = cv2.cvtColor(
        (frames[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB
    )
    axes[0].imshow(face_rgb)
    axes[0].set_title(f"Original — {meta['video']}")
    axes[0].axis("off")

    im1 = axes[1].imshow(results["FFT"]["freq_map"][0, 0].cpu().numpy(), cmap="viridis")
    axes[1].set_title(f"FFT magnitude (std={results['FFT']['std']:.3f})")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(results["DCT"]["freq_map"][0, 0].cpu().numpy(), cmap="plasma")
    axes[2].set_title(f"DCT magnitude (std={results['DCT']['std']:.3f})")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.suptitle(f"[B4] DCT vs FFT — {os.path.basename(video_path)}", y=1.02)
    plt.tight_layout()
    path = os.path.join(CONFIG["CHECKPOINT_DIR"], "dct_vs_fft_b4.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log("B4", f"Plot saved → {path}")
    return results


# ── [B6] Baseline training ────────────────────────────────────


def run_baseline_training():
    """
    [B6] Train spatial-only baseline for ablation study.

    Saves results to checkpoints/baseline/ (does not overwrite
    the full model checkpoints).
    """
    from src.train import train_model

    section("ABLATION BASELINE TRAINING [B6]")
    log("B6", "BASELINE_MODE=True — freq branch + Transformer bypassed")

    baseline_dir = os.path.join(CONFIG["CHECKPOINT_DIR"], "baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    with override_config(BASELINE_MODE=True, CHECKPOINT_DIR=baseline_dir):
        try:
            # Own run id so baseline files never share the canonical T4-20 name.
            model, history = train_model(resume=False, run_id="B6-20")
            best_auc = max(history["val_auc"])
            log("B6", f"Baseline done. Best Val AUC: {best_auc:.4f}")
            log("B6", "Compare with full model AUC in your ablation table.")
        finally:
            log("B6", "CONFIG restored.")

    return model, history
