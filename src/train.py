"""Training loop for DeepfakeDetector v7.0 (T4-20 configuration).

  train_model(resume=False) -> (model, history)

- AdamW + manual warmup + cosine LR decay to 1e-6
- BCE with label smoothing, gradient clipping, gradient accumulation
- WeightedRandomSampler (1:6 real:fake) — see dataset.py
- early stopping on validation AUC (patience 10)
- saves checkpoints/latest.pt + checkpoints/best_model.pt
- per-epoch jsonl + markdown run logs, training curves + gate drift plots
"""

import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src.config import CONFIG, DEVICE, TRAIN_CONFIG
from src.logger import log, section
from src.model import DeepfakeDetector, count_params

_CKPT_KEYS = ["model_state", "optimizer_state", "epoch", "history", "metrics", "config"]
_RUN_ID = os.environ.get("DD_RUN_ID", "T4")


# ── ANSI colors + small printing helpers ─────────────────────────


class A:
    """ANSI color codes (reset-safe so every colored cell ends clean)."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"


def _bar(fraction, width=14):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def _fmt_dur(seconds):
    seconds = int(round(max(0, seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _loss_col(value):
    """BCE loss: green = good, yellow = okay, red = bad."""
    if value != value:
        return A.GRAY
    return A.GREEN if value <= 0.65 else A.YELLOW if value <= 0.72 else A.RED


def _score_col(value, good=0.60):
    """acc/auc: green above `good`, yellow within 0.15, red below."""
    if value != value:
        return A.GRAY
    return A.GREEN if value >= good else A.YELLOW if value >= good - 0.15 else A.RED


def _paint(text, color):
    return f"{color}{text}{A.RESET}"


def _cell(text, width, color=A.WHITE):
    return _paint(f"{text:>{width}}", color)


def _amp():
    """bfloat16 autocast when enabled on CUDA, else a no-op context."""
    use = DEVICE.type == "cuda" and TRAIN_CONFIG.get("AMP_ENABLED", True)
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use
        else torch.autocast("cuda", enabled=False)
    )


# ── LR schedule: linear warmup then cosine to MIN_LR ─────────────


def _lr_at(epoch_idx):
    warmup = TRAIN_CONFIG["WARMUP_EPOCHS"]
    base = TRAIN_CONFIG["LR"]
    if epoch_idx < warmup:
        return base * (epoch_idx + 1) / warmup
    t = epoch_idx / TRAIN_CONFIG["EPOCHS"]
    return TRAIN_CONFIG["MIN_LR"] + (base - TRAIN_CONFIG["MIN_LR"]) * 0.5 * (
        1.0 + math.cos(math.pi * t)
    )


# ── One training / validation epoch ──────────────────────────────


def _bce_smoothed(logits, targets, ls):
    """BCE-with-logits loss with targets smoothed toward 0.5 (label smoothing)."""
    if ls > 0:
        targets = targets * (1.0 - ls) + 0.5 * ls
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def _train_epoch(model, loader, optimizer, epoch_idx, n_epochs):
    model.train()
    accum = TRAIN_CONFIG["GRAD_ACCUM"]
    interval = TRAIN_CONFIG["BATCH_LOG_INTERVAL"]
    n_batches = len(loader)
    total_loss, total_correct, total = 0.0, 0, 0
    gate_sum, gate_n = 0.0, 0
    probs_all, labels_all = [], []
    optimizer.zero_grad(set_to_none=True)

    start = time.time()
    t_last = start

    print(
        _paint(
            f"  Epoch {epoch_idx + 1}/{n_epochs}  {_bar(0.0)}  0%  (run {_RUN_ID})", A.BOLD + A.CYAN
        ),
        flush=True,
    )
    print(
        _paint(
            "   step   loss      acc     auc     gate      lr      time       eta", A.BOLD + A.BLUE
        ),
        flush=True,
    )

    for step, batch in enumerate(loader, 1):
        # A 4th element is the frequency branch's own, gentler view of the clip (R4). The
        # training dataset yields it only when FREQ_AUGMENT is not "full"; validation and every
        # evaluation script yield the 3-tuple, so unpacking tolerantly keeps one code path.
        frames, labels, pad_mask = batch[0], batch[1], batch[2]
        freq_frames = batch[3].to(DEVICE) if len(batch) > 3 else None
        frames = frames.to(DEVICE)  # (B, N, 3, 224, 224)
        pad_mask = pad_mask.to(DEVICE)  # (B, N) True = padded
        labels = labels.to(DEVICE).float()  # (B,)
        labels = labels.unsqueeze(1)

        with _amp():
            logits, gate = model(frames, pad_mask, freq_frames)
            loss = _bce_smoothed(logits, labels, TRAIN_CONFIG["LABEL_SMOOTH"]) / accum
        loss.backward()

        probs = torch.sigmoid(logits.detach())
        preds = (probs > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total += labels.numel()
        total_loss += loss.item() * accum
        probs_all.append(probs.detach().float().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())
        if gate is not None:
            gate_sum += float(gate.detach().mean())
            gate_n += 1

        if step % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["GRAD_CLIP"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if (step % interval == 0) or (step == n_batches):
            now = time.time()
            run_loss = total_loss / step
            run_acc = total_correct / max(total, 1)
            run_probs = np.concatenate(probs_all).ravel()
            run_labels = np.concatenate(labels_all).ravel()
            try:
                run_auc = roc_auc_score(run_labels, run_probs)
            except ValueError:
                run_auc = float("nan")
            run_gate = gate_sum / max(gate_n, 1)
            dps = (now - t_last) / min(interval, step - ((step - 1) // interval) * interval)
            t_last = now
            eta = dps * (n_batches - step)
            lr_now = optimizer.param_groups[0]["lr"]
            pct = 100.0 * step / n_batches
            auc_txt = "  n/a" if run_auc != run_auc else f"{run_auc:6.4f}"
            print(
                f"  {_cell(step, 6)} "
                f"{_cell(f'{run_loss:7.4f}', 8, _loss_col(run_loss))} "
                f"{_cell(f'{run_acc * 100:5.1f}%', 7, _score_col(run_acc, 0.60))} "
                f"{_cell(auc_txt, 7, _score_col(run_auc, 0.55))} "
                f"{_cell(f'{run_gate:6.3f}', 7, A.MAGENTA)} "
                f"{_cell(f'{lr_now:8.1e}', 9, A.YELLOW)} "
                f"{_cell(_fmt_dur(now - start), 7, A.CYAN)} "
                f"{_cell(f'{eta:7.0f}s', 8, A.GRAY)} "
                f"[{_bar(step / n_batches)}]{pct:3.0f}%",
                flush=True,
            )

    return {
        "loss": total_loss / len(loader),  # mean per-batch (accum-scaled)
        "acc": total_correct / max(total, 1),
        "gate": gate_sum / max(gate_n, 1),
    }


def _validate(model, loader):
    model.eval()
    losses, probs_all, labels_all, gates = [], [], [], []
    n_batches = len(loader)
    interval = TRAIN_CONFIG["BATCH_LOG_INTERVAL"]
    start = time.time()

    with torch.no_grad():
        print(_paint("  ─ Validate ─", A.BOLD + A.MAGENTA), flush=True)
        print(
            _paint("   step   vloss     vacc     vauc    gate      time", A.BOLD + A.BLUE),
            flush=True,
        )

        for step, (frames, labels, pad_mask) in enumerate(loader, 1):
            frames = frames.to(DEVICE)
            pad_mask = pad_mask.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)
            with _amp():
                logits, gate = model(frames, pad_mask)
                losses.append(_bce_smoothed(logits, labels, TRAIN_CONFIG["LABEL_SMOOTH"]).item())
            probs_all.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
            labels_all.append(labels.cpu().numpy())
            if gate is not None:
                g = float(gate.detach().mean())
                gates.append(g)
            if (step % interval == 0) or (step == n_batches):
                seg = np.concatenate(probs_all).ravel()
                seg_lab = np.concatenate(labels_all).ravel()
                seg_pred = (seg > 0.5).astype(int)
                try:
                    seg_auc = roc_auc_score(seg_lab, seg)
                except ValueError:
                    seg_auc = float("nan")
                seg_gate = float(np.mean(gates)) if gates else 1.0
                val_acc = accuracy_score(seg_lab, seg_pred)
                auc_txt = "  n/a" if seg_auc != seg_auc else f"{seg_auc:6.4f}"
                print(
                    f"  {_cell(step, 6)} "
                    f"{_cell(f'{np.mean(losses):7.4f}', 8, _loss_col(np.mean(losses)))} "
                    f"{_cell(f'{val_acc * 100:5.1f}%', 8, _score_col(val_acc, 0.60))} "
                    f"{_cell(auc_txt, 8, _score_col(seg_auc, 0.55))} "
                    f"{_cell(f'{seg_gate:6.3f}', 7, A.MAGENTA)} "
                    f"{_cell(_fmt_dur(time.time() - start), 7, A.CYAN)}",
                    flush=True,
                )

    probs = np.concatenate(probs_all).ravel()
    labels = np.concatenate(labels_all).ravel()
    preds = (probs > 0.5).astype(int)

    return {
        "loss": float(np.mean(losses)),
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": float(roc_auc_score(labels, probs)),
        "gate": float(np.mean(gates)) if gates else 1.0,
        "threshold": _youden_threshold(labels, probs),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
    }


def _youden_threshold(labels, probs):
    """Probability cut that maximises (TPR - FPR) on this validation run.

    Stored with the best epoch's checkpoint so later evaluation can report
    accuracy at a threshold chosen on val, instead of assuming 0.5 on a
    dataset whose classes are 1:5 (FIX_PLAN P3.4).
    """
    labels = np.asarray(labels).ravel()
    if len(np.unique(labels)) < 2:
        return 0.5  # one class present: no curve to cut
    fpr, tpr, thr = roc_curve(labels, probs)
    best = float(thr[int(np.argmax(tpr - fpr))])
    # roc_curve's first threshold is +inf (the "predict nothing" corner).
    return 0.5 if not np.isfinite(best) else best


# ── Checkpoint helpers ───────────────────────────────────────────


def _git_commit():
    """Short hash of the commit this run was launched from, or None."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _split_provenance():
    """Which split file this run used, and its checksum.

    Pins the data behind a checkpoint: if the split file is ever rebuilt, a
    checkpoint trained on the old one is identifiable by its hash rather than
    silently compared against different videos (audit-fyp2.md §5.1).
    """
    path = CONFIG.get("SPLIT_FILE")
    if not path or not os.path.isfile(path):
        return {"file": path, "sha256": None}
    from src.face_cache import sha256_file

    return {"file": os.path.basename(path), "sha256": sha256_file(path)}


def _arch_block():
    """The architecture switches the weights were trained under."""
    return {
        "baseline_mode": bool(CONFIG["BASELINE_MODE"]),
        "freq_enabled": bool(CONFIG["FREQ_ENABLED"]),
        "transformer_enabled": bool(CONFIG["TRANSFORMER_ENABLED"]),
        "freq_mode": "dct" if CONFIG["DCT_ENABLED"] else "fft",
        "freq_norm": str(CONFIG["FREQ_NORM"]),
        "freq_input": str(CONFIG.get("FREQ_INPUT", "spectrum")),
        "sample_count": int(CONFIG["SAMPLE_COUNT"]),
        "channel_order": "RGB" if "official" in str(CONFIG.get("SPLIT_FILE", "")) else "BGR",
        "train_augment": bool(TRAIN_CONFIG["TRAIN_AUGMENT"]),
        "freq_augment": str(CONFIG.get("FREQ_AUGMENT", "full")),
        "seed": int(TRAIN_CONFIG["SEED"]),
    }


def _rng_state():
    """Python / numpy / torch / cuda RNG states, in weights_only-safe types."""
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            np_state[0],
            np_state[1].tolist(),
            int(np_state[2]),
            int(np_state[3]),
            float(np_state[4]),
        ),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state):
    """Put the RNGs back where a saved checkpoint left them.

    Without this, resuming replays the augmentation and sampler draws from the
    seed rather than from where the run stopped, so a resumed run is not the
    run it claims to continue (FIX_PLAN P3.4).
    """
    if not state:
        return False
    try:
        random.setstate(_as_tuple(state["python"]))
        np_s = state["numpy"]
        np.random.set_state(
            (np_s[0], np.array(np_s[1], dtype=np.uint32), np_s[2], np_s[3], np_s[4])
        )
        torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
        if torch.cuda.is_available() and len(state.get("cuda", [])):
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in state["cuda"]])
        return True
    except Exception as e:
        log("TRAIN", f"WARNING: could not restore RNG state — {e}")
        return False


def _as_tuple(obj):
    """torch.save round-trips random.getstate()'s nested tuple as lists."""
    if isinstance(obj, (list, tuple)):
        return tuple(_as_tuple(x) for x in obj)
    return obj


#: Fields a checkpoint's metrics block must carry. `threshold` and
#: `balanced_acc` are here because dropping them is exactly the regression
#: audit-fyp2.md §13.3 describes: every M1 checkpoint stored threshold 0.5.
CKPT_METRIC_KEYS = ("auc", "accuracy", "f1", "epoch", "threshold", "balanced_acc")


def _epoch_metrics(va, epoch):
    """Build a checkpoint metrics block from a validation result.

    Forwards the whole validation dict rather than picking fields out of it, so
    a metric added to _validate() reaches the checkpoint without a second edit.
    """
    metrics = {**va, "accuracy": va["acc"], "epoch": epoch}
    missing = [k for k in CKPT_METRIC_KEYS if k not in metrics]
    if missing:
        raise KeyError(
            f"validation result is missing {missing}; "
            f"_validate() must return them (audit-fyp2 §13.3)"
        )
    return metrics


def _save_ckpt(model, optimizer, epoch_idx, history, metrics, path):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch_idx,
            "history": history,
            "metrics": metrics,
            "config": {"TRAIN_CONFIG": TRAIN_CONFIG, "CONFIG": CONFIG},
            # Provenance: what this checkpoint is, and what made it (FIX_PLAN P3.4).
            "arch": _arch_block(),
            "split": _split_provenance(),
            "git_commit": _git_commit(),
            "threshold": float(metrics.get("threshold", 0.5)),
            "rng": _rng_state(),
        },
        path,
    )


def _best_from_history(history):
    """(best_auc, best_epoch, patience) recovered from a saved history.
    best_epoch is 1-based (-1 if no epochs); patience = epochs since best."""
    aucs = history.get("val_auc", [])
    if not aucs:
        return 0.0, -1, 0
    best_idx = int(np.argmax(aucs))
    return float(aucs[best_idx]), best_idx + 1, len(aucs) - (best_idx + 1)


def _load_ckpt(model, optimizer, path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return (ckpt.get("epoch", 0), ckpt.get("history", {}), ckpt.get("metrics", {}), ckpt.get("rng"))


# ── Run logs (jsonl + markdown) ──────────────────────────────────


def _log_dir():
    d = os.path.join(CONFIG["CHECKPOINT_DIR"], "..", "logs")
    d = os.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    return d


def _write_jsonl(log_dir, run_id, row):
    with open(os.path.join(log_dir, f"{run_id}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _summarise_run(run_id, best_auc, history):
    lines = [
        f"# Training Run: {run_id}",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"**Device:** {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'}",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Batch Size | {TRAIN_CONFIG['BATCH_SIZE']} |",
        f"| Effective Batch | {TRAIN_CONFIG['BATCH_SIZE'] * TRAIN_CONFIG['GRAD_ACCUM']} |",
        f"| Learning Rate | {TRAIN_CONFIG['LR']} |",
        f"| Min LR | {TRAIN_CONFIG['MIN_LR']} |",
        f"| Epochs | {TRAIN_CONFIG['EPOCHS']} |",
        f"| Warmup Epochs | {TRAIN_CONFIG['WARMUP_EPOCHS']} |",
        f"| Dropout | {TRAIN_CONFIG['DROPOUT']} |",
        f"| Grad Clip | {TRAIN_CONFIG['GRAD_CLIP']} |",
        f"| Label Smooth | {TRAIN_CONFIG['LABEL_SMOOTH']} |",
        f"| Early Stop Patience | {TRAIN_CONFIG['EARLY_STOP_PATIENCE']} |",
        f"| Grad Accum Steps | {TRAIN_CONFIG['GRAD_ACCUM']} |",
        f"| EMA | {'ON' if TRAIN_CONFIG['EMA'] else 'OFF'} |",
        f"| DCT | {'ON' if CONFIG['DCT_ENABLED'] else 'OFF (FFT)'} |",
        f"| Baseline Mode | {'ON' if CONFIG['BASELINE_MODE'] else 'OFF'} |",
        f"| Dataset | {CONFIG['DATASET_ROOT']} |",
        "",
        "## Per-Epoch Metrics",
        "",
        "| Ep | TrLoss | VaLoss | TrAcc | VaAcc | F1 | AUC | LR | Gate | Time | Best |",
        "|---:|-------:|-------:|------:|------:|---:|----:|---:|-----:|-----:|:----:|",
    ]
    for i in range(len(history["val_auc"])):
        star = "★" if history["val_auc"][i] == max(history["val_auc"][: i + 1]) else ""
        lines.append(
            f"| {i + 1} | {history['train_loss'][i]:.4f} | {history['val_loss'][i]:.4f} | "
            f"{history['train_acc'][i]:.3f} | {history['val_acc'][i]:.3f} | "
            f"{history['val_f1'][i]:.3f} | {history['val_auc'][i]:.4f} | "
            f"{history['lr'][i]:.1e} | {history['gate'][i]:.3f} | {history['time'][i]:.0f}s | {star} |"  # noqa: E501
        )
    lines += [
        "",
        f"**Best Val AUC = {best_auc:.4f}**",
    ]
    return "\n".join(lines)


# ── Plots ────────────────────────────────────────────────────────


def _plot_training(history, run_id=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history["val_auc"]) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train", color="#2E5FA3")
    axes[0].plot(epochs, history["val_loss"], label="val", color="#E25B4A")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="BCE loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train", color="#2E5FA3")
    axes[1].plot(epochs, history["val_acc"], label="val", color="#E25B4A")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["val_auc"], color="#2FA36B")
    axes[2].axhline(0.85, color="green", linestyle="--", alpha=0.6)
    axes[2].axhline(0.75, color="orange", linestyle="--", alpha=0.4)
    axes[2].set(title="Validation AUC", xlabel="Epoch", ylabel="AUC")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    base = os.path.join(
        CONFIG["CHECKPOINT_DIR"],
        f"{run_id}_training_curves.png" if run_id else "training_curves.png",
    )
    plt.savefig(base, dpi=150, bbox_inches="tight")
    plt.close()
    log("TRAIN", f"Curves → {base}")


def _plot_gate_drift(gate_means, run_id=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(gate_means) + 1))
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, gate_means, color="#2E5FA3", marker="o", markersize=3)
    plt.axhline(0.5, color="black", linestyle="--", alpha=0.4)
    # Auto y-limits: the fixed 0.45–0.60 range clipped T4-20's gate (peak 0.605).
    plt.xlabel("Epoch")
    plt.ylabel("Mean spatial gate weight")
    plt.title("[B3] Fusion gate drift")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    base = os.path.join(
        CONFIG["CHECKPOINT_DIR"],
        f"{run_id}_fusion_gate_drift.png" if run_id else "fusion_gate_drift.png",
    )
    plt.savefig(base, dpi=150, bbox_inches="tight")
    plt.close()
    log("TRAIN", f"Gate drift → {base}")


# ── Main training entry ──────────────────────────────────────────


def _ckpt_paths(run_id):
    """Return (id_latest, id_best, latest, best) checkpoint paths for a run."""
    cdir = CONFIG["CHECKPOINT_DIR"]
    return (
        os.path.join(cdir, f"{run_id}.latest.pt"),
        os.path.join(cdir, f"{run_id}.best.pt"),
        os.path.join(cdir, "latest.pt"),
        os.path.join(cdir, "best_model.pt"),
    )


def train_model(resume=False, run_id=None):
    """Train DeepfakeDetector v7.0. Returns (model, history).
    run_id like 'T4-20' sets the base run name AND the epoch count."""
    section("TRAINING")

    # ── Resolve run name + epoch count from id ─────────────────────
    if run_id:
        stem, sep, maybe_ep = run_id.rpartition("-")
        if sep and maybe_ep.isdigit():
            run_id, n_epochs = stem, int(maybe_ep)
            TRAIN_CONFIG["EPOCHS"] = n_epochs
        else:
            run_id = run_id  # already the base name, keep config epochs
    else:
        run_id = os.environ.get("DD_RUN_ID", _RUN_ID)
    run_id = f"{run_id}-{TRAIN_CONFIG['EPOCHS']}"

    log(
        "TRAIN",
        f"Run id={run_id} | Device={DEVICE} | batch={TRAIN_CONFIG['BATCH_SIZE']} "
        f"accum={TRAIN_CONFIG['GRAD_ACCUM']} | "
        f"lr={TRAIN_CONFIG['LR']:.0e}→{TRAIN_CONFIG['MIN_LR']:.0e} | "
        f"epochs={TRAIN_CONFIG['EPOCHS']}",
    )

    torch.manual_seed(TRAIN_CONFIG["SEED"])
    np.random.seed(TRAIN_CONFIG["SEED"])
    random.seed(TRAIN_CONFIG["SEED"])

    if TRAIN_CONFIG.get("CUDNN_BENCHMARK", True):
        torch.backends.cudnn.benchmark = True
    if TRAIN_CONFIG.get("TF32_MATMUL", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from src.dataset import build_dataloaders

    model = DeepfakeDetector().to(DEVICE)
    log("TRAIN", f"Params: {count_params(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["LR"], weight_decay=0.01)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_auc": [],
        "val_f1": [],
        "lr": [],
        "gate": [],
        "time": [],
    }
    best_auc, best_epoch, patience = 0.0, -1, 0
    gate_means = []
    start_epoch = 0

    log_dir = _log_dir()
    os.makedirs(CONFIG["CHECKPOINT_DIR"], exist_ok=True)
    id_latest, id_best, ck_latest, ck_best = _ckpt_paths(run_id)

    if resume:
        # Prefer the id-keyed latest (T4-20.latest.pt), fall back to canonical latest.pt
        src = id_latest if os.path.exists(id_latest) else ck_latest
        if os.path.exists(src):
            start_epoch, old_hist, _, rng_state = _load_ckpt(model, optimizer, src)
            # Trim to the epochs actually completed so history and start_epoch agree.
            history = {k: list(old_hist.get(k, []))[:start_epoch] for k in history}
            # latest.pt's metrics hold THAT epoch's AUC, not the best — derive the
            # best from history so a worse epoch can't overwrite best_model.pt.
            best_auc, best_epoch, patience = _best_from_history(history)
            gate_means = list(history["gate"])
            rng_ok = _restore_rng(rng_state)
            log(
                "TRAIN",
                f"Resumed from {src} (epoch {start_epoch}, best AUC {best_auc:.4f} "
                f"@ ep {best_epoch}, patience {patience}, "
                f"RNG {'restored' if rng_ok else 'reseeded'})",
            )
        else:
            log(
                "TRAIN",
                f"WARNING: --resume but no {run_id}.latest.pt / latest.pt — starting fresh.",
            )

    train_loader, val_loader = build_dataloaders()

    try:
        for epoch_idx in range(start_epoch, TRAIN_CONFIG["EPOCHS"]):
            lr = _lr_at(epoch_idx)
            for g in optimizer.param_groups:
                g["lr"] = lr

            t0 = time.time()
            tr = _train_epoch(model, train_loader, optimizer, epoch_idx, TRAIN_CONFIG["EPOCHS"])
            va = _validate(model, val_loader)
            elapsed = time.time() - t0

            history["train_loss"].append(tr["loss"])
            history["val_loss"].append(va["loss"])
            history["train_acc"].append(tr["acc"])
            history["val_acc"].append(va["acc"])
            history["val_auc"].append(va["auc"])
            history["val_f1"].append(va["f1"])
            history["lr"].append(lr)
            history["gate"].append(va["gate"])
            history["time"].append(elapsed)
            gate_means.append(va["gate"])

            is_best = va["auc"] > best_auc
            if is_best:
                best_auc, best_epoch = va["auc"], epoch_idx + 1
                patience = 0
            else:
                patience += 1

            # Checkpoints — id-keyed + canonical mirrors.
            # Carry the whole validation dict, so `threshold` and `balanced_acc`
            # reach _save_ckpt. Rebuilding a four-key dict here used to drop them
            # and every checkpoint stored the 0.5 threshold fallback
            # (audit-fyp2.md §13.3).
            metrics = _epoch_metrics(va, epoch_idx + 1)
            _save_ckpt(model, optimizer, epoch_idx + 1, history, metrics, id_latest)
            _save_ckpt(model, optimizer, epoch_idx + 1, history, metrics, ck_latest)
            if is_best:
                # `auc` is the best-so-far by construction on a best epoch, but
                # state it explicitly so the field never depends on that.
                best_metrics = {**metrics, "auc": best_auc, "epoch": best_epoch}
                _save_ckpt(model, optimizer, epoch_idx + 1, history, best_metrics, id_best)
                _save_ckpt(model, optimizer, epoch_idx + 1, history, best_metrics, ck_best)

            row = {
                "epoch": epoch_idx + 1,
                "train_loss": tr["loss"],
                "val_loss": va["loss"],
                "train_acc": tr["acc"],
                "val_acc": va["acc"],
                "f1": va["f1"],
                "auc": va["auc"],
                "lr": lr,
                "gate": va["gate"],
                # Balanced accuracy and the calibrated threshold are what the
                # 1:5 class ratio makes meaningful; AUC alone hid the real-class
                # collapse in M1 (audit-fyp2.md §13.4).
                "balanced_acc": va["balanced_acc"],
                "threshold": va["threshold"],
                "time_s": elapsed,
                "best": is_best,
            }
            _write_jsonl(log_dir, run_id, row)

            _plot_training(history, run_id)
            _plot_gate_drift(gate_means, run_id)

            star = "★" if is_best else ""
            focus = "SPATIAL" if va["gate"] >= 0.5 else "FREQ"
            left = TRAIN_CONFIG["EPOCHS"] - epoch_idx - 1
            total_ep = TRAIN_CONFIG["EPOCHS"]
            auc_col = _score_col(va["auc"], 0.55)
            tr_loss, tr_acc = tr["loss"], tr["acc"]
            va_loss, va_acc, va_f1, va_auc, va_gate = (
                va["loss"],
                va["acc"],
                va["f1"],
                va["auc"],
                va["gate"],
            )
            va_bal = va["balanced_acc"]
            print(
                f"  {_paint('done', A.BOLD + A.CYAN)} "
                f"{_paint(f'ep {epoch_idx + 1:>2}/{total_ep}', A.BOLD + A.WHITE)} "
                f"{_paint(f'trL {tr_loss:.4f}', _loss_col(tr_loss))} "
                f"{_paint(f'trA {tr_acc * 100:5.1f}%', _score_col(tr_acc, 0.60))} "
                f"{_paint(f'vaL {va_loss:.4f}', _loss_col(va_loss))} "
                f"{_paint(f'vaA {va_acc * 100:5.1f}%', _score_col(va_acc, 0.60))} "
                # On a 1:5 split, accuracy flatters: always answering FAKE
                # already scores 83.3%. Show the balanced figure next to it.
                f"{_paint(f'bal {va_bal * 100:5.1f}%', _score_col(va_bal, 0.60))} "
                f"{_paint(f'F1 {va_f1:.3f}', A.GREEN)} "
                f"{_paint(f'AUC {va_auc:.4f}', auc_col)} "
                f"{_paint(f'gate {va_gate:.3f} ({focus})', A.MAGENTA)} "
                f"{_paint(f'lr {lr:.1e}', A.YELLOW)} "
                f"{_paint(f'{_fmt_dur(elapsed)}', A.CYAN)} "
                f"{_paint(f'best {best_auc:.4f} @ ep {best_epoch}', A.BOLD + A.GREEN if is_best else A.GRAY)} "  # noqa: E501
                f"{_paint(star, A.BOLD + A.YELLOW)} "
                f"{_paint(f'[{left} ep left]', A.GRAY)}",
                flush=True,
            )

            if patience >= TRAIN_CONFIG["EARLY_STOP_PATIENCE"]:
                log(
                    "TRAIN",
                    f"Early stop at epoch {epoch_idx + 1} "
                    f"(patience {patience}). Best AUC {best_auc:.4f} @ ep {best_epoch}.",
                )
                break
    except KeyboardInterrupt:
        # Graceful shutdown: persist resume state + partial plots/summary
        saved_epoch = len(history["val_auc"])
        log("TRAIN", "Ctrl-C — saving resume state ...")
        _save_ckpt(
            model,
            optimizer,
            saved_epoch,
            history,
            {"auc": best_auc, "epoch": saved_epoch},
            id_latest,
        )
        _save_ckpt(
            model,
            optimizer,
            saved_epoch,
            history,
            {"auc": best_auc, "epoch": saved_epoch},
            ck_latest,
        )
        if len(history["val_auc"]) > 0:
            _plot_training(history, run_id)
            _plot_gate_drift(gate_means, run_id)
            md = _summarise_run(run_id, best_auc, history)
            with open(os.path.join(log_dir, f"{run_id}.md"), "w", encoding="utf-8") as f:
                f.write(md + "\n")
        print(
            f"  ✋ Interrupted after {saved_epoch} epoch(s). Best so far = "
            f"{best_auc:.4f} @ ep {best_epoch}.",
            flush=True,
        )
        print(
            f"  Resume with: python train.py --resume {run_id.split('-')[0]}-"
            f"{TRAIN_CONFIG['EPOCHS']}",
            flush=True,
        )
        log("TRAIN", f"INTERRUPTED. State saved → {id_latest}")
        return model, history

    _plot_training(history, run_id)
    _plot_gate_drift(gate_means, run_id)

    md = _summarise_run(run_id, best_auc, history)
    md_path = os.path.join(log_dir, f"{run_id}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    log("TRAIN", f"Run summary → {md_path}")

    log("TRAIN", f"DONE. Best Val AUC = {best_auc:.4f} @ epoch {best_epoch}")
    print(
        f"  ✓ RUN COMPLETE — best Val AUC {best_auc:.4f} @ ep {best_epoch} | "
        f"final val acc {history['val_acc'][-1]:.3f} | F1 {history['val_f1'][-1]:.3f} | "
        f"gate {history['gate'][-1]:.3f} | epochs ran {len(history['val_auc'])}/{TRAIN_CONFIG['EPOCHS']}",  # noqa: E501
        flush=True,
    )
    return model, history
