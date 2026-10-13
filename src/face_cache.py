"""Face cache v2: RGB uint8 crops + per-frame metadata, no padding stored.

Why a second format (audit-fyp2.md §4.1, §4.2, FIX_PLAN P2.2):
  * v1 (cache/) stores BGR float32 (N,3,224,224), ~5.7 MB per video, and
    pads short clips by repeating the last frame (25% of DFD clips).
  * v2 (cache_v2/) stores RGB uint8 (k,3,224,224), k <= SAMPLE_COUNT, plus the
    frame indices and detection scores actually used. When a sampled frame has
    no face, neighbouring frames (idx±1 … idx±RETRY_WINDOW) are tried first.

The v1 cache and src/preprocess.py are left untouched so every FYP-1 result
(T4-20, website) still reproduces exactly.

The detector here is SCRFD detection-only with a bounded ONNX Runtime thread
pool, so several extraction processes can share the CPU. Crop geometry
(bbox clamp, cv2.resize) matches src/preprocess.extract_faces.
"""

import hashlib
import os
import time

import cv2
import numpy as np
import torch

from src.config import CONFIG

CACHE_VERSION = 2
RETRY_WINDOW = 5

_DETECTOR = None


def _detector_model_path():
    root = os.path.expanduser(os.path.join("~", ".insightface", "models", "buffalo_l"))
    path = os.path.join(root, "det_10g.onnx")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"SCRFD model not found: {path}. Run the legacy pipeline once "
            f"(python infer.py ...) so InsightFace downloads buffalo_l."
        )
    return path


def get_detector(threads=None):
    """Process-local SCRFD detector (created on first use)."""
    global _DETECTOR
    if _DETECTOR is None:
        import onnxruntime
        from insightface.model_zoo.scrfd import SCRFD

        if threads:
            cv2.setNumThreads(1)  # parallelism comes from the worker processes
            torch.set_num_threads(1)

        opts = onnxruntime.SessionOptions()
        if threads:
            opts.intra_op_num_threads = int(threads)
            opts.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(
            _detector_model_path(), opts, providers=["CPUExecutionProvider"]
        )
        det = SCRFD(session=session)
        det.prepare(0, input_size=CONFIG["DET_SIZE"])
        _DETECTOR = det
    return _DETECTOR


def _candidate_indices(target, total, window):
    """target, target+1, target-1, target+2, … within [0, total)."""
    out = [target]
    for d in range(1, window + 1):
        for idx in (target + d, target - d):
            if 0 <= idx < total:
                out.append(idx)
    return out


def _best_face(detector, frame, threshold):
    bboxes, _ = detector.detect(frame, threshold=threshold, max_num=0, metric="default")
    if bboxes is None or bboxes.shape[0] == 0:
        return None
    best = bboxes[int(np.argmax(bboxes[:, 4]))]
    return best[:4], float(best[4])


def _crop_rgb(frame, bbox, size):
    """Same clamp + resize as src/preprocess.extract_faces, then BGR→RGB."""
    h0, w0 = frame.shape[:2]
    x1 = int(np.clip(bbox[0], 0, w0 - 1))
    y1 = int(np.clip(bbox[1], 0, h0 - 1))
    x2 = int(np.clip(bbox[2], x1 + 1, w0))
    y2 = int(np.clip(bbox[3], y1 + 1, h0))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    crop = cv2.resize(crop, size)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), [x1, y1, x2, y2]


def extract_video(
    video_path, source=None, sample_count=None, retry_window=RETRY_WINDOW, threads=None
):
    """
    Extract up to `sample_count` RGB uint8 face crops from one video.

    Returns the cache-v2 dict (always, even when no face is found — then
    n_detected = 0 and frames has shape (0, 3, H, W)).
    """
    sample_count = sample_count or CONFIG["SAMPLE_COUNT"]
    threshold = CONFIG["DETECTION_THRESHOLD"]
    size = CONFIG["FACE_SIZE"]
    detector = get_detector(threads)
    t0 = time.time()

    rec = {
        "version": CACHE_VERSION,
        "channel_order": "RGB",
        "dtype": "uint8",
        "source": source or os.path.basename(video_path),
        "sample_count": sample_count,
        "retry_window": retry_window,
        "frame_idx": [],
        "target_idx": [],
        "det_score": [],
        "bbox": [],
        "n_detected": 0,
        "retries": 0,
        "errors": [],
    }

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rec.update(
        total_frames=total,
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        resolution=f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
    )
    crops, used = [], set()

    if total > 0:
        targets = np.linspace(0, total - 1, sample_count, dtype=int)
        for target in targets:
            for attempt, idx in enumerate(_candidate_indices(int(target), total, retry_window)):
                if idx in used:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                try:
                    found = _best_face(detector, frame, threshold)
                except Exception as e:  # keep going; record it
                    rec["errors"].append(f"frame {idx}: {e}")
                    continue
                if found is None:
                    continue
                crop, box = _crop_rgb(frame, found[0], size)
                if crop is None:
                    continue
                crops.append(crop)
                used.add(idx)
                rec["frame_idx"].append(idx)
                rec["target_idx"].append(int(target))
                rec["det_score"].append(round(found[1], 4))
                rec["bbox"].append(box)
                rec["retries"] += int(attempt > 0)
                break
    else:
        rec["errors"].append("cannot read frame count")
    cap.release()

    if crops:
        arr = np.stack(crops)  # (k, H, W, 3) RGB
        rec["frames"] = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    else:
        rec["frames"] = torch.zeros((0, 3, size[1], size[0]), dtype=torch.uint8)
    rec["n_detected"] = len(crops)
    rec["extract_time"] = round(time.time() - t0, 2)
    return rec


# ── Loading ──────────────────────────────────────────────────────


def load_frames(path: str) -> tuple[torch.Tensor, dict]:
    """Load a v2 cache file → (frames float32 (k,3,H,W) in [0,1] RGB, record)."""
    rec = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(rec, dict) or rec.get("version") != CACHE_VERSION:
        raise RuntimeError(f"{path} is not a face-cache v{CACHE_VERSION} file")
    frames = rec["frames"]
    if frames.dtype != torch.uint8 or frames.dim() != 4 or frames.shape[1] != 3:
        raise RuntimeError(f"{path}: unexpected frames {frames.dtype} {tuple(frames.shape)}")
    return frames.float() / 255.0, rec


def to_sequence(frames: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Zero-pad or truncate (k,3,H,W) frames to `length`.
    Returns (frames (length,3,H,W), pad_mask (length,) bool, True = padded).
    Padded frames are zeros, never repeated real frames (audit §4.2).
    """
    k = frames.shape[0]
    mask = torch.zeros(length, dtype=torch.bool)
    if k >= length:
        return frames[:length], mask
    pad = frames.new_zeros((length - k,) + tuple(frames.shape[1:]))
    mask[k:] = True
    return torch.cat([frames, pad], dim=0), mask


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file, read in ``chunk``-byte blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ── Worker entry point (ProcessPoolExecutor) ─────────────────────


def extract_to_file(job):
    """job = (video_path, rel_key, out_path, threads). Returns a manifest entry."""
    video_path, rel, out_path, threads = job
    try:
        rec = extract_video(video_path, source=rel, threads=threads)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp"
        torch.save(rec, tmp)
        os.replace(tmp, out_path)
        return rel, {
            "n_detected": rec["n_detected"],
            "retries": rec["retries"],
            "total_frames": rec["total_frames"],
            "resolution": rec["resolution"],
            "extract_time": rec["extract_time"],
            "errors": len(rec["errors"]),
            "size": os.path.getsize(out_path),
            "sha256": sha256_file(out_path),
        }
    except Exception as e:  # reported, not fatal
        return rel, {"failed": f"{type(e).__name__}: {e}"}
