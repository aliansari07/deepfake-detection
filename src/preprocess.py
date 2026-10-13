"""Extracts and normalises face crops from video files.
Import: from src.preprocess import extract_faces
"""

import os
import time

import cv2
import numpy as np
import torch

from src.config import CONFIG
from src.detector import get_face_app
from src.logger import debug, log, tensor_stats


def extract_faces(
    video_path: str, sample_count: int | None = None
) -> tuple[torch.Tensor | None, dict]:
    """
    Extract face crops from a video file.

    Pipeline:
        1. Open video → sample N evenly-spaced frames.
        2. Run InsightFace SCRFD on each frame.
        3. Crop the highest-confidence face.
        4. Resize to 224×224, normalise to [0, 1].
        5. Return (N, 3, 224, 224) tensor.

    Args:
        video_path   : Path to .mp4 file.
        sample_count : Override CONFIG["SAMPLE_COUNT"] if set.

    Returns:
        faces (Tensor | None) : (N, 3, 224, 224) or None if no faces found.
        meta  (dict)          : Extraction metadata.
    """
    if sample_count is None:
        sample_count = CONFIG["SAMPLE_COUNT"]

    meta = {"video": os.path.basename(video_path), "frames": [], "errors": []}

    if not os.path.exists(video_path):
        log("PREPROCESS", f"ERROR: File not found — {video_path}")
        return None, meta

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        log("PREPROCESS", "ERROR: Cannot read video metadata.")
        cap.release()
        return None, meta

    meta.update(
        {
            "total_frames": total_frames,
            "fps": fps,
            "resolution": f"{w}x{h}",
            "duration": f"{total_frames / max(fps, 1):.1f}s",
        }
    )

    log(
        "PREPROCESS",
        f"{meta['video']} ({w}x{h}, {total_frames}f, {fps:.0f}fps, {meta['duration']})",
    )

    frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
    faces_extracted = []
    extract_start = time.time()

    for frame_idx in frame_indices:
        t0 = time.time()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            meta["errors"].append(f"Frame {frame_idx}: read failed")
            continue

        try:
            faces = get_face_app().get(frame)
            faces = [f for f in faces if f.det_score >= CONFIG["DETECTION_THRESHOLD"]]
        except Exception as e:
            meta["errors"].append(f"Frame {frame_idx}: {str(e)}")
            continue

        dt = time.time() - t0

        if faces:
            face = max(faces, key=lambda f: f.det_score)
            box = face.bbox.astype(int)
            conf = float(face.det_score)
            kps = face.kps.astype(int)
            # Clamp bbox to frame bounds — SCRFD boxes can poke outside the
            # frame (e.g. head cut off at the top); a raw slice with a
            # negative index would wrap around and yield an EMPTY crop,
            # silently dropping otherwise-detectable videos.
            h0, w0 = frame.shape[:2]
            x1 = int(np.clip(face.bbox[0], 0, w0 - 1))
            y1 = int(np.clip(face.bbox[1], 0, h0 - 1))
            x2 = int(np.clip(face.bbox[2], x1 + 1, w0))
            y2 = int(np.clip(face.bbox[3], y1 + 1, h0))
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                face_crop = cv2.resize(face_crop, CONFIG["FACE_SIZE"])
                tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float() / 255.0
                faces_extracted.append(tensor)
                meta["frames"].append(
                    {
                        "idx": int(frame_idx),
                        "conf": conf,
                        "time": dt,
                        "box": box.tolist(),
                        # The clamped crop region actually sliced out of the frame — not the raw
                        # SCRFD bbox above, which can poke outside the frame. src/explain.py maps
                        # landmarks through this box into the resized face crop, so it needs the
                        # box that matches what extract_faces() actually cropped.
                        "crop_box": [x1, y1, x2, y2],
                        "landmarks": kps.tolist(),
                    }
                )
                debug(
                    f"Frame {frame_idx:>4d}: conf={conf:.3f} | "
                    f"crop={box[2] - box[0]}x{box[3] - box[1]}px | {dt:.2f}s"
                )
        else:
            debug(f"Frame {frame_idx:>4d}: no face | {dt:.2f}s")

    cap.release()
    meta["extract_time"] = time.time() - extract_start

    log(
        "PREPROCESS",
        f"Extracted {len(faces_extracted)}/{sample_count} faces in {meta['extract_time']:.1f}s",
    )

    if not faces_extracted:
        log("PREPROCESS", "WARNING: No faces extracted.")
        return None, meta

    result = torch.stack(faces_extracted)
    debug(tensor_stats(result, "Face batch"))
    return result, meta
