"""Turns the model's raw signals (verdict, confidence, gate weights, and
the Grad-CAM heatmap) into a plain-language explanation for the website.

Two stages, kept separate so the language layer can never invent evidence
the model didn't produce:
  1. analyze_regions()  — grounds the heatmap's hottest cell against SCRFD's
     5-point landmarks, so "eyes" / "mouth" etc. is read off real geometry,
     not guessed. Returns {"available": False} rather than a label it can't
     support (e.g. landmarks missing).
  2. explain_verdict()  — composes the sentence from those grounded facts.
     Two providers:
       DD_EXPLAIN_PROVIDER=local  (default) — _local_explanation(), a deterministic
           template. No network, no key, cannot hallucinate — the demo-day fallback.
       DD_EXPLAIN_PROVIDER=claude — Custom_AI_Explainer_Agent(), a live call to the
           Anthropic API. Needs ANTHROPIC_API_KEY set in the environment; optionally
           DD_EXPLAIN_MODEL to override the model (default claude-haiku-4-5-20251001).
           If the call fails for any reason (no key, no network, rate limit) it logs a
           warning and falls back to the local template rather than showing nothing.

Import: from src.explain import analyze_regions, explain_verdict
"""

import json
import os

import numpy as np

# SCRFD's 5-point order (see src/preprocess.py kps): left eye, right eye,
# nose tip, left mouth corner, right mouth corner.
_LANDMARK_NAMES = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]

# Plain-English name for each region — used no matter the verdict.
REGION_LABELS = {
    "eyes": "the eyes",
    "nose": "the nose bridge",
    "mouth": "the mouth",
    "forehead": "the forehead and hairline",
    "jaw": "the jawline and chin",
    "cheek": "the cheek",
    "center": "the center of the face",
}

# Why that region matters FOR A FAKE READING. Only ever appended when the verdict
# is actually FAKE/UNCERTAIN — pairing this with a REAL verdict is what produced
# the confusing "seam line... but nothing pushed it toward FAKE" sentence.
REGION_FAKE_REASONS = {
    "eyes": "fake videos often struggle to keep blinking and eye movement looking natural",
    "nose": "this is a spot where a swapped face can leave a visible seam",
    "mouth": "fake videos often struggle to match lip movement and teeth correctly",
    "forehead": "this is where a face swap can visibly meet the original scalp",
    "jaw": "this is a common edge where two faces get blended together",
    "cheek": "lighting and skin texture can look slightly off here in a fake video",
    "center": "this area can show mismatched texture when two faces are blended",
}


def _map_point_to_cell(px, py, crop_box, face_size, grid):
    """Map an (x, y) point in ORIGINAL frame coordinates into a (row, col)
    cell of the grid x grid Grad-CAM, through the same crop + resize that
    extract_faces() applied to produce the face tensor."""
    x1, y1, x2, y2 = crop_box
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    fx = (px - x1) / w * face_size[0]
    fy = (py - y1) / h * face_size[1]
    col = int(min(max(fx // (face_size[0] / grid), 0), grid - 1))
    row = int(min(max(fy // (face_size[1] / grid), 0), grid - 1))
    return row, col


def _landmark_cells(landmarks, crop_box, face_size, grid):
    return {
        name: _map_point_to_cell(lx, ly, crop_box, face_size, grid)
        for name, (lx, ly) in zip(_LANDMARK_NAMES, landmarks)
    }


def _region_for_cell(cell, cells):
    """Label the hottest CAM cell using distance to the real landmark cells,
    falling back to coarse position (forehead/jaw/cheek) only when it isn't
    close to any named feature."""
    row, col = cell

    def dist(c):
        return (c[0] - row) ** 2 + (c[1] - col) ** 2

    nearest_name, nearest_dist = min(
        (
            ("eyes", min(dist(cells["left_eye"]), dist(cells["right_eye"]))),
            ("nose", dist(cells["nose"])),
            ("mouth", min(dist(cells["left_mouth"]), dist(cells["right_mouth"]))),
        ),
        key=lambda t: t[1],
    )
    if nearest_dist <= 1.1:
        return nearest_name

    eye_row = min(cells["left_eye"][0], cells["right_eye"][0])
    mouth_row = max(cells["left_mouth"][0], cells["right_mouth"][0])
    nose_col = cells["nose"][1]
    if row < eye_row:
        return "forehead"
    if row > mouth_row:
        return "jaw"
    if abs(col - nose_col) >= 2:
        return "cheek"
    return "center"


def analyze_regions(cam, frames_meta, face_size=(224, 224)):
    """
    Ground the Grad-CAM heatmap's hottest cell, per frame, against SCRFD's
    landmarks for that frame.

    Args:
        cam         : (N, 1, grid, grid) raw Grad-CAM tensor (pre-upsampling,
                      pre-normalisation — only the argmax position is used).
        frames_meta : list of per-frame dicts aligned 1:1 with cam's N, each
                      carrying "crop_box" [x1,y1,x2,y2] and "landmarks"
                      (5 x [x,y]), both in ORIGINAL frame pixel coordinates
                      (src/preprocess.py).
        face_size   : (w, h) the crop was resized to before the model saw it.

    Returns:
        {"available": False} if landmarks/crop_box are missing or
        misaligned — never guesses a region it cannot ground — else
        {"available": True, "per_frame": [region, ...], "dominant": region,
         "consistency": fraction of frames whose hottest cell was `dominant`}.
    """
    if not frames_meta or cam.shape[0] != len(frames_meta):
        return {"available": False}
    if any("crop_box" not in f or "landmarks" not in f for f in frames_meta):
        return {"available": False}

    grid = cam.shape[-1]
    cam_np = cam.detach().cpu().numpy()
    labels = []
    for i, f in enumerate(frames_meta):
        heat = cam_np[i, 0]
        row, col = np.unravel_index(np.argmax(heat), heat.shape)
        cells = _landmark_cells(f["landmarks"], f["crop_box"], face_size, grid)
        labels.append(_region_for_cell((row, col), cells))

    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    dominant = max(counts, key=counts.get)

    return {
        "available": True,
        "per_frame": labels,
        "dominant": dominant,
        "consistency": counts[dominant] / len(labels),
    }


def _confidence_phrase(confidence):
    if confidence >= 0.9:
        return "very confident"
    if confidence >= 0.75:
        return "fairly confident"
    if confidence >= 0.6:
        return "somewhat confident"
    return "not very confident"


def _local_explanation(facts: dict) -> str:
    """The default generator: a deterministic template over grounded facts.
    No network call, no hallucination risk — every claim traces back to a
    number the pipeline actually computed.

    REAL and FAKE/UNCERTAIN get DIFFERENT region wording on purpose: naming a
    region as "a common face-swap seam" and then saying it didn't matter, in the
    same breath, reads as the video being suspicious even when the verdict is
    REAL — a non-technical reader stops at the scary half of the sentence."""
    verdict = facts.get("verdict", "UNCERTAIN")
    confidence = float(facts.get("confidence", 0.5))
    pct = confidence * 100

    sentences = []
    if verdict == "REAL":
        sentences.append(
            f"This video looks REAL — the system is {_confidence_phrase(confidence)} "
            f"({pct:.0f}%) that it has not been manipulated."
        )
    elif verdict == "FAKE":
        sentences.append(
            f"This video looks FAKE — the system is {_confidence_phrase(confidence)} "
            f"({pct:.0f}%) that it has been manipulated."
        )
    else:
        sentences.append(
            f"The system can't confidently call this one ({pct:.0f}%) — it's too "
            "close to the line between real and fake to trust either way."
        )

    region = facts.get("region") or {}
    if region.get("available"):
        dominant = region["dominant"]
        per_frame = region.get("per_frame", [])
        n_frames = len(per_frame)
        n_dominant = round(region.get("consistency", 0.0) * n_frames)
        label = REGION_LABELS.get(dominant, "the face")
        if verdict == "REAL":
            sentences.append(
                f"It looked most closely at {label} (in {n_dominant} of the "
                f"{n_frames} frames it checked), and found nothing unusual there."
            )
        else:
            reason = REGION_FAKE_REASONS.get(dominant, "")
            tail = f" — {reason}" if reason else ""
            sentences.append(
                f"The clearest sign was around {label}, showing up in "
                f"{n_dominant} of the {n_frames} frames it checked{tail}."
            )
    else:
        sentences.append(
            "It wasn't able to pin this down to one specific part of the face, "
            "so the highlighted areas in the image should be read as a general "
            "guide, not an exact spot."
        )

    if facts.get("has_gate", False):
        s_pct = float(facts.get("spatial_weight", 0.5)) * 100
        f_pct = float(facts.get("freq_weight", 0.5)) * 100
        lead = "what the image looks like" if s_pct >= f_pct else "hidden pixel patterns"
        sentences.append(
            f"To decide, it leaned more on {lead} than on the other check it runs "
            f"({s_pct:.0f}% vs {f_pct:.0f}%)."
        )

    sentences.append(
        "This is an AI estimate based on patterns it learned during training — "
        "not a guaranteed or certified result, so use your own judgement too."
    )
    return " ".join(sentences)


_CLAUDE_SYSTEM_PROMPT = """\
You explain a deepfake detector's verdict on a video to a non-technical viewer.
You are given the detector's own computed signals as JSON: a verdict, a confidence
score, spatial/frequency fusion-gate weights (when the model has a gate), and — when
available — which facial region the Grad-CAM heatmap focused on and how consistently
across the sampled frames.

Rules:
- Use ONLY the facts in the JSON. Never invent an artifact, a facial region, or a
  number that isn't there. If region.available is false, do not name any facial
  region at all — describe the evidence in general terms instead.
- Write 2-4 short plain sentences for someone with NO technical background at all.
  No jargon: never say "heatmap", "activation", "fusion gate", "spatial/frequency
  branch", or similar ML terms. Say what happened in everyday words instead.
- Lead with the verdict and confidence, in plain words (e.g. "looks real", "looks
  fake", "can't confidently tell").
- If verdict is REAL: when you mention a region, say the system looked closely at it
  and found nothing unusual — do NOT describe that region as a typical fake-video
  problem spot. Pairing "common face-swap seam" language with a REAL verdict reads
  as the video being suspicious even though it isn't; that is the one thing you must
  never do.
- If verdict is FAKE or UNCERTAIN: you may say why that region is a common sign of
  manipulation, in one plain clause.
- If a gate is available, describe which kind of evidence it leaned on using plain
  words (e.g. "what the image looks like" vs "hidden pixel patterns"), not the terms
  "spatial" or "frequency".
- End with one short sentence noting this is an AI estimate, not a certified result.
- Plain prose only — no markdown, no headers, no bullet points, no restating the JSON.
"""


def Custom_AI_Explainer_Agent(facts: dict) -> str:
    """Live provider: sends `facts` to the Anthropic Messages API with the system
    prompt above and returns its prose explanation.

    Requires ANTHROPIC_API_KEY in the environment (the SDK reads it directly; no key
    is ever hardcoded here). DD_EXPLAIN_MODEL overrides the model. Raises on any
    failure (missing key, network, rate limit) — explain_verdict() is what decides to
    fall back, this function stays honest about whether the call actually worked.
    """
    import anthropic

    model = os.environ.get("DD_EXPLAIN_MODEL", "claude-haiku-4-5-20251001")
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=220,
        system=_CLAUDE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(facts)}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def explain_verdict(facts: dict) -> str:
    """Public entry point. `facts` is plain JSON — the same payload any provider
    needs. DD_EXPLAIN_PROVIDER picks which one runs (see module docstring)."""
    provider = os.environ.get("DD_EXPLAIN_PROVIDER", "local")
    if provider == "local":
        return _local_explanation(facts)
    if provider == "claude":
        try:
            return Custom_AI_Explainer_Agent(facts)
        except Exception as e:
            from src.logger import log

            log(
                "EXPLAIN",
                f"WARNING: Custom_AI_Explainer_Agent failed ({e}) "
                "— falling back to the local template.",
            )
            return _local_explanation(facts)
    raise RuntimeError(
        f"Unknown DD_EXPLAIN_PROVIDER={provider!r}. `facts` is JSON-serialisable "
        "— add a branch in src/explain.explain_verdict() that hands it to that "
        "provider and returns its text."
    )
