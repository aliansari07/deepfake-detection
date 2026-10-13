"""Grad-CAM explainability on the spatial EfficientNet branch [6.7].

Hooks the final conv block (bn2 → 1280×7×7 feature map) of the
spatial stream, weights its channels by the GAP of the gradients of
the fake logit, and overlays the upsampled heatmap (JET colormap)
onto each face frame. Saves a per-video grid as a PNG.

Import: from src.gradcam import generate_gradcam
"""

import os

import cv2
import numpy as np
import torch

from src.config import CONFIG
from src.logger import log

CAM_BACKBONE = "spatial"  # [6.7] spatial stream only
CAM_TARGET_LAYER = "bn2"  # final conv block before global pooling


def _hook_forward(module, input, output):
    module._activations = output.detach()


def _hook_backward(module, grad_input, grad_output):
    module._gradients = grad_output[0].detach()


def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    raw_frames: torch.Tensor,
    video_name: str,
    meta: dict | None = None,
    frames_meta: list | None = None,
    region_out: dict | None = None,
) -> str | None:
    """
    Generate a Grad-CAM overlay grid for a single video.

    Args:
        model       : DeepfakeDetector (eval mode).
        input_tensor: (N, 3, 224, 224) frames actually passed to the model
                      (post-augmentation, on DEVICE).
        raw_frames  : (N, 3, 224, 224) original face crops (CPU, [0,1]).
        video_name  : Base name of the video (without extension).
        meta        : Optional dict of verdict/scores embedded into the PNG
                      (video_name, verdict, prob, confidence, logit, s_weight,
                      faces, faces_total). Defaults are used when omitted.
        frames_meta : Optional per-frame list (meta["frames"] from
                      src.preprocess.extract_faces, aligned 1:1 with the N
                      dimension above) — crop_box + landmarks, used to ground
                      the heatmap against real facial geometry. Only read
                      when region_out is given.
        region_out  : Optional dict; filled in-place with
                      src.explain.analyze_regions()'s result when given, so
                      callers that don't ask for it (heatmap_demo.py) pay
                      nothing extra and keep the plain path return value.

    Returns:
        str : Path to the saved PNG grid (or None on failure).
    """
    target = model.spatial.bn2
    target._activations = None
    target._gradients = None
    fh = target.register_forward_hook(_hook_forward)
    bh = target.register_full_backward_hook(_hook_backward)

    try:
        flat = input_tensor  # (N,3,224,224)
        logit, _ = model(flat)

        # PLAN task E1. The model emits ONE logit, and sigmoid(logit) is p(fake), so
        # logit.backward() always asks "what pushed this toward FAKE". On a REAL verdict the
        # caption then said REAL while the heatmap highlighted the evidence for the opposite
        # conclusion — an explanation of a decision the model did not make.
        #
        # Backpropagating -logit for a REAL verdict asks "what pushed this toward REAL", which is
        # what the caption claims. UNCERTAIN keeps the fake direction, since the interesting
        # question there is what fake-evidence was found at all.
        verdict = str((meta or {}).get("verdict", "FAKE")).upper()
        explain_fake = verdict != "REAL"
        target.zero_grad()
        (logit if explain_fake else -logit).backward()

        acts = target._activations  # (N, 1280, 7, 7)
        grads = target._gradients  # (N, 1280, 7, 7)

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (N,1280,1,1)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (N,1,7,7)
        cam = torch.relu(cam)

        if region_out is not None:
            from src.explain import analyze_regions

            region_out.update(analyze_regions(cam, frames_meta or []))

        grid = _normalise_grid(cam, raw_frames, cam.device)
        grid = _annotate_grid(grid, meta or {})
        os.makedirs(CONFIG["GRADCAM_DIR"], exist_ok=True)
        out_path = os.path.join(CONFIG["GRADCAM_DIR"], f"{video_name}_gradcam.png")
        _save_grid(grid, out_path)
        log("GRADCAM", f"saved → {out_path}")
        return out_path
    finally:
        fh.remove()
        bh.remove()
        # Drop parameter grads from logit.backward(): a model reused across
        # requests (website backend) would otherwise keep them alive.
        model.zero_grad(set_to_none=True)


_VERDICT_COLORS = {  # (fill, text)
    "FAKE": ((200, 30, 30), (255, 255, 255)),
    "REAL": ((40, 145, 60), (255, 255, 255)),
    "UNCERTAIN": ((225, 170, 0), (60, 40, 0)),
}


def _load_font(size, bold=False):
    """Windows Arial with a DejaVu/fallback chain — no new deps."""
    from PIL import ImageFont

    candidates = []
    if os.name == "nt":
        fonts_dir = os.environ.get("WINDIR", r"C:\Windows") + r"\Fonts"
        name = "arialbd.ttf" if bold else "arial.ttf"
        candidates.append(os.path.join(fonts_dir, name))
    candidates.append("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)
    except Exception:
        from PIL import ImageFont as _IF

        return _IF.load_default()


def _text_w(draw, text, font):
    return draw.textlength(text, font=font)


def _truncate(draw, text, font, max_w, ellipsis="\u2026"):
    """Clip text to max width with an ellipsis (never overflows)."""
    if _text_w(draw, text, font) <= max_w:
        return text
    n = len(text)
    while n > 1 and _text_w(draw, text[:n] + ellipsis, font=font) > max_w:
        n -= 1
    return text[:n] + ellipsis


def _fit_font(draw, text, font, max_w, bold=False):
    """Shrink font (floor 9) until text fits max_w."""
    size = font.size
    while size > 9 and _text_w(draw, text, font=font) > max_w:
        size -= 1
        font = _load_font(size, bold=bold)
    return font


def _jet_rgb01(t):
    """JET colormap color for t in [0,1], returned as an RGB tuple."""
    t = max(0.0, min(1.0, t))
    heat = np.uint8(round(255 * t))
    bgr = cv2.applyColorMap(np.uint8([[heat]]), cv2.COLORMAP_JET)[0, 0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def _annotate_grid(grid, meta):
    """Wrap the CAM grid with a verdict header, gate-split bar + JET color key.

    Header: title/video name, verdict box (red=FAKE / green=REAL / amber=UNSURE),
    confidence + logit, and a SPATIAL-vs-FREQ split bar.
    Footer: explicit COLOR KEY  (blue = no positive FAKE evidence,
    red = regions supporting the captioned verdict — the FAKE score for a FAKE or
    UNCERTAIN verdict, the REAL score for a REAL one). All text is width-guarded.
    """
    from PIL import Image, ImageDraw

    PAD = 14
    HDR = 150  # header band (px)
    FT = 66  # footer band (px)
    g_h, g_w, _ = grid.shape
    canvas = np.full((HDR + g_h + FT, g_w, 3), 250, dtype=np.uint8)
    canvas[HDR : HDR + g_h, :, :] = grid

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    video = meta.get("video_name", "video")
    verdict = meta.get("verdict", "N/A")
    prob = float(meta.get("prob", 0.0))
    conf = float(meta.get("confidence", prob))
    logit = float(meta.get("logit", 0.0))
    s_w = float(meta.get("s_weight", 0.5))
    f_w = 1.0 - s_w
    faces = meta.get("faces", "?")
    total = meta.get("faces_total", 10)

    font_title = _load_font(18, bold=True)
    font_vname = _load_font(12, bold=False)
    font_verdict = _load_font(28, bold=True)
    font_small = _load_font(12, bold=False)
    font_smallb = _load_font(12, bold=True)

    # ── Row 1: title + video name (truncated so it never overlaps) ──
    title = "DEEPFAKE DETECTION \u2014 GRAD-CAM"
    draw.text((PAD, 8), title, fill=(15, 15, 15), font=font_title)
    vname = os.path.basename(video)
    vmax = max(g_w - PAD - _text_w(draw, title, font=font_title) - 24, 80)
    draw.text(
        (g_w - PAD, 10),
        _truncate(draw, vname, font_vname, vmax),
        fill=(95, 95, 95),
        font=font_vname,
        anchor="ra",
    )

    # ── Row 2: verdict box + confidence/logit line ──
    fill, text_col = _VERDICT_COLORS.get(verdict, ((90, 90, 90), (255, 255, 255)))
    vw = _text_w(draw, verdict, font=font_verdict)
    bx0, by0 = PAD, 36
    bh, bw = 62, vw + 30
    draw.rounded_rectangle([bx0, by0, bx0 + bw, by0 + bh], radius=8, fill=fill)
    draw.text((bx0 + 15, by0 + bh // 2), verdict, fill=text_col, font=font_verdict, anchor="lm")

    info1 = (
        f"{verdict.lower()} confidence {conf * 100:.1f}%"
        f"   \u00b7   logit {logit:+.3f}"
        f"   \u00b7   faces {faces}/{total}"
    )
    ix = bx0 + bw + 16
    avail = g_w - ix - PAD
    if avail > 60:
        i_font = _fit_font(draw, info1, font_small, avail)
        draw.text((ix, by0 + bh // 2), info1, fill=(25, 25, 25), font=i_font, anchor="lm")

    # ── Row 3: fusion gate split bar (spatial vs freq trusted) ──
    # A spatial-only checkpoint has no gate at all. Drawing "spatial 100% / freq 0%" for it
    # would claim a gate weighed two streams and chose one — the same false reading that had
    # to be removed from the website (PLAN F2a). meta["has_gate"] is False for those models.
    gy = 106
    if not meta.get("has_gate", True):
        draw.text(
            (PAD, gy),
            "architecture  —  spatial backbone only; no fusion gate in this model",
            fill=(35, 35, 35),
            font=font_smallb,
        )
    else:
        draw.text(
            (PAD, gy),
            f"fusion gate  \u2014  spatial trusted {s_w * 100:.1f}%   |   "
            f"freq trusted {f_w * 100:.1f}%",
            fill=(35, 35, 35),
            font=font_smallb,
        )

        track_y0, th = gy + 15, 10
        x0 = PAD
        x1 = g_w - PAD
        sp_w = int((x1 - x0) * s_w)
        draw.rectangle(
            [x0, track_y0, x0 + sp_w, track_y0 + th], fill=(51, 201, 255)
        )  # cyan = spatial
        draw.rectangle(
            [x0 + sp_w, track_y0, x1, track_y0 + th], fill=(245, 193, 60)
        )  # amber = freq
        draw.rectangle([x0, track_y0, x1, track_y0 + th], outline=(120, 120, 120), width=1)

        draw.text(
            (x0, track_y0 + th + 3),
            f"SPATIAL {s_w * 100:.1f}%",
            fill=(0, 120, 180),
            font=font_smallb,
        )
        draw.text(
            (x1, track_y0 + th + 3),
            f"FREQ {f_w * 100:.1f}%",
            fill=(175, 120, 0),
            font=font_smallb,
            anchor="ra",
        )

    # ── Footer: explicit COLOR KEY ─────────────────────────────
    fy = HDR + g_h

    draw.text(
        (PAD, fy + 2),
        "COLOR KEY \u2014 heatmap = spatial-branch focus",
        fill=(60, 60, 60),
        font=font_smallb,
    )

    bar_y0 = fy + 18
    bar_y1 = fy + 30
    for x in range(PAD, g_w - PAD):
        t = (x - PAD) / max((g_w - 2 * PAD), 1)
        draw.rectangle([x, bar_y0, x + 1, bar_y1], fill=_jet_rgb01(t))
    draw.rectangle([PAD, bar_y0, g_w - PAD, bar_y1], outline=(120, 120, 120), width=1)

    ly = bar_y1 + 7
    sw = 14
    draw.rounded_rectangle([PAD, ly, PAD + sw, ly + sw], radius=3, fill=_jet_rgb01(0.0))
    # Must name the same direction as the red label below, or the two halves of the key
    # contradict each other on a REAL verdict (PLAN E1).
    left_lbl = (
        "BLUE = no positive REAL evidence"
        if str(meta.get("verdict", "")).upper() == "REAL"
        else "BLUE = no positive FAKE evidence"
    )
    draw.text(
        (PAD + sw + 8, ly + sw // 2), left_lbl, fill=(40, 40, 40), font=font_small, anchor="lm"
    )

    # The caption must name the direction that was actually backpropagated (PLAN E1);
    # a fixed "increasing the FAKE score" was wrong on every REAL verdict.
    right_lbl = (
        "RED = regions supporting the REAL verdict"
        if str(meta.get("verdict", "")).upper() == "REAL"
        else "RED = regions increasing the FAKE score"
    )
    r_font = _fit_font(
        draw,
        right_lbl,
        font_small,
        g_w - 2 * PAD - sw - 8 - _text_w(draw, left_lbl, font=_load_font(12)),
    )
    rw = _text_w(draw, right_lbl, font=r_font)
    red_sw_x = g_w - PAD - sw
    draw.rounded_rectangle([red_sw_x, ly, red_sw_x + sw, ly + sw], radius=3, fill=_jet_rgb01(1.0))
    draw.text(
        (red_sw_x - 8 - rw, ly + sw // 2), right_lbl, fill=(40, 40, 40), font=r_font, anchor="rm"
    )

    return np.asarray(img)


def _normalise_grid(cam, raw_frames, device):
    """Resize + JET-colourise each CAM tile and tile frames into a grid."""
    # extract_faces() and the .pt cache hold BGR tensors (OpenCV order); flip to
    # RGB for display so faces aren't rendered blue. Model inputs are unaffected.
    raw = raw_frames.detach().cpu().float()[:, [2, 1, 0]]
    cam_tiles = []
    for i in range(raw.shape[0]):
        heat = cam[i, 0].cpu().numpy()  # (7,7)
        heat = cv2.resize(heat, (224, 224), interpolation=cv2.INTER_CUBIC)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        overlay = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)  # BGR
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        frame = (raw[i].permute(1, 2, 0).numpy() * 255).clip(0, 255)
        # Opacity follows heat (0.15 → 0.6) so cold regions don't tint the face blue.
        alpha = (0.15 + 0.45 * heat)[..., None]
        blended = ((1.0 - alpha) * frame + alpha * overlay).clip(0, 255).astype(np.uint8)
        cam_tiles.append(blended)

    cols = 5
    rows = int(np.ceil(len(cam_tiles) / cols))
    tile_h, tile_w = 224, 224
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for i, tile in enumerate(cam_tiles):
        r, c = divmod(i, cols)
        canvas[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
    return canvas


def _save_grid(canvas, path):
    ok, enc = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode Grad-CAM PNG")
    with open(path, "wb") as f:
        f.write(enc.tobytes())
