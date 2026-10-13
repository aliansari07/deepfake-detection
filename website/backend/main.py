"""Thin FastAPI wrapper around the existing src/inference.py pipeline.
Exposes POST /analyze — accepts a video upload, runs the model,
and returns the verdict + Grad-CAM PNG (base64).

Run (from the repository root):
  uvicorn website.backend.main:app --port 8000

Env:
  DD_CHECKPOINT     checkpoint name under checkpoints/, a project-relative
                    path, or an absolute path
                    (default models/demo.pt — the promoted model, which is the
                    one website/lib/modelStats.json quotes)
  DD_MAX_UPLOAD_MB  upload size cap in MB (default 200)
  DD_ALLOWED_ORIGINS
                    comma-separated CORS origins allowed to call the API
                    (default http://localhost:3000,http://127.0.0.1:3000)
"""

import base64
import os
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager

# Make the project-root `src/` package importable no matter how uvicorn
# is launched (uvicorn does not reliably add the CWD to sys.path).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402


def _sample_count() -> int:
    try:
        from src.config import CONFIG

        return int(CONFIG["SAMPLE_COUNT"])
    except Exception:
        return 10


# Pinned to a named model, not the best_model.pt mirror that any training run overwrites.
#
# The default is the *promoted* model rather than a hand-written checkpoint name, because the site
# quotes one model's numbers: tools/make_site_stats.py reads results/models.csv and writes
# website/lib/modelStats.json from the last `demo` row. Serving any other weights while displaying
# those numbers would be a false claim, so the served weights and the advertised architecture are
# the same thing by construction.
#
# DD_CHECKPOINT accepts three forms, so the promoted alias from tools/promote.py can be served
# without moving files around (PLAN task F3):
#   "demo.pt"            -> checkpoints/demo.pt      (a bare name, the historical behaviour)
#   "models/demo.pt"     -> <project>/models/demo.pt (a project-relative path)
#   "C:/.../demo.pt"     -> used as given            (absolute)
DEFAULT_CHECKPOINT = "models/demo.pt"
_requested = os.environ.get("DD_CHECKPOINT", DEFAULT_CHECKPOINT)
if os.path.isabs(_requested):
    CHECKPOINT = _requested
elif os.sep in _requested or "/" in _requested:
    CHECKPOINT = os.path.join(PROJECT_ROOT, _requested)
else:
    CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", _requested)

if not os.path.isfile(CHECKPOINT):
    raise RuntimeError(
        f"Checkpoint not found: {CHECKPOINT}. Set DD_CHECKPOINT to a checkpoint, or restore the "
        "promoted model with:\n"
        "    python tools/promote.py <checkpoint.pt> --as demo\n"
        "or download the released checkpoint (see README.md).\n"
        "Refusing to serve weights the site does not describe."
    )

MAX_UPLOAD_MB = int(os.environ.get("DD_MAX_UPLOAD_MB", "200"))

# Browser origins allowed to call the API (the Next.js front end).
DEFAULT_ALLOWED_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("DD_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(",")
    if origin.strip()
]
_CHUNK = 1 << 20

# The model is loaded once and reused; analyses run one at a time on the GPU.
_model = None
_model_lock = threading.Lock()
_gpu_lock = threading.Lock()
_BUSY_TIMEOUT_S = 90

# Hash of the served weights, so /health can prove they are the ones the site advertises
# (website/lib/modelStats.json carries the same prefix under model.sha256).
_checkpoint_sha = None


def _get_sha():
    global _checkpoint_sha
    if _checkpoint_sha is None:
        from src.test_access import sha256_file

        _checkpoint_sha = sha256_file(CHECKPOINT)
    return _checkpoint_sha


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from src.inference import _load_model

            _model = _load_model(CHECKPOINT)
    return _model


@asynccontextmanager
async def lifespan(_app):
    """Hash and load the model at startup so errors surface before the first request."""
    _get_sha()  # hash once at startup, not on every /health poll
    _get_model()  # load at startup so the first upload isn't slow and errors surface early
    yield


app = FastAPI(title="Deepfake Detector API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


def _run_analysis(video_path: str) -> dict:
    """Run the existing pipeline and normalise its output for the API."""
    from src.inference import run_pipeline

    start = time.time()
    result = run_pipeline(video_path=video_path, model=_get_model())

    if not result:  # no faces extracted
        raise HTTPException(
            status_code=422,
            detail="No faces were detected in the video. Supply a clip of a "
            "single person's face and retry.",
        )

    gradcam_png = None
    gradcam_path = result.get("gradcam_path")
    if gradcam_path and os.path.exists(gradcam_path):
        try:
            with open(gradcam_path, "rb") as f:
                gradcam_png = base64.b64encode(f.read()).decode("ascii")
        finally:
            # Never retain imagery of user uploads on disk (SRS NFR-12).
            try:
                os.remove(gradcam_path)
            except OSError:
                pass

    meta_faces = None
    faces_total = _sample_count()
    if isinstance(result, dict):
        meta_faces = result.get("faces_detected")
        faces_total = result.get("faces_total", faces_total)

    return {
        "verdict": result["result"],
        "prob": float(result["prob"]),
        "logit": float(result.get("logit", 0.0)),
        "confidence": float(result.get("confidence", result["prob"])),
        "spatial_weight": float(result.get("s_weight", 0.5)),
        "freq_weight": float(result.get("freq_weight", 1.0 - result.get("s_weight", 0.5))),
        "gradcam_base64": gradcam_png or "",
        "faces_detected": int(meta_faces) if meta_faces else 0,
        "faces_total": int(faces_total) if faces_total else 0,
        "analysis_time": round(time.time() - start, 2),
        "ai_explanation": result.get("ai_explanation", ""),
    }


# Sync endpoint: FastAPI runs it in its threadpool, so the event loop (and
# /health) stays responsive while a video is being analysed.
@app.post("/analyze")
def analyze(video: UploadFile = File(...)):  # noqa: B008 (FastAPI idiom)
    """Analyse an uploaded video and return the verdict and Grad-CAM image."""
    if video.content_type not in ("video/mp4", "video/quicktime", "video/webm"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{video.content_type}'. Accepts MP4, MOV, WEBM.",
        )

    max_bytes = MAX_UPLOAD_MB * _CHUNK
    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            size = 0
            while chunk := video.file.read(_CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413, detail=f"File is larger than {MAX_UPLOAD_MB} MB."
                    )
                tmp.write(chunk)

        if not _gpu_lock.acquire(timeout=_BUSY_TIMEOUT_S):
            raise HTTPException(
                status_code=503,
                detail="The detector is busy with another video. Try again shortly.",
            )
        try:
            return _run_analysis(tmp_path)
        finally:
            _gpu_lock.release()
    except HTTPException:
        raise
    except Exception as exc:  # surface as 500 for the UI
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.get("/health")
async def health():
    """Report service status and which checkpoint is being served."""
    return {
        "status": "ok",
        "checkpoint": os.path.basename(CHECKPOINT),
        "checkpoint_path": CHECKPOINT,
        "sha256": _get_sha()[:16],
        "model_loaded": _model is not None,
    }
