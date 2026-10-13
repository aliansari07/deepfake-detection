"""InsightFace SCRFD face detector, built on first use.

Import: from src.detector import get_face_app
        app = get_face_app()

The detector used to be constructed at import time, which meant importing
ANY module under src/ downloaded and loaded the buffalo_l model pack. That
made training, the tests and CI pay for a detector they never call, and it
is why the suite could not run anywhere without the model files present
(FIX_PLAN P3.5, audit-fyp2.md L1). The v2 extraction path in
src/face_cache.py::get_detector already worked this way; this matches it.
"""

from src.config import CONFIG
from src.logger import log

_FACE_APP = None


def get_face_app():
    """Return the shared FaceAnalysis instance, building it on first call."""
    global _FACE_APP
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis  # heavy: imported on demand

        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=CONFIG["DET_SIZE"])
        log("DETECTOR", "InsightFace SCRFD ready.")
        _FACE_APP = app
    return _FACE_APP


def __getattr__(name):
    """Keep `from src.detector import face_app` working, but lazily.

    PEP 562 module-level __getattr__: the attribute is only resolved when
    something actually touches it, so the import itself stays cheap.
    """
    if name == "face_app":
        return get_face_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
