import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def stub_face_detector():
    """src.preprocess imports src.detector, which loads InsightFace at import
    time. Tests that only need src.inference replace it with a stub."""
    if "src.preprocess" not in sys.modules:
        stub = types.ModuleType("src.preprocess")
        stub.extract_faces = lambda *a, **k: (None, {})
        sys.modules["src.preprocess"] = stub
