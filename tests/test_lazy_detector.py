"""Importing src/ must not load InsightFace (FIX_PLAN P3.5, audit-fyp2 L1).

Each check runs in a fresh subprocess: once any test in this session has touched
insightface, it is in sys.modules for good, so an in-process assertion would
pass or fail depending on test order.
"""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code):
    return subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=180
    )


def _assert_clean(module):
    """Importing `module` must leave insightface unloaded."""
    out = _run(
        "import sys\n"
        f"import {module}\n"
        "print('LOADED' if 'insightface' in sys.modules else 'CLEAN')\n"
    )
    assert out.returncode == 0, f"import {module} failed:\n{out.stderr[-2000:]}"
    assert out.stdout.strip().endswith("CLEAN"), (
        f"importing {module} pulled in insightface:\n{out.stdout}"
    )


@pytest.mark.parametrize(
    "module",
    [
        "src.detector",  # the module that owns the detector
        "src.preprocess",  # its only consumer
        "src.train",  # training never calls the detector
        "src.dataset",  # reads the cache, not videos
        "src.model",
    ],
)
def test_import_does_not_load_insightface(module):
    _assert_clean(module)


def test_face_app_attribute_still_resolves_lazily():
    """`from src.detector import face_app` kept working, but the attribute is
    only built when something touches it."""
    out = _run(
        "import sys, src.detector as d\n"
        "assert 'insightface' not in sys.modules, 'import alone loaded it'\n"
        "assert hasattr(d, 'get_face_app')\n"
        "print('OK')\n"
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "OK" in out.stdout


def test_unknown_attribute_still_raises():
    out = _run(
        "import src.detector as d\n"
        "try:\n"
        "    d.not_a_real_attribute\n"
        "except AttributeError:\n"
        "    print('OK')\n"
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "OK" in out.stdout
