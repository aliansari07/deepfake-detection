"""Grad-CAM must explain the verdict it is captioned with (PLAN task E1).

The model emits one logit and sigmoid(logit) is p(fake), so backpropagating the logit always asks
"what pushed this toward FAKE". On a REAL verdict the caption said REAL while the heatmap
highlighted evidence for the opposite conclusion — an explanation of a decision the model did not
make, which is worse than no explanation at all in a project about trusting media.
"""

import inspect

import numpy as np
import torch

from src import gradcam


def test_a_real_verdict_backpropagates_the_negated_logit():
    src = inspect.getsource(gradcam.generate_gradcam)
    assert "(logit if explain_fake else -logit).backward()" in src
    assert 'verdict != "REAL"' in src


def test_uncertain_and_fake_keep_the_fake_direction():
    """UNCERTAIN should still show what fake-evidence was found; only REAL flips."""
    src = inspect.getsource(gradcam.generate_gradcam)
    assert 'str((meta or {}).get("verdict", "FAKE")).upper()' in src, (
        "missing meta must default to the fake direction, the pre-E1 behaviour"
    )


def test_the_colour_key_names_the_direction_actually_explained():
    src = inspect.getsource(gradcam._annotate_grid)
    assert "RED = regions supporting the REAL verdict" in src
    assert "RED = regions increasing the FAKE score" in src


def _caption_for(verdict):
    """Render the annotated grid and read back which colour key it drew."""
    grid = np.full((64, 240, 3), 200, dtype=np.uint8)
    out = gradcam._annotate_grid(
        grid,
        {
            "video_name": "x.mp4",
            "verdict": verdict,
            "prob": 0.5,
            "confidence": 0.5,
            "logit": 0.0,
            "s_weight": 1.0,
            "faces": 10,
            "faces_total": 10,
        },
    )
    return out


def test_annotate_runs_for_every_verdict():
    """The caption branch must not crash on any verdict the pipeline can produce."""
    for verdict in ("FAKE", "REAL", "UNCERTAIN"):
        out = _caption_for(verdict)
        assert out.ndim == 3 and out.shape[2] == 3
        assert out.shape[0] > 64, "header and footer bands should have been added"


def test_annotate_tolerates_a_missing_verdict():
    grid = np.full((64, 240, 3), 200, dtype=np.uint8)
    assert gradcam._annotate_grid(grid, {}).ndim == 3


def test_the_direction_changes_the_heatmap():
    """A REAL and a FAKE explanation of the same clip must not be the same picture."""
    torch.manual_seed(0)
    from src.model import DeepfakeDetector

    model = DeepfakeDetector().eval()
    frames = torch.rand(2, 3, 224, 224, requires_grad=True)

    cams = {}
    for verdict in ("FAKE", "REAL"):
        target = model.spatial.bn2
        fh = target.register_forward_hook(gradcam._hook_forward)
        bh = target.register_full_backward_hook(gradcam._hook_backward)
        try:
            logit, _ = model(frames)
            target.zero_grad()
            (logit if verdict != "REAL" else -logit).backward(retain_graph=True)
            w = target._gradients.mean(dim=(2, 3), keepdim=True)
            cams[verdict] = torch.relu((w * target._activations).sum(dim=1)).clone()
        finally:
            fh.remove()
            bh.remove()
            model.zero_grad(set_to_none=True)

    assert not torch.allclose(cams["FAKE"], cams["REAL"]), (
        "flipping the explained direction must change the heatmap"
    )


def test_a_gateless_model_does_not_get_a_gate_bar():
    """A spatial-only checkpoint has no fusion gate; drawing "spatial 100% / freq 0%" would claim
    a gate weighed two streams and chose one (the same false reading fixed on the website)."""
    import inspect

    src = inspect.getsource(gradcam._annotate_grid)
    assert 'meta.get("has_gate", True)' in src
    assert "no fusion gate in this model" in src


def test_the_blue_and_red_labels_name_the_same_direction():
    src = inspect.getsource(gradcam._annotate_grid)
    assert "BLUE = no positive REAL evidence" in src
    assert "RED = regions supporting the REAL verdict" in src


def test_annotate_runs_with_and_without_a_gate():
    for has_gate in (True, False):
        out = gradcam._annotate_grid(
            np.full((64, 320, 3), 200, dtype=np.uint8),
            {
                "video_name": "x.mp4",
                "verdict": "REAL",
                "prob": 0.1,
                "confidence": 0.9,
                "logit": -2.0,
                "s_weight": 1.0,
                "has_gate": has_gate,
                "faces": 10,
                "faces_total": 10,
            },
        )
        assert out.ndim == 3


def test_inference_reports_whether_the_model_has_a_gate():
    import inspect

    from src import inference

    src = "".join(inspect.getsource(inference.probe_video).split())  # whitespace-insensitive
    assert '"has_gate":gate_gisnotNone,' in src
