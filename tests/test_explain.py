# ================================================================
# tests/test_explain.py
# The AI explanation layer must never claim a facial region it can't
# ground against real landmarks, and the text it writes must trace back
# to the facts it was given (verdict, confidence, gate weights, region).
# ================================================================

import pytest
import torch

from src import explain
from src.explain import analyze_regions, explain_verdict


def _cam_hot_at(row, col, grid=7, n=1):
    cam = torch.zeros(n, 1, grid, grid)
    cam[:, 0, row, col] = 1.0
    return cam


def _frame_meta(landmarks, crop_box=(0, 0, 224, 224)):
    return {"crop_box": list(crop_box), "landmarks": landmarks}


# A face roughly centred in a 224x224 crop: eyes near the top, mouth near
# the bottom, nose in between — the geometry a real SCRFD detection gives.
_CENTRED_LANDMARKS = [
    [80, 70],  # left eye
    [144, 70],  # right eye
    [112, 110],  # nose
    [90, 150],  # left mouth corner
    [134, 150],  # right mouth corner
]


def test_no_landmarks_means_no_region_claim():
    cam = _cam_hot_at(3, 3)
    assert analyze_regions(cam, [{}]) == {"available": False}


def test_misaligned_frame_count_refuses_to_guess():
    cam = _cam_hot_at(3, 3, n=2)
    assert analyze_regions(cam, [_frame_meta(_CENTRED_LANDMARKS)]) == {"available": False}


def test_hot_cell_near_the_eyes_is_labelled_eyes():
    # Eyes sit at row ~70/224*7 ≈ 2.2 → cell row 2.
    cam = _cam_hot_at(2, 2)
    result = analyze_regions(cam, [_frame_meta(_CENTRED_LANDMARKS)])
    assert result["available"] is True
    assert result["dominant"] == "eyes"
    assert result["consistency"] == 1.0


def test_hot_cell_near_the_mouth_is_labelled_mouth():
    # Left mouth corner (90, 150) maps to cell (row 4, col 2) exactly.
    cam = _cam_hot_at(4, 2)
    result = analyze_regions(cam, [_frame_meta(_CENTRED_LANDMARKS)])
    assert result["dominant"] == "mouth"


def test_hot_cell_above_the_eyes_is_forehead_not_a_guessed_feature():
    cam = _cam_hot_at(0, 3)
    result = analyze_regions(cam, [_frame_meta(_CENTRED_LANDMARKS)])
    assert result["dominant"] == "forehead"


def test_explanation_names_the_verdict_and_confidence():
    text = explain_verdict(
        {
            "verdict": "FAKE",
            "confidence": 0.93,
            "has_gate": False,
            "region": {"available": False},
        }
    )
    assert "FAKE" in text
    assert "93%" in text


def test_explanation_cites_the_grounded_region_when_available():
    text = explain_verdict(
        {
            "verdict": "FAKE",
            "confidence": 0.8,
            "has_gate": False,
            "region": {
                "available": True,
                "dominant": "mouth",
                "per_frame": ["mouth"] * 7 + ["nose"] * 3,
                "consistency": 0.7,
            },
        }
    )
    assert "mouth" in text
    assert "7 of the 10 frames" in text


def test_a_real_verdict_never_gets_the_fake_region_warning():
    """Naming a region as 'a common face-swap seam' and then saying it didn't
    matter — in the same sentence — reads as the video being suspicious even
    though the verdict is REAL. A non-technical reader stops at the scary half."""
    text = explain_verdict(
        {
            "verdict": "REAL",
            "confidence": 0.94,
            "has_gate": True,
            "spatial_weight": 0.65,
            "freq_weight": 0.35,
            "region": {
                "available": True,
                "dominant": "nose",
                "per_frame": ["nose"] * 5 + ["mouth"] * 5,
                "consistency": 0.5,
            },
        }
    )
    assert "seam" not in text.lower()
    assert "swapped" not in text.lower()
    assert "found nothing unusual" in text
    # No ML jargon for a non-technical reader.
    for term in ("heatmap", "activation", "fusion gate", "spatial", "frequency"):
        assert term not in text.lower()


def test_explanation_does_not_mention_frequency_without_a_gate():
    text = explain_verdict(
        {
            "verdict": "REAL",
            "confidence": 0.7,
            "has_gate": False,
            "region": {"available": False},
        }
    )
    assert "frequency" not in text.lower()


def test_explanation_mentions_the_gate_split_when_present():
    text = explain_verdict(
        {
            "verdict": "FAKE",
            "confidence": 0.85,
            "has_gate": True,
            "spatial_weight": 0.7,
            "freq_weight": 0.3,
            "region": {"available": False},
        }
    )
    assert "70%" in text and "30%" in text


def test_unknown_provider_fails_loudly_instead_of_silently_falling_back():
    import os

    os.environ["DD_EXPLAIN_PROVIDER"] = "some-other-ai"
    try:
        with pytest.raises(RuntimeError):
            explain_verdict({"verdict": "FAKE", "confidence": 0.9, "region": {"available": False}})
    finally:
        del os.environ["DD_EXPLAIN_PROVIDER"]


def test_claude_provider_calls_the_custom_agent(monkeypatch):
    monkeypatch.setenv("DD_EXPLAIN_PROVIDER", "claude")
    called_with = {}

    def fake_agent(facts):
        called_with.update(facts)
        return "live claude text"

    monkeypatch.setattr(explain, "Custom_AI_Explainer_Agent", fake_agent)
    facts = {"verdict": "FAKE", "confidence": 0.9, "region": {"available": False}}
    assert explain_verdict(facts) == "live claude text"
    assert called_with == facts


def test_claude_provider_falls_back_to_local_on_failure(monkeypatch):
    monkeypatch.setenv("DD_EXPLAIN_PROVIDER", "claude")

    def broken_agent(facts):
        raise RuntimeError("no ANTHROPIC_API_KEY")

    monkeypatch.setattr(explain, "Custom_AI_Explainer_Agent", broken_agent)
    facts = {"verdict": "FAKE", "confidence": 0.9, "region": {"available": False}}
    text = explain_verdict(facts)
    assert "FAKE" in text
    assert "90%" in text


def test_custom_ai_explainer_agent_sends_facts_as_the_user_message(monkeypatch):
    """No real network call: a stub anthropic client stands in for the SDK, and
    the test only checks that Custom_AI_Explainer_Agent wires facts -> request
    -> text the way the Messages API expects."""
    import sys
    import types

    sent = {}

    class FakeBlock:
        type = "text"
        text = "stub explanation"

    class FakeMessages:
        def create(self, **kwargs):
            sent.update(kwargs)
            return types.SimpleNamespace(content=[FakeBlock()])

    class FakeAnthropic:
        def __init__(self):
            self.messages = FakeMessages()

    fake_module = types.SimpleNamespace(Anthropic=FakeAnthropic)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    facts = {"verdict": "REAL", "confidence": 0.8, "region": {"available": False}}
    result = explain.Custom_AI_Explainer_Agent(facts)

    assert result == "stub explanation"
    assert sent["system"] == explain._CLAUDE_SYSTEM_PROMPT
    assert sent["messages"] == [
        {
            "role": "user",
            "content": '{"verdict": "REAL", "confidence": 0.8, "region": {"available": false}}',
        }
    ]
