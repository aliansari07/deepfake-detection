"""Bootstrap intervals and paired comparison (PLAN D2, audit-fyp2.md 14.4).

These tests exist because every claim the thesis makes about one model beating another will be
produced by src/metrics.py. The properties that matter are: the interval brackets the point
estimate, it is deterministic, the paired comparison really is paired, and degenerate inputs raise
instead of silently returning a number.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.metrics import (
    CI,
    accuracy_ci,
    auc_ci,
    balanced_accuracy_ci,
    bootstrap_metric,
    compare,
    summarise,
)

RESAMPLES = 200  # keeps the suite fast; the properties do not depend on the count


def _scores(n=400, sep=1.0, seed=0):
    """A 1:5 real/fake split with `sep` controlling how separable the two classes are."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 5 / 6).astype(int)
    probs = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + sep * (y * 2 - 1))))
    return y, probs


def test_interval_brackets_the_point_estimate():
    y, p = _scores()
    ci = auc_ci(y, p, resamples=RESAMPLES)
    assert ci.lo <= ci.point <= ci.hi
    assert 0.0 <= ci.lo and ci.hi <= 1.0
    assert ci.point == pytest.approx(roc_auc_score(y, p))


def test_same_seed_reproduces_the_interval_exactly():
    y, p = _scores()
    a = auc_ci(y, p, resamples=RESAMPLES, seed=7)
    b = auc_ci(y, p, resamples=RESAMPLES, seed=7)
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_a_different_seed_moves_the_interval():
    y, p = _scores()
    a = auc_ci(y, p, resamples=RESAMPLES, seed=1)
    b = auc_ci(y, p, resamples=RESAMPLES, seed=2)
    assert (a.lo, a.hi) != (b.lo, b.hi)
    assert a.point == b.point  # the point estimate never depends on the seed


def test_more_data_narrows_the_interval():
    """Subsample the SAME data rather than drawing a fresh small set: interval width is only
    comparable at a comparable AUC, because the [0, 1] bound squeezes it near the ceiling."""
    y, p = _scores(n=2000, seed=3)
    wide = auc_ci(y[:200], p[:200], resamples=RESAMPLES)
    narrow = auc_ci(y, p, resamples=RESAMPLES)
    assert abs(narrow.point - wide.point) < 0.1, "subsample drifted too far to compare widths"
    assert (narrow.hi - narrow.lo) < (wide.hi - wide.lo)


def test_one_class_raises_rather_than_returning_a_number():
    with pytest.raises(ValueError, match="one class"):
        auc_ci(np.ones(50, dtype=int), np.linspace(0, 1, 50), resamples=RESAMPLES)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        bootstrap_metric(
            np.array([0, 1, 0]), np.array([0.1, 0.9]), roc_auc_score, resamples=RESAMPLES
        )


def test_empty_input_raises():
    with pytest.raises(ValueError):
        auc_ci(np.array([]), np.array([]), resamples=RESAMPLES)


def test_identical_models_give_a_difference_interval_containing_zero():
    y, p = _scores()
    c = compare(y, p, p, resamples=RESAMPLES)
    assert c.diff == pytest.approx(0.0)
    assert c.lo <= 0.0 <= c.hi
    assert not c.paired_excludes_zero


def test_a_clearly_better_model_is_detected():
    rng = np.random.default_rng(11)
    y = (rng.random(600) < 5 / 6).astype(int)
    good = np.where(y == 1, rng.uniform(0.6, 1.0, 600), rng.uniform(0.0, 0.4, 600))
    bad = rng.uniform(0, 1, 600)  # pure noise
    c = compare(y, good, bad, name_a="good", name_b="noise", resamples=RESAMPLES)
    assert c.diff > 0.3
    assert c.paired_excludes_zero
    assert c.p_better > 0.99


def test_the_comparison_is_paired_not_two_independent_bootstraps():
    """Two models that differ by a tiny constant offset are ranked identically on every video, so
    the AUC difference is exactly zero on every resample. Only a paired bootstrap can see that:
    independent resampling would produce a wide spread of differences."""
    y, p = _scores(n=600)
    c = compare(y, p, np.clip(p - 1e-9, 0, 1), resamples=RESAMPLES)
    assert c.hi - c.lo < 1e-6, "difference interval should collapse for a rank-identical model"


def test_compare_rejects_differently_shaped_inputs():
    y, p = _scores(n=100)
    with pytest.raises(ValueError, match="same samples"):
        compare(y, p, p[:50], resamples=RESAMPLES)


def test_both_claim_rules_are_reported_and_can_disagree():
    """The paired rule is strictly more powerful, so it may fire when the marginal rule does not.
    The reverse should not happen. This is the substance of decision D6."""
    y, p = _scores(n=800, sep=1.0, seed=5)
    better = np.clip(p + 0.05 * (y * 2 - 1), 0, 1)  # a small, consistent improvement
    c = compare(y, better, p, resamples=500)
    assert isinstance(c.paired_excludes_zero, bool)
    assert isinstance(c.marginal_disjoint, bool)
    if c.marginal_disjoint:
        assert c.paired_excludes_zero, "marginal fired but paired did not - impossible ordering"
    assert "paired rule" in c.verdict() and "marginal rule" in c.verdict()


def test_overlap_helper():
    assert CI(0.9, 0.85, 0.95, 10).overlaps(CI(0.92, 0.90, 0.99, 10))
    assert not CI(0.9, 0.85, 0.89, 10).overlaps(CI(0.95, 0.91, 0.99, 10))


def test_summarise_has_every_headline_metric_with_its_baseline():
    y, p = _scores()
    s = summarise(y, p, threshold=0.5, resamples=RESAMPLES)
    for key in ("auc", "accuracy", "balanced_accuracy", "f1"):
        assert {"point", "ci_low", "ci_high"} <= set(s[key])
        assert s[key]["ci_low"] <= s[key]["point"] <= s[key]["ci_high"]
    # F5-a: the majority baseline must travel with the numbers, or accuracy reads as better
    # than it is on a 1:5 split.
    assert 0.5 < s["majority_class_accuracy"] < 1.0
    assert s["n"] == s["n_real"] + s["n_fake"]


def test_balanced_accuracy_respects_the_threshold():
    y, p = _scores()
    a = balanced_accuracy_ci(y, p, threshold=0.5, resamples=RESAMPLES)
    b = balanced_accuracy_ci(y, p, threshold=0.9, resamples=RESAMPLES)
    assert a.point != b.point


def test_json_round_trip():
    import json

    y, p = _scores()
    json.dumps(summarise(y, p, resamples=RESAMPLES))
    json.dumps(compare(y, p, p, resamples=RESAMPLES).as_dict())


def test_accuracy_ci_is_a_probability():
    y, p = _scores()
    ci = accuracy_ci(y, p, resamples=RESAMPLES)
    assert 0.0 <= ci.lo <= ci.hi <= 1.0
