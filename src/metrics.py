"""Bootstrap confidence intervals and paired model comparison (PLAN task D2).

Nothing in this project could previously defend a claim that one model beats another. Every
interval quoted in `audit-fyp2.md` sections 14 and 15 came from a throwaway script that cannot be
regenerated. This module is where those numbers are produced from now on.

Two kinds of interval, and the difference matters:

  marginal   each model's own metric resampled independently. Answers "how precisely do we know
             this model's AUC?"
  paired     both models scored on the SAME resample of the SAME videos, then subtracted. Answers
             "is model A better than model B?"

They disagree, and the disagreement is not a bug. All our models are scored on the same 840
validation videos, so their errors are strongly correlated: a resample containing a hard video
hurts both models together. The paired bootstrap cancels that shared variance; comparing marginal
intervals does not, which makes the overlap test badly over-conservative. On the M1-vs-A2
comparison the marginal intervals overlap while the paired interval excludes zero
(`audit-fyp2.md` section 14.4, decision D6).

Because that decision is open, `compare()` reports BOTH and states which rule each one passes.
It does not choose for you.

Usage:

    from src.metrics import auc_ci, compare

    ci = auc_ci(y_true, probs)                      # CI(point=0.9354, lo=.., hi=..)
    print(ci.format())                               # '0.9354 [0.9145, 0.9546]'

    cmp = compare(y_true, probs_a, probs_b, name_a="A2-20", name_b="M1-20")
    print(cmp.verdict())
"""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

#: Defaults. 2000 resamples is enough for a stable 95% interval on n~840 and costs under a second.
DEFAULT_RESAMPLES = 2000
DEFAULT_ALPHA = 0.05
#: Fixed so every run of every script reproduces the same interval from the same inputs.
DEFAULT_SEED = 42


class CI:
    """A point estimate with a bootstrap percentile interval."""

    __slots__ = ("point", "lo", "hi", "resamples", "skipped", "metric")

    def __init__(self, point, lo, hi, resamples, skipped=0, metric="metric"):
        self.point, self.lo, self.hi = point, lo, hi
        self.resamples, self.skipped, self.metric = resamples, skipped, metric

    def format(self, places=4):
        """Format as ``point [low, high]`` with ``places`` decimals."""
        return f"{self.point:.{places}f} [{self.lo:.{places}f}, {self.hi:.{places}f}]"

    def overlaps(self, other):
        """True when the two intervals share any value — the stricter claim rule."""
        return self.lo <= other.hi and other.lo <= self.hi

    def as_dict(self):
        """Return the interval as a JSON-serialisable dict."""
        return {
            "point": float(self.point),
            "ci_low": float(self.lo),
            "ci_high": float(self.hi),
            "resamples": int(self.resamples),
            "skipped_resamples": int(self.skipped),
            "metric": self.metric,
        }

    def __repr__(self):
        return f"CI({self.metric}={self.format()})"


class Comparison:
    """A paired difference between two models, plus both marginal intervals."""

    __slots__ = (
        "name_a",
        "name_b",
        "ci_a",
        "ci_b",
        "diff",
        "lo",
        "hi",
        "p_better",
        "resamples",
        "metric",
    )

    def __init__(self, name_a, name_b, ci_a, ci_b, diff, lo, hi, p_better, resamples, metric):
        self.name_a, self.name_b = name_a, name_b
        self.ci_a, self.ci_b = ci_a, ci_b
        self.diff, self.lo, self.hi = diff, lo, hi
        self.p_better, self.resamples, self.metric = p_better, resamples, metric

    @property
    def paired_excludes_zero(self):
        """The paired rule: the difference interval does not contain zero."""
        return self.lo > 0.0 or self.hi < 0.0

    @property
    def marginal_disjoint(self):
        """The stricter rule as PLAN.md currently words it: the intervals do not overlap."""
        return not self.ci_a.overlaps(self.ci_b)

    def verdict(self):
        """One-line summary of the difference and whether both tests agree."""
        paired = "excludes zero" if self.paired_excludes_zero else "SPANS ZERO"
        marginal = "disjoint" if self.marginal_disjoint else "OVERLAP"
        agree = "agree" if self.paired_excludes_zero == self.marginal_disjoint else "DISAGREE"
        return (
            f"{self.name_a} - {self.name_b} ({self.metric}): {self.diff:+.4f} "
            f"95% CI [{self.lo:+.4f}, {self.hi:+.4f}], P(better) {self.p_better * 100:.1f}%\n"
            f"  paired rule   : {paired}\n"
            f"  marginal rule : {marginal}  ({self.ci_a.format()} vs {self.ci_b.format()})\n"
            f"  the two rules {agree} (see audit-fyp2.md 14.4, decision D6)"
        )

    def as_dict(self):
        """Return the comparison as a JSON-serialisable dict."""
        return {
            "name_a": self.name_a,
            "name_b": self.name_b,
            "metric": self.metric,
            "difference": float(self.diff),
            "ci_low": float(self.lo),
            "ci_high": float(self.hi),
            "p_a_better": float(self.p_better),
            "resamples": int(self.resamples),
            "paired_excludes_zero": bool(self.paired_excludes_zero),
            "marginal_disjoint": bool(self.marginal_disjoint),
            "rules_agree": bool(self.paired_excludes_zero == self.marginal_disjoint),
            "marginal_a": self.ci_a.as_dict(),
            "marginal_b": self.ci_b.as_dict(),
        }


def _resample_indices(n, resamples, seed):
    """Deterministic bootstrap indices, shape (resamples, n).

    Generated once and shared by every model in a comparison — that sharing is exactly what makes
    the comparison paired.
    """
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def _both_classes(labels):
    return labels.min() != labels.max()


def bootstrap_metric(
    y_true,
    y_score,
    metric_fn,
    resamples=DEFAULT_RESAMPLES,
    alpha=DEFAULT_ALPHA,
    seed=DEFAULT_SEED,
    name="metric",
    indices=None,
):
    """Percentile bootstrap interval for any metric of the form fn(y_true, y_score).

    Resamples in which only one class survives are skipped, because AUC and balanced accuracy are
    undefined there. With a 1:5 split that is rare, but it happens, and silently returning nan for
    those would bias the interval.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError(f"y_true {y_true.shape} and y_score {y_score.shape} differ")
    if y_true.size == 0:
        raise ValueError("no samples")
    if not _both_classes(y_true):
        raise ValueError("only one class present; no interval is defined")

    point = float(metric_fn(y_true, y_score))
    if indices is None:
        indices = _resample_indices(y_true.size, resamples, seed)

    values, skipped = [], 0
    for idx in indices:
        yt = y_true[idx]
        if not _both_classes(yt):
            skipped += 1
            continue
        values.append(metric_fn(yt, y_score[idx]))
    if not values:
        raise ValueError("every resample degenerated to one class")

    values = np.asarray(values, dtype=float)
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CI(point, float(lo), float(hi), len(values), skipped, name)


def _thresholded(metric_fn, threshold):
    def inner(yt, score):
        return metric_fn(yt, (score > threshold).astype(int))

    return inner


def auc_ci(y_true, probs, **kw):
    """Bootstrap confidence interval for ROC AUC."""
    return bootstrap_metric(y_true, probs, roc_auc_score, name="auc", **kw)


def balanced_accuracy_ci(y_true, probs, threshold=0.5, **kw):
    """Bootstrap confidence interval for balanced accuracy at ``threshold``."""
    return bootstrap_metric(
        y_true,
        probs,
        _thresholded(balanced_accuracy_score, threshold),
        name="balanced_accuracy",
        **kw,
    )


def accuracy_ci(y_true, probs, threshold=0.5, **kw):
    """Bootstrap confidence interval for accuracy at ``threshold``."""
    return bootstrap_metric(
        y_true, probs, _thresholded(accuracy_score, threshold), name="accuracy", **kw
    )


def f1_ci(y_true, probs, threshold=0.5, **kw):
    """Bootstrap confidence interval for F1 at ``threshold``."""

    def _f1(yt, yp):
        return f1_score(yt, yp, zero_division=0)

    return bootstrap_metric(y_true, probs, _thresholded(_f1, threshold), name="f1", **kw)


def compare(
    y_true,
    probs_a,
    probs_b,
    name_a="A",
    name_b="B",
    metric_fn=roc_auc_score,
    metric="auc",
    resamples=DEFAULT_RESAMPLES,
    alpha=DEFAULT_ALPHA,
    seed=DEFAULT_SEED,
):
    """Paired bootstrap comparison of two models scored on the same samples.

    Both models are evaluated on identical resamples, so the shared per-video difficulty cancels.
    Also returns each model's marginal interval, so the caller can apply either claim rule.
    """
    y_true = np.asarray(y_true).ravel()
    probs_a = np.asarray(probs_a).ravel()
    probs_b = np.asarray(probs_b).ravel()
    if not (y_true.shape == probs_a.shape == probs_b.shape):
        raise ValueError(
            "compare() needs both models scored on the same samples, in the same "
            f"order: got {y_true.shape}, {probs_a.shape}, {probs_b.shape}"
        )

    indices = _resample_indices(y_true.size, resamples, seed)
    ci_a = bootstrap_metric(y_true, probs_a, metric_fn, name=metric, alpha=alpha, indices=indices)
    ci_b = bootstrap_metric(y_true, probs_b, metric_fn, name=metric, alpha=alpha, indices=indices)

    diffs = []
    for idx in indices:
        yt = y_true[idx]
        if not _both_classes(yt):
            continue
        diffs.append(metric_fn(yt, probs_a[idx]) - metric_fn(yt, probs_b[idx]))
    diffs = np.asarray(diffs, dtype=float)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return Comparison(
        name_a,
        name_b,
        ci_a,
        ci_b,
        float(metric_fn(y_true, probs_a) - metric_fn(y_true, probs_b)),
        float(lo),
        float(hi),
        float((diffs > 0).mean()),
        len(diffs),
        metric,
    )


def summarise(y_true, probs, threshold=0.5, resamples=DEFAULT_RESAMPLES, seed=DEFAULT_SEED):
    """Every headline metric with its interval, as a JSON-ready dict.

    Balanced accuracy and the majority-class baseline are always included: with a 1:5 split,
    accuracy alone hides the real-class collapse (audit-fyp2.md 13.2, action F5-a).
    """
    y_true = np.asarray(y_true).ravel()
    indices = _resample_indices(y_true.size, resamples, seed)
    kw = {"indices": indices}
    n_fake = int(y_true.sum())
    n_real = int(y_true.size - n_fake)
    return {
        "n": int(y_true.size),
        "n_real": n_real,
        "n_fake": n_fake,
        "threshold": float(threshold),
        "resamples": int(resamples),
        "bootstrap_seed": int(seed),
        "auc": auc_ci(y_true, probs, **kw).as_dict(),
        "accuracy": accuracy_ci(y_true, probs, threshold, **kw).as_dict(),
        "balanced_accuracy": balanced_accuracy_ci(y_true, probs, threshold, **kw).as_dict(),
        "f1": f1_ci(y_true, probs, threshold, **kw).as_dict(),
        "majority_class_accuracy": max(n_real, n_fake) / max(y_true.size, 1),
    }
