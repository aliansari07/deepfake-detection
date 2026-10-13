"""The test-split access log (PLAN task D1).

The FF++ test split may be read **once per model**. That rule is what makes a test number
trustworthy: a split you can re-read after seeing the result is a second validation set, and any
number taken from it has been selected on. `src/dataset.py` already locks the split behind
`allow_test=True`; this module is the other half — a durable, append-only record of every unlock,
so the thesis can state how many times the test set was touched and the claim can be checked.

The log lives at `results/TEST_ACCESS_LOG.md` and is committed to git deliberately. A log that can
be quietly regenerated protects nobody.

Repeat access is refused by default. It is not forbidden outright, because legitimate reasons exist
(a bug in the evaluation script itself), but it must be deliberate and it is recorded as a repeat
with its reason, so a reader can weigh it.
"""

import hashlib
import os
from datetime import datetime, timezone

LOG_PATH = os.path.join("results", "TEST_ACCESS_LOG.md")

_HEADER = """# Test-split access log

The FF++ **test** split is read once per model. Every unlock is appended here automatically by
`evaluate.py --split test --final`; nothing in this file is written by hand.

Why this exists: a held-out split that gets re-read after each look is not held out. If a row below
says `repeat`, the number that came from it was taken after the model had already been measured on
this split, and it must be reported that way (`audit-fyp2.md`, PLAN task D1).

| # | UTC timestamp | Model | Checkpoint sha256 | Git commit | Split file | Threshold | n | AUC | Balanced acc | Kind | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
"""


def sha256_file(path, chunk=1 << 20):
    """Stream the file so a 190 MB checkpoint does not land in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _rows(log_path):
    """Parse the pipe table back into dicts. Returns [] when the log does not exist yet."""
    if not os.path.exists(log_path):
        return []
    rows = []
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
            # skip the header row and the |---|---| separator
            if len(cells) < 12 or cells[0] in ("#", "") or set(cells[0]) <= {"-", ":"}:
                continue
            rows.append(
                {
                    "index": cells[0],
                    "timestamp": cells[1],
                    "model": cells[2],
                    "sha256": cells[3],
                    "commit": cells[4],
                    "split_file": cells[5],
                    "threshold": cells[6],
                    "n": cells[7],
                    "auc": cells[8],
                    "balanced_acc": cells[9],
                    "kind": cells[10],
                    "reason": cells[11],
                }
            )
    return rows


def previous_access(checkpoint_sha, log_path=LOG_PATH):
    """Every prior unlock for this exact checkpoint, matched on content not filename.

    The full digest is stored and compared, never a prefix: a truncated hash in a verification
    log cannot be checked against the file it claims to describe.

    Matching on sha256 rather than path is deliberate: renaming or copying a checkpoint must not
    buy a fresh look at the test set.
    """
    return [r for r in _rows(log_path) if r["sha256"] == checkpoint_sha]


def check_access(checkpoint_sha, allow_repeat=False, log_path=LOG_PATH):
    """Raise unless this checkpoint may read the test split. Returns "first" or "repeat"."""
    prior = previous_access(checkpoint_sha, log_path)
    if not prior:
        return "first"
    if not allow_repeat:
        first = prior[0]
        raise PermissionError(
            f"This checkpoint has already been evaluated on the test split "
            f"{len(prior)} time(s) — first on {first['timestamp']} as '{first['model']}'.\n"
            f"The test split is read once per model (PLAN D1). Re-reading it turns the held-out "
            f"set into a second validation set and any number taken from it has been selected on.\n"
            f"If you genuinely must (for example the evaluation script itself was wrong), pass "
            f'--allow-repeat --repeat-reason "..." and the row will be recorded as a repeat.'
        )
    return "repeat"


def record_access(
    *,
    model,
    checkpoint_path,
    checkpoint_sha,
    git_commit,
    split_file,
    threshold,
    n,
    auc,
    balanced_acc,
    kind,
    reason="",
    log_path=LOG_PATH,
):
    """Append one row. Creates the log with its header on first use."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    existing = _rows(log_path)
    if not existing and not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(_HEADER)

    row = (
        "| {i} | {ts} | {model} | `{sha}` | `{commit}` | `{split}` | {thr:.4f} | {n} | "
        "{auc:.4f} | {bal:.4f} | {kind} | {reason} |\n"
    ).format(
        i=len(existing) + 1,
        ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        model=model,
        sha=checkpoint_sha,
        commit=(git_commit or "unknown")[:12],
        split=os.path.basename(split_file),
        thr=threshold,
        n=n,
        auc=auc,
        bal=balanced_acc,
        kind=kind,
        reason=(reason or "—").replace("|", "/"),
    )
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(row)
    return row


def resolve_test_threshold(checkpoint, explicit=None):
    """The operating point for a final test evaluation.

    A test number at 0.5 on a 1:5 split is close to meaningless — M1 needs 0.9442 and A2 needs
    0.4642 to be usable, and reporting both at 0.5 would flatter one and bury the other
    (audit-fyp2.md 13.3, action F5-c). So the default here is the threshold the checkpoint
    calibrated on *validation*, never 0.5, and never anything fitted on the test split itself.

    Returns (threshold, source). Raises when the checkpoint predates task N1 and has no calibrated
    threshold to offer, rather than silently falling back to 0.5.
    """
    if explicit is not None:
        return float(explicit), "given on the command line"
    stored = checkpoint.get("threshold")
    if stored is None:
        raise ValueError(
            "This checkpoint stores no calibrated threshold, so it predates PLAN task N1. "
            "Evaluating it on the test split at the 0.5 fallback would misreport it. "
            "Backfill the threshold from a validation pass first, or pass --threshold explicitly."
        )
    if abs(float(stored) - 0.5) < 1e-9:
        raise ValueError(
            "This checkpoint's stored threshold is exactly 0.5, the pre-N1 fallback value rather "
            "than a calibrated one. Backfill it from a validation pass, or pass --threshold "
            "explicitly if 0.5 really is the intended operating point."
        )
    return float(stored), "validation-calibrated, stored in the checkpoint"
