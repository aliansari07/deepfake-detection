"""The test-split access log and its locks (PLAN task D1).

The whole value of a held-out split is that it is read once. These tests protect the mechanics of
that: the lock keys on checkpoint content rather than filename, a repeat is refused unless asked
for explicitly, and a final evaluation never silently falls back to the 0.5 threshold.
"""

import pytest

from src.dataset import DeepfakeDataset
from src.test_access import (
    check_access,
    previous_access,
    record_access,
    resolve_test_threshold,
    sha256_file,
)


@pytest.fixture
def log(tmp_path):
    return str(tmp_path / "TEST_ACCESS_LOG.md")


def _record(log_path, sha, model="M1-20", kind="first", reason=""):
    return record_access(
        model=model,
        checkpoint_path=f"checkpoints/{model}.best.pt",
        checkpoint_sha=sha,
        git_commit="90aa6cb0844b",
        split_file="splits/ffpp_official_v1.json",
        threshold=0.9442,
        n=840,
        auc=0.91,
        balanced_acc=0.84,
        kind=kind,
        reason=reason,
        log_path=log_path,
    )


# ── the lock ──────────────────────────────────────────────────────


def test_a_fresh_checkpoint_may_read_the_test_split(log):
    assert check_access("a" * 64, log_path=log) == "first"


def test_a_second_read_of_the_same_checkpoint_is_refused(log):
    _record(log, "a" * 64)
    with pytest.raises(PermissionError, match="already been evaluated"):
        check_access("a" * 64, log_path=log)


def test_the_refusal_explains_why_and_how_to_override(log):
    _record(log, "a" * 64)
    with pytest.raises(PermissionError) as e:
        check_access("a" * 64, log_path=log)
    assert "--allow-repeat" in str(e.value)
    assert "once per model" in str(e.value)


def test_a_repeat_is_permitted_when_asked_for_explicitly(log):
    _record(log, "a" * 64)
    assert check_access("a" * 64, allow_repeat=True, log_path=log) == "repeat"


def test_a_different_checkpoint_is_unaffected(log):
    _record(log, "a" * 64)
    assert check_access("b" * 64, log_path=log) == "first"


def test_renaming_a_checkpoint_does_not_buy_a_second_look(log):
    """The lock keys on content, so copying M1-20.best.pt to M1-final.pt changes nothing."""
    _record(log, "a" * 64, model="M1-20")
    with pytest.raises(PermissionError):
        check_access("a" * 64, log_path=log)


# ── the log itself ────────────────────────────────────────────────


def test_the_log_is_created_with_an_explanatory_header(log):
    _record(log, "a" * 64)
    text = open(log, encoding="utf-8").read()
    assert "Test-split access log" in text
    assert "not held out" in text  # the rationale must travel with the data
    assert "| # |" in text


def test_rows_accumulate_and_are_numbered(log):
    _record(log, "a" * 64, model="M1-20")
    _record(log, "b" * 64, model="A2-20")
    _record(log, "c" * 64, model="A1-20")
    rows = previous_access("b" * 64, log_path=log)
    assert len(rows) == 1 and rows[0]["model"] == "A2-20" and rows[0]["index"] == "2"


def test_a_repeat_row_records_its_reason(log):
    _record(log, "a" * 64)
    _record(log, "a" * 64, kind="repeat", reason="evaluation script had a bug")
    rows = previous_access("a" * 64, log_path=log)
    assert len(rows) == 2
    assert rows[1]["kind"] == "repeat"
    assert "bug" in rows[1]["reason"]


def test_a_pipe_in_the_reason_cannot_break_the_table(log):
    _record(log, "a" * 64, kind="repeat", reason="a | b | c")
    rows = previous_access("a" * 64, log_path=log)
    assert len(rows) == 1


def test_missing_log_reads_as_no_prior_access(tmp_path):
    assert previous_access("a" * 64, log_path=str(tmp_path / "nope.md")) == []


def test_sha256_of_a_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert sha256_file(str(p)) == expected


# ── the threshold ─────────────────────────────────────────────────


def test_a_final_evaluation_uses_the_checkpoints_calibrated_threshold():
    thr, source = resolve_test_threshold({"threshold": 0.9442})
    assert thr == pytest.approx(0.9442)
    assert "validation" in source


def test_an_explicit_threshold_wins():
    thr, source = resolve_test_threshold({"threshold": 0.9442}, explicit=0.5)
    assert thr == 0.5 and "command line" in source


def test_a_checkpoint_with_no_threshold_is_refused():
    """Silently evaluating a pre-N1 checkpoint at 0.5 would misreport it (audit 13.3)."""
    with pytest.raises(ValueError, match="predates"):
        resolve_test_threshold({})


def test_the_pre_n1_fallback_value_is_refused():
    with pytest.raises(ValueError, match="fallback"):
        resolve_test_threshold({"threshold": 0.5})


# ── the dataset lock this builds on ───────────────────────────────


def test_the_dataset_still_refuses_the_test_split_by_default():
    from src.config import CONFIG

    with pytest.raises(PermissionError, match="locked"):
        DeepfakeDataset(CONFIG["DATASET_ROOT"], split="test")
