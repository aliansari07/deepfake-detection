"""Promote a checkpoint to a stable, named model (PLAN task F3).

Every training run writes `checkpoints/<run>.best.pt` and also overwrites
`checkpoints/best_model.pt`.
That mirror is whatever ran last, which makes it useless as a record: the website reads a filename,
the thesis quotes a number, and nothing ties either to the weights that produced them.

Promotion copies a checkpoint to `models/<alias>.pt` and appends a row to `results/models.csv` with
its sha256, architecture, training split and git commit. `models/demo.pt` is the alias the website
serves **by default** (`DD_CHECKPOINT` overrides it), so switching the demo model is one command
and leaves a record of when it changed and why.

Two commands move together, and skipping the second is how the site ends up describing one model
while serving another: promote the weights, then run `tools/make_site_stats.py` to regenerate
`website/lib/modelStats.json` — which reads the last `demo` row above and is the only source of
the architecture and metrics the front end renders.

The CSV is append-only. A re-promotion under the same alias adds a row rather than editing one, so
"what was the demo serving on the day of the viva" stays answerable.

Usage:
    python tools/promote.py checkpoints/A2-20.best.pt --as demo --note "best cross-dataset AUC"
    python tools/promote.py checkpoints/M1-20.best.pt --as m1
    python tools/promote.py --list
"""

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.test_access import sha256_file  # noqa: E402

MODELS_DIR = "models"
CSV_PATH = os.path.join("results", "models.csv")
FIELDS = [
    "promoted_utc",
    "alias",
    "path",
    "source_checkpoint",
    "sha256",
    "epoch",
    "arch",
    "split_file",
    "split_sha256",
    "git_commit",
    "threshold",
    "val_auc",
    "note",
]


def read_rows():
    """Read ``results/models.csv`` as a list of dicts (empty if absent)."""
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def describe(ckpt_path):
    """Everything worth recording about a checkpoint, read from the checkpoint itself."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch") or {}
    split = ck.get("split") or {}
    metrics = ck.get("metrics") or {}
    return {
        "epoch": ck.get("epoch", ""),
        "arch": json.dumps(arch, sort_keys=True) if arch else "(legacy, no arch block)",
        "split_file": os.path.basename(str(split.get("file", ""))) or "(unrecorded)",
        "split_sha256": str(split.get("sha256", ""))[:16],
        "git_commit": str(ck.get("git_commit", ""))[:12],
        "threshold": ck.get("threshold", ""),
        "val_auc": round(metrics["auc"], 4) if isinstance(metrics.get("auc"), float) else "",
    }


def promote(ckpt_path, alias, note=""):
    """Copy a checkpoint to ``models/<alias>.pt`` and record it in ``models.csv``."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH) or ".", exist_ok=True)

    dest = os.path.join(MODELS_DIR, f"{alias}.pt")
    info = describe(ckpt_path)
    sha = sha256_file(ckpt_path)

    previous = [r for r in read_rows() if r["alias"] == alias]
    if previous and previous[-1]["sha256"] == sha:
        print(f"  {alias} already points at these exact weights ({sha[:16]}); nothing to do.")
        return dest

    shutil.copy2(ckpt_path, dest)
    copied = sha256_file(dest)
    if copied != sha:
        os.remove(dest)
        raise RuntimeError("the copy does not match the source; refusing to record it")

    row = {
        "promoted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "alias": alias,
        "path": dest.replace(os.sep, "/"),
        "source_checkpoint": ckpt_path.replace(os.sep, "/"),
        "sha256": sha,
        "note": note,
        **info,
    }

    is_new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)

    if previous:
        print(
            f"  {alias}: replaced {previous[-1]['sha256'][:16]} "
            f"(promoted {previous[-1]['promoted_utc']})"
        )
    print(f"  {alias} -> {dest}")
    print(
        f"    sha256 {sha[:16]}  epoch {info['epoch']}  val AUC {info['val_auc']}  "
        f"threshold {info['threshold']}"
    )
    return dest


def main():
    """Promote a checkpoint, or list the promotions made so far."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint", nargs="?", help="checkpoint to promote")
    ap.add_argument(
        "--as",
        dest="alias",
        default=None,
        help="alias under models/. 'demo' is the one the website serves",
    )
    ap.add_argument("--note", default="", help="why this model was promoted")
    ap.add_argument("--list", action="store_true", help="show the promotion history")
    args = ap.parse_args()

    if args.list or not args.checkpoint:
        rows = read_rows()
        if not rows:
            print(f"No promotions recorded in {CSV_PATH}.")
            return 0
        print(f"{'alias':<10}{'promoted (UTC)':<22}{'sha256':<18}{'val AUC':>9}  note")
        for r in rows:
            print(
                f"{r['alias']:<10}{r['promoted_utc']:<22}{r['sha256'][:16]:<18}"
                f"{r['val_auc']:>9}  {r['note']}"
            )
        current = {}
        for r in rows:
            current[r["alias"]] = r
        print("\ncurrently:")
        for alias, r in current.items():
            print(f"  {alias:<10} {r['source_checkpoint']}")
        return 0

    if not args.alias:
        print("ERROR: --as <alias> is required (use 'demo' for the model the website serves)")
        return 1

    promote(args.checkpoint, args.alias, args.note)
    print(f"\n  recorded in {CSV_PATH}")
    if args.alias == "demo":
        print(
            "  The website serves models/demo.pt by default — no DD_CHECKPOINT needed."
            "\n  Refresh the numbers it advertises with:\n"
            "      python tools/make_site_stats.py"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
