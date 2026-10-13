"""Copy the irreplaceable checkpoints somewhere else, and verify the copies.

Every trained model exists on exactly one machine. `checkpoints/` is gitignored because the weights
are hundreds of megabytes, and `results/models.csv` records their hashes but not the bytes. If this
laptop fails, roughly five GPU-hours of training is gone and every number in `audit-fyp2.md` becomes
unreproducible — the test split has already been read once per model, so they could not simply be
retrained and re-measured on equal terms.

This copies the checkpoints that matter to a destination of your choosing, verifies each copy by
sha256 against the source, and writes a manifest beside them. Re-running it skips files already
present and verified, so it resumes over a slow USB link.

Usage:
    python tools/backup_checkpoints.py --to E:\\fyp-backup
    python tools/backup_checkpoints.py --to D:\\backup --verify-only
    python tools/backup_checkpoints.py --list
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.test_access import sha256_file  # noqa: E402

#: What is worth backing up, and why. Ordered by how painful the loss would be.
WANTED = [
    (
        "checkpoints/A2-20.best.pt",
        "THE DEMO MODEL — served by the website, promoted as models/demo.pt",
    ),
    ("checkpoints/M1-20.best.pt", "the full architecture, the proposal's model"),
    ("checkpoints/A1-20.best.pt", "no-frequency ablation"),
    ("checkpoints/A4-20.best.pt", "no-augmentation control — carries the S3 claim"),
    ("archive/t4-20_legacy/T4-20.best.pt", "FYP-1 canonical model; the legacy reproduction gate"),
    ("checkpoints/R4-20.best.pt", "the failed R4 run, epoch 5 — evidence for audit section 16"),
]


def present():
    """Return the wanted checkpoints that exist locally, with the reason for each."""
    return [(p, why) for p, why in WANTED if os.path.isfile(p)]


def human(n):
    """Format a byte count in megabytes."""
    return f"{n / 1024**2:,.0f} MB"


def main():
    """Copy the irreplaceable checkpoints to the destination and verify hashes."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--to", help="destination directory (an external drive, a network share)")
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="check an existing backup without copying anything",
    )
    ap.add_argument("--list", action="store_true", help="show what would be copied")
    args = ap.parse_args()

    items = present()
    missing = [p for p, _ in WANTED if not os.path.isfile(p)]
    total = sum(os.path.getsize(p) for p, _ in items)

    if args.list or not args.to:
        print(f"{'file':<44}{'size':>10}   why")
        for p, why in items:
            print(f"{p:<44}{human(os.path.getsize(p)):>10}   {why}")
        if missing:
            print("\nnot on disk: " + ", ".join(missing))
        print(f"\ntotal {human(total)}")
        if not args.to:
            print("\nNothing copied. Pass --to <directory> to back these up, for example:")
            print("    python tools/backup_checkpoints.py --to E:\\fyp-backup")
        return 0

    os.makedirs(args.to, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "source": os.path.abspath("."),
        "files": {},
    }

    ok = failed = skipped = 0
    for src, why in items:
        name = os.path.basename(src)
        dst = os.path.join(args.to, name)
        src_sha = sha256_file(src)

        if os.path.isfile(dst):
            if sha256_file(dst) == src_sha:
                print(f"  ok (already there)  {name}")
                manifest["files"][name] = {"sha256": src_sha, "source": src, "why": why}
                skipped += 1
                continue
            if args.verify_only:
                print(f"  MISMATCH            {name}  — the copy differs from the source")
                failed += 1
                continue
            print(f"  re-copying          {name}  (existing copy differs)")

        if args.verify_only:
            print(f"  MISSING             {name}")
            failed += 1
            continue

        print(
            f"  copying             {name}  ({human(os.path.getsize(src))}) ...", end="", flush=True
        )
        shutil.copy2(src, dst)
        if sha256_file(dst) != src_sha:
            print(" FAILED — the copy does not match")
            failed += 1
            continue
        print(" verified")
        manifest["files"][name] = {"sha256": src_sha, "source": src, "why": why}
        ok += 1

    mpath = os.path.join(args.to, "MANIFEST.json")
    if not args.verify_only:
        with open(mpath, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    print(f"\n  copied {ok}, already present {skipped}, failed {failed}")
    if not args.verify_only:
        print(f"  manifest: {mpath}")
        print(
            "  Every copy was re-hashed after writing, so a silent truncation would have been "
            "caught."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
