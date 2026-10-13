"""Generate the website's headline numbers from the evaluation JSON (PLAN task F2b).

`website/app/page.tsx` carried 95.5% accuracy and 0.991 AUC, hard-coded. Those are T4-20's numbers
on the FYP-1 split, which leaked identities, and they were labelled "validation" on a public page.
The site now serves A2, whose honest test numbers are considerably lower, and the old figures would
have been a false claim in front of a viva panel.

This reads `results/models.csv` to learn which checkpoint is promoted as `demo`, then pulls that
model's numbers out of the evaluation JSON and writes `website/lib/modelStats.json`. The page
imports it, so no number on the site is typed by hand and refreshing them is one command.

It also records what the served architecture actually *is*. A2 has neither a frequency branch nor a
Transformer, so the site's "≈60% gate / ≈40% gate" rows describe components the demo does not run.

Usage:
    python tools/make_site_stats.py
    python tools/make_site_stats.py --check      # non-zero exit if stale
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = "results"
OUT = os.path.join("website", "lib", "modelStats.json")


def load(name):
    """Load a JSON file from ``results/``, or return ``None`` if it is missing."""
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def promoted_demo():
    """Which checkpoint the site serves, from the last `demo` row of results/models.csv."""
    path = os.path.join(RESULTS, "models.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["alias"] == "demo"]
    return rows[-1] if rows else None


def find_model(data, source_checkpoint):
    """Match a comparison JSON's model entry to the promoted checkpoint by run name."""
    if not data:
        return None, None
    stem = os.path.basename(source_checkpoint).split(".")[0].lower().replace("-", "")
    for key, entry in data["models"].items():
        flat = key.lower().replace("-", "").replace("_", "").replace(".", "")
        if stem in flat or stem.replace("20", "") in flat:
            return key, entry
    return None, None


def build():
    """Assemble the website stats for the model promoted as ``demo``."""
    demo = promoted_demo()
    if demo is None:
        raise SystemExit(
            "No model promoted as 'demo'. Run: python tools/promote.py <ckpt> --as demo"
        )

    arch = json.loads(demo["arch"]) if demo["arch"].startswith("{") else {}
    test = load("comparison_test.json")
    celeb = load("comparison_celebdf.json")

    _, t = find_model(test, demo["source_checkpoint"])
    _, c = find_model(celeb, demo["source_checkpoint"])

    stats = {
        "_generated_by": "tools/make_site_stats.py — do not edit by hand (PLAN F2b)",
        "model": {
            "alias": demo["alias"],
            "run": os.path.basename(demo["source_checkpoint"]).split(".")[0],
            "sha256": demo["sha256"][:16],
            "promoted_utc": demo["promoted_utc"],
            "epoch": demo["epoch"],
            "threshold": float(demo["threshold"]) if demo["threshold"] else None,
            "split": demo["split_file"],
        },
        "architecture": {
            "frequency_branch": bool(arch.get("freq_enabled", True))
            and not bool(arch.get("baseline_mode", False)),
            "transformer": bool(arch.get("transformer_enabled", True))
            and not bool(arch.get("baseline_mode", False)),
            "spatial_backbone": "EfficientNet-B0",
            # What the frequency branch looks at (R5). The site names the transform in three
            # places, and an R5 checkpoint takes no Fourier transform at all, so this has to be
            # generated rather than written into the copy — that is the same mistake as F2a,
            # where the page described components the served model did not run.
            "frequency_input": str(arch.get("freq_input", "spectrum")),
            "frames_per_clip": arch.get("sample_count", 10),
        },
    }

    if t:
        stats["test"] = {
            "dataset": "FaceForensics++ C23, identity-disjoint test split",
            "n": test["n"],
            "accuracy": t["accuracy"]["point"],
            "balanced_accuracy": t["balanced_accuracy"]["point"],
            "auc": t["auc"]["point"],
            "auc_ci": [t["auc"]["ci_low"], t["auc"]["ci_high"]],
            "threshold": t["threshold"],
        }
    if c:
        stats["cross_dataset"] = {
            "dataset": "Celeb-DF v2, official 518-video protocol",
            "n": celeb["n"],
            "accuracy": c["accuracy"]["point"],
            "auc": c["auc"]["point"],
            "auc_ci": [c["auc"]["ci_low"], c["auc"]["ci_high"]],
        }
    return json.dumps(stats, indent=2) + "\n"


def main():
    """Write ``website/lib/modelStats.json``, or check that it is current."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    text = build()
    if args.check:
        current = open(args.out, encoding="utf-8").read() if os.path.exists(args.out) else ""
        if current != text:
            print(f"{args.out} is out of date — run: python tools/make_site_stats.py")
            return 1
        print(f"{args.out} is up to date")
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"written: {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
