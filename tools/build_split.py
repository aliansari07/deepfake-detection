#!/usr/bin/env python
"""Build the identity-disjoint FF++ split used from FYP-2 onwards.

  python tools/build_split.py                  # → splits/ffpp_official_v1.json
  python tools/build_split.py --check          # verify an existing file only

Source: the official FaceForensics++ splits (splits/ffpp_official/*.json,
360 / 70 / 70 source-target pairs). A pair (a, b) puts original/a, original/b
and, for every method, a_b and b_a into the same split, so no identity or
source clip crosses splits (fixes audit-fyp2.md F1).

DeepFakeDetection is not included (decision D2, option B): its matching
real footage is not in the dataset, so it would add a scene shortcut.

Items must exist in cache_v2/manifest.json:
  n_detected == 0            → excluded from every split (cannot be scored)
  n_detected <  --min-faces  → excluded from train only; kept and flagged
                               in val/test so evaluation is not cherry-picked
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import CONFIG  # noqa: E402

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
SPLITS = ("train", "val", "test")
OFFICIAL_DIR = os.path.join(ROOT, "splits", "ffpp_official")
DEFAULT_OUT = os.path.join(ROOT, "splits", "ffpp_official_v1.json")
EXPECTED_PAIRS = {"train": 360, "val": 70, "test": 70}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pairs():
    """Load the official FF++ train/val/test pair lists and check their sizes."""
    pairs = {}
    for split in SPLITS:
        with open(os.path.join(OFFICIAL_DIR, f"{split}.json"), encoding="utf-8") as f:
            pairs[split] = [tuple(p) for p in json.load(f)]
        if len(pairs[split]) != EXPECTED_PAIRS[split]:
            raise ValueError(
                f"{split}.json has {len(pairs[split])} pairs, expected {EXPECTED_PAIRS[split]}"
            )
    return pairs


def candidates(pairs):
    """Split → ordered list of 'class/id' keys (no extension)."""
    out = {}
    for split, plist in pairs.items():
        ids = sorted({i for p in plist for i in p})
        items = [f"original/{i}" for i in ids]
        for m in METHODS:
            for a, b in sorted(plist):
                items += [f"{m}/{a}_{b}", f"{m}/{b}_{a}"]
        out[split] = items
    return out


def ids_of(key):
    """Identities referenced by a cache key ('original/000' or 'Deepfakes/000_003')."""
    return set(key.split("/", 1)[1].split("_"))


def verify(spec):
    """Raise if any identity or source clip crosses splits."""
    ids = {s: set().union(*[ids_of(k[:-3]) for k in spec[s]]) for s in SPLITS}
    clips = {
        s: {frozenset(ids_of(k[:-3])) for k in spec[s] if not k.startswith("original/")}
        for s in SPLITS
    }
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            if ids[a] & ids[b]:
                raise AssertionError(f"identity overlap {a}/{b}: {sorted(ids[a] & ids[b])[:5]}")
            if clips[a] & clips[b]:
                raise AssertionError(f"source-clip overlap {a}/{b}")
    for s in SPLITS:
        if any(k.split("/")[0] not in ["original"] + METHODS for k in spec[s]):
            raise AssertionError(f"unexpected class in {s}")
        if len(set(spec[s])) != len(spec[s]):
            raise AssertionError(f"duplicate items in {s}")


def build(cache_root, min_faces):
    """Build the identity-disjoint split spec from the cache v2 manifest."""
    manifest_path = os.path.join(cache_root, "manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        files = json.load(f)["files"]

    spec = {
        "name": "ffpp_official_v1",
        "description": "Identity-disjoint FaceForensics++ C23 split (official pairs), "
        "5 methods + originals, face cache v2 (RGB uint8).",
        "source": "https://github.com/ondyari/FaceForensics/tree/master/dataset/splits",
        "created": date.today().isoformat(),
        "tool": "tools/build_split.py",
        "cache_dir": "cache_v2",
        "cache_version": 2,
        "channel_order": "RGB",
        "methods": METHODS,
        "excluded_classes": {
            "DeepFakeDetection": "no matching real footage in the dataset (decision D2, option B)"
        },
        "min_faces_train": min_faces,
        "cache_manifest_sha256": _sha256(manifest_path),
    }
    excluded, flagged, counts = {}, {}, {}
    for split, keys in candidates(load_pairs()).items():
        kept, exc, flag = [], {}, []
        for key in keys:
            e = files.get(key)
            if e is None:
                exc[key] = "not extracted"
            elif "failed" in e:
                exc[key] = f"extraction failed: {e['failed']}"
            elif e["n_detected"] == 0:
                exc[key] = "no face detected"
            elif e["n_detected"] < min_faces and split == "train":
                exc[key] = f"only {e['n_detected']} faces (< {min_faces}), train only"
            else:
                if e["n_detected"] < min_faces:
                    flag.append(key + ".pt")
                kept.append(key + ".pt")
        spec[split] = kept
        excluded[split] = exc
        flagged[split] = flag
        per_class = {}
        for k in kept:
            c = k.split("/")[0]
            per_class[c] = per_class.get(c, 0) + 1
        counts[split] = {
            "total": len(kept),
            "real": per_class.get("original", 0),
            "fake": len(kept) - per_class.get("original", 0),
            "per_class": per_class,
            "excluded": len(exc),
            "flagged_low_faces": len(flag),
        }
    spec["excluded"] = excluded
    spec["flagged_low_faces"] = flagged
    spec["counts"] = counts
    verify(spec)
    return spec


def main():
    """Write the split file, or verify an existing one with ``--check``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--cache-root", default=os.path.join(CONFIG["DATASET_ROOT"], "cache_v2"))
    ap.add_argument("--min-faces", type=int, default=5)
    ap.add_argument("--check", action="store_true", help="only verify --out")
    args = ap.parse_args()

    if args.check:
        with open(args.out, encoding="utf-8") as f:
            spec = json.load(f)
        verify(spec)
        print(f"OK: {args.out} is identity- and clip-disjoint")
    else:
        spec = build(args.cache_root, args.min_faces)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=1)
            f.write("\n")
        print(f"wrote {args.out}")

    for s in SPLITS:
        c = spec["counts"][s]
        print(
            f"  {s:<5} {c['total']:>5} ({c['real']} real / {c['fake']} fake)  "
            f"excluded {c['excluded']}  flagged {c['flagged_low_faces']}"
        )


if __name__ == "__main__":
    main()
