#!/usr/bin/env python
"""Build per-video .pt cache of InsightFace-aligned face crops.

  python preextract.py              # extract all
  python preextract.py --folder original  # one class
  python preextract.py --dry-run         # count only

  # Face cache v2 (RGB uint8, detection retries, no padding; FIX_PLAN P2.2)
  python preextract.py --cache-version 2 --workers 10 --threads 2
  python preextract.py --cache-version 2 --folder original
  python preextract.py --cache-version 2 --dataset celebdf            # official 518
  python preextract.py --cache-version 2 --dataset celebdf --celebdf-sample

  v2 output: datasets/<dataset>/cache_v2/<class>/<video_id>.pt (dict, see
  src/face_cache.py) + cache_v2/manifest.json (sha256, n_detected, retries).
  DeepFakeDetection is skipped by default for FF++ (decision D2, option B);
  pass --folder DeepFakeDetection to extract it anyway.

Dataset layout expected (see README.md):
  datasets/FaceForensics++_C23/{class}/{video_id}.mp4

Output:
  datasets/FaceForensics++_C23/cache/{class}/{video_id}.pt
  Each .pt: dict(video_id, sample_count, tensor, meta)
    tensor  — (N, 3, 224, 224) uint8 on CPU
    meta    — list[N] of dicts(frames_skipped, total_frames)

Resume-safe: skips videos that already have a valid cache.
Report target: 6,648 / 7,000 successfully extracted.
"""

import argparse
import json
import os
import sys
import time
import traceback

import torch

from src.config import CONFIG
from src.logger import log, section

# v2 FF++ classes: the five FF++ methods with source-pair names + originals.
V2_FFPP_CLASSES = [
    "original",
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
    "FaceShifter",
]


def _csv_dir():
    return os.path.join(CONFIG["DATASET_ROOT"], "csv")


def _cache_root():
    return os.path.join(CONFIG["DATASET_ROOT"], "cache")


def _list_classes(folder=None):
    root = CONFIG["DATASET_ROOT"]
    if folder:
        return [folder] if os.path.isdir(os.path.join(root, folder)) else []
    classes = sorted(
        d
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and d not in ("cache", "cache_v2", "csv", "archive.zip")
    )
    return classes


def _list_videos(cls):
    folder = os.path.join(CONFIG["DATASET_ROOT"], cls)
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.endswith(".mp4"))


def _video_id(fname):
    return os.path.splitext(fname)[0]


def _cache_path(cls, vid):
    return os.path.join(_cache_root(), cls, f"{vid}.pt")


def _valid_cache(path):
    if not os.path.exists(path):
        return False
    try:
        t = torch.load(path, map_location="cpu", weights_only=True)
        return isinstance(t, torch.Tensor) and t.dim() == 4 and t.shape[1:] == (3, 224, 224)
    except Exception:
        return False


def extract_class(cls, dry_run=False):
    """Extract all videos in a class folder. Returns (success, total, skipped)."""
    cache_dir = os.path.join(_cache_root(), cls)
    os.makedirs(cache_dir, exist_ok=True)

    videos = _list_videos(cls)
    success, skipped, total = 0, 0, len(videos)
    t0 = time.time()

    for i, vf in enumerate(videos, 1):
        vid = _video_id(vf)
        cp = _cache_path(cls, vid)

        if _valid_cache(cp):
            skipped += 1
            continue

        if dry_run:
            success += 1
            continue

        try:
            # Imported here so v2 worker processes never load the legacy detector.
            from src.preprocess import extract_faces

            vpath = os.path.join(CONFIG["DATASET_ROOT"], cls, vf)
            tensor, meta = extract_faces(vpath, sample_count=CONFIG["SAMPLE_COUNT"])

            if tensor is None:
                print(f"  [{cls}] skip {vid} — extraction failed")
                continue

            # NOTE: dataset.py torch.load()s the cache and expects a bare
            # (N, 3, 224, 224) tensor — save the tensor directly.
            torch.save(tensor, cp)
            success += 1

            if i % 100 == 0 or i == total:
                elapsed = time.time() - t0
                print(
                    f"  [{cls}] {i}/{total} — {elapsed:.0f}s — {success} saved, {skipped} skipped"
                )

        except Exception as e:
            print(f"  [{cls}] ERROR {vid}: {e}")
            traceback.print_exc()

    elapsed = time.time() - t0
    log("PREEXTRACT", f"[{cls}] {success}/{total} saved, {skipped} skipped — {elapsed:.0f}s")
    return success, total, skipped


# ── Face cache v2 (parallel) ─────────────────────────────────────


def _v2_jobs(args):
    """[(video_path, rel_key)] and the cache_v2 root for the chosen dataset."""
    if args.dataset == "celebdf":
        from src.celebdf import CELEB_ROOT, build_official, build_sample, rel_key

        videos, _ = build_official()
        if args.celebdf_sample:
            sample, _ = build_sample(890, 890, 42)
            videos = sorted(set(videos) | set(sample))
        return [(v, rel_key(v)) for v in videos], os.path.join(CELEB_ROOT, "cache_v2")

    root = CONFIG["DATASET_ROOT"]
    classes = [args.folder] if args.folder else V2_FFPP_CLASSES
    jobs = []
    for cls in classes:
        for vf in _list_videos(cls):
            jobs.append((os.path.join(root, cls, vf), f"{cls}/{_video_id(vf)}"))
    return jobs, os.path.join(root, "cache_v2")


def _write_manifest(path, manifest):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def run_v2(args):
    """Extract face cache v2 files in parallel and write the cache manifest."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from src.face_cache import CACHE_VERSION, extract_to_file

    jobs, cache_root = _v2_jobs(args)
    os.makedirs(cache_root, exist_ok=True)
    manifest_path = os.path.join(cache_root, "manifest.json")
    manifest = {
        "version": CACHE_VERSION,
        "channel_order": "RGB",
        "dtype": "uint8",
        "sample_count": CONFIG["SAMPLE_COUNT"],
        "files": {},
    }
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

    def _done(rel):
        e = manifest["files"].get(rel)
        p = os.path.join(cache_root, *rel.split("/")) + ".pt"
        return (
            e is not None
            and "failed" not in e
            and os.path.exists(p)
            and os.path.getsize(p) == e.get("size")
        )

    pending = [
        (v, rel, os.path.join(cache_root, *rel.split("/")) + ".pt", args.threads)
        for v, rel in jobs
        if not _done(rel)
    ]
    print(f"cache v2 → {cache_root}")
    print(
        f"{len(jobs)} videos, {len(jobs) - len(pending)} already cached, "
        f"{len(pending)} to extract ({args.workers} workers x {args.threads} threads)"
    )
    if args.dry_run or not pending:
        return

    t0, n = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract_to_file, j) for j in pending]
        for fut in as_completed(futures):
            rel, entry = fut.result()
            manifest["files"][rel] = entry
            n += 1
            if "failed" in entry:
                print(f"  FAILED {rel}: {entry['failed']}", flush=True)
            if n % 100 == 0 or n == len(pending):
                el = time.time() - t0
                eta = el / n * (len(pending) - n)
                print(
                    f"  {n}/{len(pending)}  {el / 60:.1f} min  eta {eta / 60:.1f} min", flush=True
                )
                _write_manifest(manifest_path, manifest)
    _write_manifest(manifest_path, manifest)

    files = manifest["files"]
    failed = [r for r, e in files.items() if "failed" in e]
    by_cls = {}
    for rel, e in files.items():
        if "failed" in e:
            continue
        c = by_cls.setdefault(rel.split("/")[0], {"n": 0, "k0": 0, "k_lt10": 0, "retried": 0})
        c["n"] += 1
        c["k0"] += e["n_detected"] == 0
        c["k_lt10"] += e["n_detected"] < CONFIG["SAMPLE_COUNT"]
        c["retried"] += e["retries"] > 0
    print(
        f"\nDONE in {(time.time() - t0) / 60:.1f} min — {len(files)} in manifest, {len(failed)} failed"  # noqa: E501
    )
    print(f"{'class':<20}{'videos':>8}{'no face':>9}{'k<10':>7}{'k<10 %':>8}{'retried':>9}")
    for cls, c in sorted(by_cls.items()):
        print(
            f"{cls:<20}{c['n']:>8}{c['k0']:>9}{c['k_lt10']:>7}"
            f"{100.0 * c['k_lt10'] / max(c['n'], 1):>7.2f}%{c['retried']:>9}"
        )


def main():
    """Parse arguments and run face pre-extraction for the chosen cache version."""
    parser = argparse.ArgumentParser(description="Pre-extract face crops to .pt cache")
    parser.add_argument("--folder", help="Restrict to one class (e.g. original)")
    parser.add_argument("--dry-run", action="store_true", help="Count only")
    parser.add_argument(
        "--cache-version",
        type=int,
        choices=(1, 2),
        default=1,
        help="1 = legacy BGR float cache/ (FYP-1), 2 = RGB uint8 cache_v2/",
    )
    parser.add_argument(
        "--dataset",
        choices=("ffpp", "celebdf"),
        default="ffpp",
        help="v2 only: which dataset to extract",
    )
    parser.add_argument(
        "--celebdf-sample",
        action="store_true",
        help="v2 celebdf: also extract the FYP-1 890/890 supplementary sample",
    )
    parser.add_argument("--workers", type=int, default=10, help="v2: extraction processes")
    parser.add_argument(
        "--threads", type=int, default=2, help="v2: ONNX Runtime threads per process"
    )
    args = parser.parse_args()

    if args.cache_version == 2:
        section("FACE EXTRACTION (cache v2)")
        run_v2(args)
        return

    section("FACE EXTRACTION")
    classes = _list_classes(args.folder)
    if not classes:
        print("No class folders found. Check DATASET_ROOT in src/config.py.")
        sys.exit(1)

    grand_total, grand_success, grand_skipped = 0, 0, 0
    t0 = time.time()

    for cls in classes:
        print(f"\n--- Extracting class: {cls} ---")
        s, t, sk = extract_class(cls, dry_run=args.dry_run)
        grand_success += s
        grand_total += t
        grand_skipped += sk

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(
        f"DONE: {grand_success}/{grand_total} extracted, "
        f"{grand_skipped} skipped — {elapsed:.0f}s total"
    )
    print(f"Cache root: {_cache_root()}")

    if not args.dry_run:
        manifest = {
            "total_videos": grand_total,
            "extracted": grand_success,
            "skipped": grand_skipped,
            "classes": classes,
        }
        mp = os.path.join(_cache_root(), "manifest.json")
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        log("PREEXTRACT", f"Manifest → {mp}")


if __name__ == "__main__":
    main()
