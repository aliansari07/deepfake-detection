"""DeepfakeDataset + build_dataloaders() for FaceForensics++.
Loads pre-extracted .pt cache files directly — no InsightFace
during training.
"""

import glob
import hashlib
import json
import os
import tarfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.config import CONFIG, TRAIN_CONFIG
from src.face_cache import load_frames, to_sequence
from src.logger import debug, log

_CACHE_DESC = (
    "Expected one .pt per video holding a float32 tensor of shape "
    "(SAMPLE_COUNT, 3, 224, 224). Re-extract the cache (preextract.py) if mixed."
)


class DeepfakeDataset(Dataset):
    """
    Dataset class for FaceForensics++_C23.

    Loads pre-extracted .pt face tensors from cache/.
    Falls back to raw video only if .pt not found.

    Cache layout:
        DATASET_ROOT/cache/original/*.pt        label=0
        DATASET_ROOT/cache/Deepfakes/*.pt       label=1
        DATASET_ROOT/cache/Face2Face/*.pt       label=1
        DATASET_ROOT/cache/FaceSwap/*.pt        label=1
        DATASET_ROOT/cache/NeuralTextures/*.pt  label=1
        DATASET_ROOT/cache/DeepFakeDetection/*.pt label=1
        DATASET_ROOT/cache/FaceShifter/*.pt     label=1
    """

    FAKE_SUBDIRS = [
        "Deepfakes",
        "Face2Face",
        "FaceSwap",
        "NeuralTextures",
        "DeepFakeDetection",
        "FaceShifter",
    ]

    def __init__(self, root, split="train", augment=False, split_file=None, allow_test=False):
        self.root = root
        self.split = split

        # The test split is for the final, one-time evaluation only
        # (FIX_PLAN P4.2); model selection must never touch it.
        if split == "test" and not allow_test:
            raise PermissionError(
                "The test split is locked. Use split='val' for model selection; "
                "pass allow_test=True only for the final evaluation."
            )

        if augment:
            from src.augmentation import CompressionAwareAugmentation

            self.augment = CompressionAwareAugmentation(seed=TRAIN_CONFIG["SEED"])
        else:
            self.augment = None

        # The split comes from a committed JSON file, never from shuffling the
        # cache listing: adding files to the cache used to re-draw the split and
        # leak training videos into validation (audit-fyp2.md §5.1).
        self.split_file = split_file or CONFIG.get("SPLIT_FILE")
        self.spec = self._read_spec(self.split_file)
        # 1 = legacy BGR float32 cache/ (repeat-padded); 2 = RGB uint8 cache_v2/
        self.cache_version = int(self.spec.get("cache_version", 1))
        self.channel_order = "RGB" if self.cache_version >= 2 else "BGR"

        if self.cache_version == 1:
            self._verify_manifest(root)
        self.samples = self._load_split(root, self.split_file, split, self.spec)

        n_r = sum(1 for _, l in self.samples if l == 0)
        n_f = sum(1 for _, l in self.samples if l == 1)
        log(
            "DATASET",
            f"{split.upper()}: {len(self.samples)} samples "
            f"({n_r} real / {n_f} fake) | {os.path.basename(self.split_file)} | "
            f"cache v{self.cache_version} {self.channel_order}",
        )

    # ── Cache validation / manifest (8.2, 8.3) ────────────────
    @classmethod
    def _manifest_path(cls, root):
        return os.path.join(root, "cache", "cache_manifest.json")

    def _find(self, subdir, label):
        """Look in cache/ for .pt files. Skip if folder missing."""
        cache_dir = os.path.join(self.root, "cache", subdir)
        if not os.path.exists(cache_dir):
            debug(f"Cache folder not found (skipped): {cache_dir}")
            return []
        pts = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        if not pts:
            debug(f"No .pt files in: {cache_dir}")
            return []
        # Fail loudly on empty/corrupt files instead of training on garbage [8.2]
        bad = [p for p in pts if os.path.getsize(p) == 0]
        if bad:
            raise RuntimeError(
                f"{len(bad)} empty/corrupt cache files in {cache_dir} (first: "
                f"{os.path.basename(bad[0])}). Delete or re-extract them before "
                f"training / evaluation. {_CACHE_DESC}"
            )
        return [(p, label) for p in pts]

    def _verify_manifest(self, root):
        """Regenerate the cache_manifest.json when missing or stale. Raises on
        any empty/corrupt .pt so garbage never silently reaches training [8.3].
        Cheap by design: structural validation + file sizes only, NO sha256
        pass, so dataset init stays fast on 7,000 cached tensors."""
        manifest_path = self._manifest_path(root)
        cache_root = os.path.join(root, "cache")
        subdirs = ["original"] + self.FAKE_SUBDIRS

        total = sum(
            len(glob.glob(os.path.join(cache_root, sub, "*.pt")))
            for sub in subdirs
            if os.path.isdir(os.path.join(cache_root, sub))
        )

        # Fast path: trust an existing manifest when the file count matches.
        # Deep corruption is still caught at torch.load() in __getitem__.
        if os.path.exists(manifest_path) and total > 0:
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                if m.get("count") == total:
                    log("CACHE", f"Manifest trusted ({total} files) — skipping per-file scan.")
                    return {"version": 1, "files": m.get("files", {}), "count": total}
            except Exception:
                pass  # fall through to a full scan

        t0 = time.time()
        entries = {}
        scanned = 0
        for sub in subdirs:
            sub_dir = os.path.join(cache_root, sub)
            if not os.path.isdir(sub_dir):
                continue
            for p in sorted(glob.glob(os.path.join(sub_dir, "*.pt"))):
                size = os.path.getsize(p)
                if size == 0:
                    raise RuntimeError(
                        f"Empty/corrupt cache file: {p}. "
                        f"Delete or re-extract before training. {_CACHE_DESC}"
                    )
                self._check_structure(p)
                rel = os.path.relpath(p, cache_root).replace(os.sep, "/")
                entries[rel] = {"size": size}
                scanned += 1
                if scanned % 1000 == 0:
                    log("CACHE", f"Scanned {scanned} cache files ({100.0 * scanned / total:.0f}%)")
        count = len(entries)
        log("CACHE", f"Scanned {count} cache files in {time.time() - t0:.1f}s ({100.0:.0f}%)")

        old_entries = {}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    old_entries = json.load(f).get("files", {})
            except Exception:
                old_entries = {}

        # Up to date only when the exact on-disk size map matches.
        if old_entries.get("sizes") == {k: v["size"] for k, v in entries.items()}:
            log("CACHE", "Manifest up to date — no re-scan write needed.")
            return {"version": 1, "files": entries, "count": count}

        manifest = {
            "version": 1,
            "files": entries,
            "sizes": {k: v["size"] for k, v in entries.items()},
            "count": count,
        }
        os.makedirs(cache_root, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        log("CACHE", f"cache_manifest.json updated ({manifest['count']} files).")
        return manifest

    @staticmethod
    def _check_structure(path):
        """Cheap structural validation that the archive header is intact.
        Reads only the 4-byte magic (PK = zip, else tar), NOT the whole
        file. Deep corruption is still caught at torch.load() time in
        __getitem__ and during preextract verification [8.2 / 8.3]."""
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic == b"PK\x03\x04":
            return
        try:
            with tarfile.open(path) as t:
                names = t.getnames()
            if "storages" not in names or "data.pkl" not in names:
                raise RuntimeError(
                    f"Archive corrupt (missing member): {path}. "
                    f"Delete or re-extract before training. {_CACHE_DESC}"
                )
        except tarfile.TarError as e:
            raise RuntimeError(
                f"Archive unreadable: {path} — {e}. "
                f"Delete or re-extract before training. {_CACHE_DESC}"
            ) from e

    @staticmethod
    def _sha256(path, chunk=1 << 20):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                block = f.read(chunk)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()

    @staticmethod
    def _read_spec(split_file):
        if not split_file or not os.path.isfile(split_file):
            raise FileNotFoundError(
                f"Split file not found: {split_file}. Set CONFIG['SPLIT_FILE'] — "
                f"runtime shuffling of the cache is disabled to prevent split drift."
            )
        with open(split_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_split(root, split_file, split, spec=None):
        """Read (path, label) pairs for `split` from a split JSON, in stored order."""
        if spec is None:
            spec = DeepfakeDataset._read_spec(split_file)
        if split not in spec or not spec[split]:
            raise ValueError(f"Split '{split}' is empty or missing in {split_file}")

        cache_root = os.path.join(root, spec.get("cache_dir", "cache"))
        pairs = [
            (os.path.join(cache_root, *rel.split("/")), 0 if rel.startswith("original/") else 1)
            for rel in spec[split]
        ]
        missing = [p for p, _ in pairs if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} files listed in {os.path.basename(split_file)} "
                f"['{split}'] are missing from the cache (first: {missing[0]}). "
                f"Re-extract them (preextract.py) or use the matching split file."
            )
        return pairs

    @staticmethod
    def _do_split(items, ratio, rng):
        items = list(items)
        rng.shuffle(items)
        n = int(len(items) * ratio)
        return items[:n], items[n:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns (frames (T,3,224,224) float [0,1], label, pad_mask (T,) bool).
        pad_mask is True for padded positions; it is all False for the legacy
        cache, whose short clips are padded by repeating the last frame."""
        path, label = self.samples[idx]
        T = CONFIG["SAMPLE_COUNT"]

        if self.cache_version >= 2:
            try:
                frames, _ = load_frames(path)
            except Exception as e:
                raise RuntimeError(f"Failed to load cache v2 file {path}: {e}") from e
            if frames.shape[0] == 0:
                raise RuntimeError(
                    f"{path} has no detected faces; it should have been excluded by the split file."
                )
            frames, pad_mask = to_sequence(frames, T)
            k = int((~pad_mask).sum())
        else:
            frames = self._load_legacy(path, T)
            pad_mask = torch.zeros(T, dtype=torch.bool)
            k = T

        # Augmentation (train only); padded zero frames stay zero.
        # One parameter set per clip, not per frame: a real re-encode applies
        # the same quality to every frame of a video (FIX_PLAN P3.3).
        label_t = torch.tensor(label, dtype=torch.long)
        if self.augment is None:
            return frames, label_t, pad_mask

        freq_profile = CONFIG.get("FREQ_AUGMENT", "full")
        if freq_profile == "full":
            # Both branches see the same frames: the pre-R4 path, unchanged.
            frames = torch.cat([self.augment.augment_video(frames[:k]), frames[k:]], dim=0)
            return frames, label_t, pad_mask

        # R4: the frequency branch gets its own, gentler view of the same clip. Only the training
        # split reaches this branch, so validation, evaluation and inference still yield the
        # 3-tuple every other script unpacks.
        spatial, freq = self.augment.augment_video_pair(frames[:k], freq_profile)
        return (
            torch.cat([spatial, frames[k:]], dim=0),
            label_t,
            pad_mask,
            torch.cat([freq, frames[k:]], dim=0),
        )

    @staticmethod
    def _load_legacy(path, T):
        try:
            frames = torch.load(path, weights_only=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load cache file {path}: {e}. "
                f"Delete/re-extract it — do NOT train on silent zeros. {_CACHE_DESC}"
            ) from e

        if (
            not isinstance(frames, torch.Tensor)
            or frames.dim() != 4
            or frames.shape[1:] != (3, 224, 224)
        ):
            raise RuntimeError(
                f"Unexpected tensor shape {tuple(frames.shape)} in {path}. {_CACHE_DESC}"
            )

        # Ensure correct sequence length (legacy: repeat the last frame)
        if frames.shape[0] < T:
            pad = T - frames.shape[0]
            frames = torch.cat([frames, frames[-1:].repeat(pad, 1, 1, 1)], dim=0)
        elif frames.shape[0] > T:
            frames = frames[:T]
        return frames

    def get_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler (class balancing)."""
        labels = [l for _, l in self.samples]
        counts = np.bincount(labels)
        weights = [1.0 / counts[l] for l in labels]
        return torch.tensor(weights, dtype=torch.float)


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker its own augmentation stream.

    Each worker gets a copy of the dataset, so without this they all inherit
    the same TRAIN_CONFIG["SEED"] and draw identical augmentation parameters —
    four workers producing one stream instead of four (audit §5.2). PyTorch
    hands each worker a distinct, generator-derived `seed`, which makes the
    streams different from each other and reproducible across runs.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    aug = getattr(info.dataset, "augment", None)
    if aug is not None:
        aug.reseed(info.seed)


def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    """Build the train and validation DataLoaders from ``CONFIG`` / ``TRAIN_CONFIG``."""
    train_ds = DeepfakeDataset(
        CONFIG["DATASET_ROOT"], split="train", augment=TRAIN_CONFIG["TRAIN_AUGMENT"]
    )
    val_ds = DeepfakeDataset(CONFIG["DATASET_ROOT"], split="val", augment=False)

    # Seeded generator: fixes the sampler's draw order and the per-worker seeds,
    # so a run is reproducible from TRAIN_CONFIG["SEED"] alone.
    generator = torch.Generator().manual_seed(TRAIN_CONFIG["SEED"])

    sampler = WeightedRandomSampler(
        weights=train_ds.get_weights(),
        num_samples=len(train_ds),
        replacement=True,
        generator=generator,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_CONFIG["BATCH_SIZE"],
        sampler=sampler,
        num_workers=TRAIN_CONFIG["NUM_WORKERS"],
        pin_memory=True,
        persistent_workers=TRAIN_CONFIG["NUM_WORKERS"] > 0,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=TRAIN_CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=TRAIN_CONFIG["NUM_WORKERS"],
        pin_memory=True,
        persistent_workers=TRAIN_CONFIG["NUM_WORKERS"] > 0,
    )

    log(
        "DATALOADER",
        f"Train: {len(train_loader)} batches | "
        f"Val: {len(val_loader)} batches | "
        f"Batch: {TRAIN_CONFIG['BATCH_SIZE']}",
    )
    return train_loader, val_loader
