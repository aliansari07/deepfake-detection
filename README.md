# Deepfake Detection System

**Trusting Digital Media — a dual-stream, gated, temporal deepfake video detector.**

This repository contains the model, training and evaluation code, and a web demo for a deepfake
video detector built as a final-year project at FAST NUCES Karachi. The detector looks at faces
across a whole clip in two ways at once: the ordinary image (spatial stream) and its frequency
content (frequency stream), where compression and blending artifacts tend to show. A learned gate
decides, per feature, how much to trust each stream, and a Transformer reasons over the sequence
of frames before a final verdict. Grad-CAM heatmaps show which facial regions drove the decision.

---

## Key features

- **Dual-stream feature extraction.** Two EfficientNet-B0 backbones: one on RGB face crops, one
  on a frequency map of the same crops (log-magnitude FFT by default, DCT or a high-pass spatial
  residual as options).
- **Gated fusion.** A small gate network computes `F = g ⊙ s + (1 − g) ⊙ f` per feature, so the
  model can lean on whichever stream is more informative for a given input. The mean gate value
  is reported with every prediction.
- **Temporal reasoning.** Ten evenly spaced face crops per video pass through a 2-layer, 8-head
  Transformer encoder with learnable positional embeddings, then mean pooling.
- **Compression-aware training.** Per-video JPEG, blur, resize and noise augmentation simulates
  social-media re-encoding.
- **Explainability.** Grad-CAM heatmaps on the spatial branch, plus an optional plain-language
  explanation of each verdict (offline template by default, or the Anthropic API if configured).
- **Reproducible evaluation.** Committed, identity-disjoint data splits; bootstrap confidence
  intervals; a locked test split that logs every access; checkpoints that record their own
  architecture, split hash and git commit.
- **Web demo.** A FastAPI backend and a Next.js front end: upload a video, get a verdict,
  confidence, gate weights and a heatmap.

## Results

The model served by the web demo (run `M1-20`, the full gated dual-stream model with the Transformer):

| Benchmark | n | AUC (95% CI) | Accuracy |
|---|---|---|---|
| FaceForensics++ C23, identity-disjoint test split | 840 | 0.912 (0.889–0.933) | 75.8% (balanced 81.5%) |
| Celeb-DF v2, official 518-video test list (cross-dataset) | 518 | 0.753 (0.710–0.796) | 72.4% |

These figures come from [`website/lib/modelStats.json`](website/lib/modelStats.json), which is
generated from the evaluation outputs by `tools/make_site_stats.py`.

---

## Architecture

```
video ──► sample 10 frames ──► InsightFace SCRFD face crop (224×224)
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
          Spatial stream                         Frequency stream
          EfficientNet-B0 (RGB)                  FFT / DCT / residual map
                     │  s ∈ ℝ¹²⁸⁰                ──► EfficientNet-B0 (1-channel stem)
                     │                                     │  f ∈ ℝ¹²⁸⁰
                     └──────────────► Gated fusion ◄───────┘
                                g = σ(W₂·ReLU(W₁·[s; f]))
                                F = g ⊙ s + (1 − g) ⊙ f
                                        │
                     Linear 1280→512 + LayerNorm + GELU
                                        │
              + positional embeddings ─► Transformer encoder (2 layers, 8 heads)
                                        │  masked mean pool over frames
                                        ▼
                         MLP head 512→256→1 ──► p(fake)
```

About 16.1 M parameters. Ablation switches drop the frequency branch (`--no-freq`), the
Transformer (`--no-transformer`), or both (spatial-only baseline) without changing the
checkpoint's key set. A module-by-module reference with code locations is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Repository layout

```
.
├── src/                    Core library
│   ├── config.py           All paths and hyperparameters (edit here first)
│   ├── model.py            DeepfakeDetector, frequency transforms, checkpoint architecture
│   ├── dataset.py          DeepfakeDataset and DataLoaders (reads a committed split file)
│   ├── preprocess.py       Frame sampling and face extraction for inference
│   ├── detector.py         InsightFace SCRFD, built lazily on first use
│   ├── face_cache.py       Face cache v2: extraction and loading
│   ├── augmentation.py     Compression-aware augmentation
│   ├── train.py            Training loop (AMP, gradient accumulation, AUC tracking)
│   ├── inference.py        run_pipeline(): video → verdict
│   ├── gradcam.py          Grad-CAM heatmap grids
│   ├── explain.py          Plain-language verdict explanations
│   ├── metrics.py          Bootstrap confidence intervals and paired comparison
│   ├── analysis.py         Fusion-gate analysis, DCT/FFT comparison, ablations
│   ├── celebdf.py          Celeb-DF v2 video lists
│   └── test_access.py      Locked test-split access log
├── tools/                  Utility scripts (split building, model promotion, result tables)
├── tests/                  pytest suite (runs without datasets or a GPU)
├── website/                Next.js front end
│   └── backend/            FastAPI service wrapping src/inference.py
├── splits/                 Committed train/val/test split files (see splits/README.md)
├── docs/ARCHITECTURE.md    Module-by-module architecture reference
├── train.py                Train a model
├── infer.py                Run inference on one video
├── evaluate.py             FF++ validation/test metrics
├── cross_eval.py           Celeb-DF v2 cross-dataset evaluation
├── analyse.py              Gate analysis and ablation tools
├── preextract.py           Pre-extract face crops to a cache
├── freq_ablate.py          Force the fusion gate to test the frequency branch
├── compress_test.py        JPEG robustness test
├── compress_video_test.py  H.264 (CRF) video-level robustness test
├── heatmap_demo.py         Quick Grad-CAM demo on any video
├── resume_preextract.py    Show or resume an interrupted extraction
├── requirements.txt        Python dependencies
├── requirements-lock.txt   Full pinned environment
├── .env.example            Environment variables
└── LICENSE                 Apache 2.0
```

---

## Prerequisites

- **Python 3.10+** (developed and tested on 3.11)
- **An NVIDIA GPU with CUDA 11.8** for training. Inference and the test suite also run on CPU,
  slowly.
- **Node.js 20+** for the web front end
- About **300 MB** of disk space for the InsightFace `buffalo_l` model pack, which downloads
  automatically on first use to `~/.insightface/models/`
- For training and evaluation, the datasets (not included; see [Datasets](#datasets))

## Installation

```bash
git clone https://github.com/aliansari07/deepfake-detection.git
cd deepfake-detection

python -m venv venv
# Windows:      venv\Scripts\activate
# Linux/macOS:  source venv/bin/activate

# 1. PyTorch with CUDA 11.8 (see requirements.txt for the CPU-only line)
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 2. Everything else
pip install -r requirements.txt

# 3. Check the install
pytest
```

The test suite needs no datasets, checkpoints or GPU. Two tests skip themselves when the face
cache is absent.

## Model checkpoint

Trained weights are not stored in git. They are published as assets on the repository's
[GitHub Releases](https://github.com/aliansari07/deepfake-detection/releases) page.

1. Download `demo.pt` (the model the web demo serves) from the latest release.
2. Place it at `models/demo.pt`:

   ```bash
   mkdir -p models
   # move the downloaded file to models/demo.pt
   ```

Every checkpoint records its own architecture, so the scripts below configure the model from the
file automatically. Pass any checkpoint with `--checkpoint <path>`.

---

## Usage

All commands run from the repository root with the virtual environment active.

### Inference on a video

```bash
python infer.py --video path/to/video.mp4 --checkpoint models/demo.pt
```

This prints the verdict (REAL / FAKE / UNCERTAIN), the probability, the spatial/frequency gate
weights and the timing, and saves a Grad-CAM grid to `gradcam/`. For a heatmap only:

```bash
python heatmap_demo.py --video path/to/video.mp4 --checkpoint models/demo.pt
```

### Web interface

The demo has two processes: the Python API and the Next.js front end.

```bash
# Terminal 1: API on http://localhost:8000 (serves models/demo.pt by default)
uvicorn website.backend.main:app --port 8000

# Terminal 2: front end on http://localhost:3000
cd website
npm install
cp .env.local.example .env.local   # points the front end at the API
npm run dev
```

Open <http://localhost:3000> and upload an MP4, MOV or WEBM file. `GET /health` on the API
reports which checkpoint is being served and its hash. The backend reads the settings listed in
[`.env.example`](.env.example) from the environment, for example:

- `DD_CHECKPOINT` to serve a different checkpoint
- `DD_ALLOWED_ORIGINS` to allow a front end hosted elsewhere
- `DD_EXPLAIN_PROVIDER=claude` with `ANTHROPIC_API_KEY` for AI-written explanations

### Datasets

The datasets are not redistributed here. Request access from their authors:

- **FaceForensics++ (C23)**: <https://github.com/ondyari/FaceForensics>
- **Celeb-DF v2**: <https://github.com/yuezunli/celeb-deepfakeforensics>

Expected layout:

```
datasets/
├── FaceForensics++_C23/
│   ├── original/            *.mp4
│   ├── Deepfakes/           *.mp4
│   ├── Face2Face/           *.mp4
│   ├── FaceSwap/            *.mp4
│   ├── NeuralTextures/      *.mp4
│   └── FaceShifter/         *.mp4
└── Celeb-DF-v2/
    ├── Celeb-real/  YouTube-real/  Celeb-synthesis/
    └── List_of_testing_videos.txt
```

### Training

Training reads pre-extracted face crops, so extract them once, then train:

```bash
# 1. Extract faces to datasets/FaceForensics++_C23/cache_v2/ (resume-safe)
python preextract.py --cache-version 2 --workers 8

# 2. Train on the identity-disjoint split
#    PowerShell: $env:DD_SPLIT_FILE="splits/ffpp_official_v1.json"
export DD_SPLIT_FILE=splits/ffpp_official_v1.json
python train.py M1-20 --freq-norm per_image

# Resume an interrupted run
python train.py M1-20 --freq-norm per_image --resume
```

The run id is a name plus the number of epochs (`M1-20` means 20 epochs). Checkpoints go to
`checkpoints/<run>.best.pt`. Hyperparameters live in [`src/config.py`](src/config.py).

**Ablations** (each is recorded in the checkpoint):

| Flag | Effect |
|---|---|
| `--no-freq` | Remove the frequency branch and the gate |
| `--no-transformer` | Mean-pool frames instead of the Transformer |
| `--no-freq --no-transformer` | Spatial-only baseline |
| `--freq-mode dct` | DCT instead of FFT |
| `--freq-input residual` | High-pass spatial residual instead of a spectrum |
| `--freq-augment gentle` | Lighter augmentation for the frequency branch |
| `--no-augment` | Disable compression-aware augmentation |
| `--seed N` | Change the random seed |

> The default split, `splits/legacy_t4-20.json`, exists only to reproduce the project's
> first-phase results. It overlaps identities between train and validation, so use
> `splits/ffpp_official_v1.json` for new work. See [`splits/README.md`](splits/README.md).

### Evaluation

```bash
# FF++ validation metrics: AUC, balanced accuracy, per-manipulation breakdown, bootstrap CIs
python evaluate.py --checkpoint checkpoints/M1-20.best.pt

# Final test-split evaluation (locked; every access is logged to results/TEST_ACCESS_LOG.md)
python evaluate.py --checkpoint checkpoints/M1-20.best.pt --split test --final

# Cross-dataset: Celeb-DF v2 official 518-video list
python cross_eval.py --checkpoint checkpoints/M1-20.best.pt

# Robustness and analysis
python compress_test.py --checkpoint <ckpt>         # JPEG quality sweep
python compress_video_test.py --checkpoints <ckpt>  # H.264 CRF sweep
python freq_ablate.py --checkpoint <ckpt>           # gate modes: full / spatial / freq
python analyse.py gate --checkpoint <ckpt>          # gate weights, real vs fake
python tools/compare_runs.py --help                 # paired comparison of two runs
```

### Development

```bash
pytest           # test suite
ruff check .     # lint
ruff format .    # format
```

Before changing behaviour, keep these rules in mind:

- `CONFIG["FREQ_NORM"]` defaults to `"none"` so that first-phase checkpoints reproduce exactly.
  `tests/assets/freq_legacy_ref.pt` pins that output, so never regenerate it.
- `CONFIG["AUGMENT_ENABLED"]` (inference-time augmentation) must stay `False`, or verdicts become
  non-deterministic.
- Always pass the dataset's `pad_mask` to `model(frames, pad_mask)`.

---

## License

Released under the [Apache License 2.0](LICENSE).
Copyright 2026 Sabih Uddin Meraj, Muhammad Hasan, Muhammad Ali Ansari (FAST NUCES Karachi).

The datasets and the InsightFace model pack are not covered by this license. Each has its own
terms, so check them before use.

## Acknowledgments

- **FaceForensics++**: Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial
  Images*, ICCV 2019.
- **Celeb-DF**: Li et al., *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics*,
  CVPR 2020.
- **InsightFace / SCRFD**: Guo et al., *Sample and Computation Redistribution for Efficient Face
  Detection*, 2021.
- **EfficientNet**: Tan and Le, ICML 2019, via the [`timm`](https://github.com/huggingface/pytorch-image-models) library.
- **Grad-CAM**: Selvaraju et al., ICCV 2017.
- Built with PyTorch, FastAPI and Next.js.

Final-year project, FAST NUCES Karachi.
