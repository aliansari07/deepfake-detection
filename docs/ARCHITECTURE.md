# DeepfakeDetector v7.0 — Architecture Module Reference

Generated from the actual codebase. Each module is keyed by its diagram ID (M1–M15, A1–A2) with corrected names, descriptions, and exact code locations. Discrepancies from the original diagram are flagged.

---

## Overall Pipeline

```
Video Input → [PREPROCESSING] → [FEATURE EXTRACTION] → [FUSION & TEMPORAL] → Output
```

Solid arrows = data flow. Dashed arrows = auxiliary annotations (training/evaluation methodology, not pipeline steps).

---

## STAGE 1 — PREPROCESSING

### M1 · Video Input
- **Description:** Entry point — path to a `.mp4` video file.
- **Code:** `probe_video(video_path)` — `src/inference.py:46`
- **Consumes:** Raw video file path from caller (`infer.py`, `website/backend/main.py`, or `batch_infer()`).

### M2 · Frame Sampling
- **Description:** Samples **N evenly-spaced frames** across the full video duration using `np.linspace`. Default N=10 (`CONFIG["SAMPLE_COUNT"]`). Not FPS-based — spans the entire video regardless of frame rate.
- **Code:** `np.linspace(0, total_frames - 1, sample_count, dtype=int)` — `src/preprocess.py:64`
- **Config:** `CONFIG["SAMPLE_COUNT"]` = 10
- **Output:** Frame indices → fed to face detection.

> **Diagram correction:** The diagram says "10–15 FPS". The implementation does **not** resample by FPS. It takes exactly N=10 frames uniformly distributed across the video. Correct to: **"N evenly-spaced frames (N=10)"**.

### M3 · Face Detection & Cropping
- **Description:** Runs **InsightFace SCRFD** on each sampled frame. Selects the highest-confidence face above `DETECTION_THRESHOLD`. Crops the bounding box, resizes to 224×224, converts to a float32 [0,1] tensor in OpenCV **BGR** channel order (FYP-1 cache; to become RGB when the cache is rebuilt).
- **Code:** `face_app.get(frame)` → crop → resize → tensor — `src/preprocess.py:78–98`
- **Detector config:** `src/detector.py` (InsightFace `buffalo_l` model)
- **Config keys:** `CONFIG["DETECTION_THRESHOLD"]` = 0.5, `CONFIG["DET_SIZE"]` = (640, 640), `CONFIG["FACE_SIZE"]` = (224, 224)
- **Output:** Per-frame face crop tensor.

> **Diagram correction:** The diagram says "(RetinaFace/MediaPipe)". The implementation uses **InsightFace SCRFD**. Correct to: **"(InsightFace SCRFD)"**.

### M4 · Face Frames
- **Description:** Stacked face tensor of shape `(N, 3, 224, 224)`, float32, values in [0, 1]. This is the preprocessed output of Stage 1, ready for feature extraction.
- **Code:** `torch.stack(faces_extracted)` — `src/preprocess.py:119`
- **Output:** Tensor passed to augmentation (if enabled) then to the model's `forward()`.

---

### ⚠ NOT IN DIAGRAM: Compression-Aware Augmentation

Sits between M4 (preprocessing output) and Stage 2 (feature extraction). Applied during training only (always on in `build_dataloaders()`). `CONFIG["AUGMENT_ENABLED"]` controls inference-time augmentation and defaults to `False`, because random distortions made verdicts non-deterministic.

- **Class:** `CompressionAwareAugmentation` — `src/augmentation.py:37`
- **Transformations** (each applied stochastically per frame, p=0.5):
  1. JPEG compression — quality 30–95 (`cv2.imencode(".jpg", ...)`)
  2. Gaussian blur — kernel 3×3 / 5×5 / 7×7
  3. Random resize — scale 50–90%, then resize back to 224×224
  4. Gaussian noise — sigma 1–5% of pixel range
- **Config:** training: always on. `CONFIG["AUGMENT_ENABLED"]` = False (inference only; keep False).
- **Called from:** `src/inference.py:71–84` (inference) and `src/dataset.py:258–259` (training via `__getitem__`).

> **Diagram note:** This step should appear between M4 and Stage 2, either as a solid box (training) or dashed auxiliary annotation.

---

## STAGE 2 — FEATURE EXTRACTION

Two parallel sub-pipelines sharing the same input (M4 face frames).

### M5 · CNN Backbone (Spatial)
- **Description:** EfficientNet-B0 pretrained on ImageNet. Takes the 3-channel RGB face frame, normalised with ImageNet mean/std. Outputs a 1280-dimensional feature vector per frame.
- **Code:** `self.spatial = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)` — `src/model.py:132`
- **Normalisation:** `_spatial_normalize(x)` — `src/model.py:81` (subtracts ImageNet mean, divides by std)
- **Forward path:** `_spatial_features(flat)` → `self.spatial(_spatial_normalize(flat))` — `src/model.py:165–166`
- **Output:** `s` — tensor of shape `(B*N, 1280)`

### M6 · f_rgb(t)
- **Description:** Spatial feature vector per face frame — the output of the RGB CNN backbone.
- **Code:** `s` variable in `DeepfakeDetector.forward()` — `src/model.py:174`
- **Shape:** `(B*N, 1280)`

### M7 · Frequency Transform
- **Description:** Converts RGB face frames to a single-channel frequency-domain representation. Two modes controlled by `CONFIG["DCT_ENABLED"]`:
  - **FFT (default, False):** grayscale → `fft2` → `fftshift` → `log(|FFT| + ε)`
  - **DCT [ablation B4] (True):** grayscale → orthonormal 2D DCT-II magnitude
- **Code:** `frequency_transform(x)` — `src/model.py:43`
- **Helper:** `_dct_2d(x)` — `src/model.py:38`; `_dct_1d(x, dim)` — `src/model.py:24`
- **Config:** `CONFIG["DCT_ENABLED"]` = False (default = FFT); `CONFIG["FREQ_EPS"]` = 1e-6
- **Grayscale weights:** 0.299R + 0.587G + 0.114B (standard luminance)
- **Output:** `(B, 1, H, W)` single-channel frequency map

### M8 · CNN Backbone (Frequency)
- **Description:** EfficientNet-B0 with a **modified stem**: the first Conv2d is changed from 3-channel to 1-channel input. Pretrained weights are preserved by averaging the original 3-channel stem weights across channels.
- **Code:** `_efficientnet_freq_backbone()` — `src/model.py:63–78`
- **Forward path:** `_frequency_features(flat)` → `self.freq(frequency_transform(flat))` — `src/model.py:168–169`
- **Output:** `f` — tensor of shape `(B*N, 1280)`

### M9 · f_freq(t)
- **Description:** Frequency feature vector per face frame — the output of the frequency CNN backbone.
- **Code:** `f` variable in `DeepfakeDetector.forward()` — `src/model.py:182`
- **Shape:** `(B*N, 1280)`

---

### A1 · Compression-Robust Training (auxiliary, dashed)

Training-time data augmentation applied to both backbones' input frames. Not a pipeline stage — a methodology note.

- **Code:** `CompressionAwareAugmentation` — `src/augmentation.py:37`
- **Config:** always on for training (`DeepfakeDataset(augment=True)`)

> **Diagram correction:** The diagram says "(H.264/H.265, bitrate reduction, resize, fps drop)". The actual augmentations are **JPEG compression, Gaussian blur, random resize, and Gaussian noise** — there is no H.264/H.265 codec simulation or fps drop. Correct to: **"(JPEG compression, Gaussian blur, random resize, Gaussian noise)"**.

---

## STAGE 3 — FUSION & TEMPORAL

### M10 · Gated Fusion
- **Description:** Per-frame gated fusion of spatial and frequency features. Concatenates `s` and `f` (2560 total), passes through a two-layer gate network, produces a 1280-dim gate vector `g`. Fused output: `fused = g⊙s + (1-g)⊙f`.
- **Gate network:** `fusion_gate_net()` — `src/model.py:89–101`
  - `Linear(2560 → 256) → ReLU → Linear(256 → 1280) → Sigmoid`
- **Forward:** `g = self.gate(torch.cat([s, f], dim=-1))` → `fused = g * s + (1.0 - g) * f` — `src/model.py:184–185`
- **Output:** `fused` — tensor of shape `(B*N, 1280)`

### M11 · Sequence of Fused Features
- **Description:** Each fused feature vector is projected from 1280 → 512 dimensions, then LayerNorm + GELU applied. Learnable positional embeddings are added to encode frame order. Assembled into a sequence of length N.
- **Projection:** `self.project = Linear(1280 → 512) → LayerNorm(512) → GELU` — `src/model.py:137–141`
- **Positional embeddings:** `self.pos_emb = nn.Parameter(zeros(1, N, 512))` — `src/model.py:143–145`
- **Config:** `TRAIN_CONFIG["POSITIONAL_EMB"]` = True; `TRAIN_CONFIG["D_MODEL"]` = 512
- **Code:** `proj = self.project(fused).view(b, n, -1)` then `proj = proj + self.pos_emb[:, :n, :]` — `src/model.py:187–191`
- **Output:** `proj` — shape `(B, N, 512)`

### M12 · Transformer Encoder
- **Description:** Standard `nn.TransformerEncoder` performing multi-head self-attention across the N frame features. Learns temporal correlations between frames.
- **Code:** `self.transformer` — `src/model.py:147–150`
- **Architecture:** 2 layers, 8 heads, d_model=512, d_ff=2048, GELU activation, dropout=0.1
- **Config:** `TRAIN_CONFIG["N_LAYERS"]` = 2; `TRAIN_CONFIG["NHEAD"]` = 8; `TRAIN_CONFIG["D_FF"]` = 2048
- **Code:** `ctx = self.transformer(proj)` — `src/model.py:193`
- **Output:** `ctx` — shape `(B, N, 512)`

### M13 · Video-level Representation
- **Description:** Mean-pools the transformer output across the N frame dimension to produce a single fixed-length vector per video.
- **Code:** `pooled = ctx.mean(dim=1)` — `src/model.py:194`
- **Output:** `pooled` — shape `(B, 512)`

### M14 · MLP Classification Head
- **Description:** Two-layer MLP that maps the video-level representation to a single scalar logit. Sigmoid is applied by the caller to get probability.
- **Code:** `self.head` — `src/model.py:152–157`
- **Layers:** `Linear(512 → 256) → GELU → Dropout(0.3) → Linear(256 → 1)`
- **Config:** `TRAIN_CONFIG["DROPOUT"]` = 0.3
- **Code:** `logit = self.head(pooled)` — `src/model.py:195`
- **Output:** `logit` — shape `(B, 1)`

### M15 · Output: Real/Fake + Confidence Score
- **Description:** Sigmoid of logit gives probability `p`. Prediction: `FAKE` if p > 0.5, else `REAL`. Confidence = `max(p, 1-p)`. If confidence < `CONFIDENCE_THRESHOLD` (0.6), verdict becomes `UNCERTAIN`. Also outputs gate weights (spatial vs frequency contribution), Grad-CAM heatmap image, and face extraction metadata.
- **Code:** `src/inference.py:91–163`
- **Verdict logic:**
  ```python
  prob = torch.sigmoid(logit).item()
  prediction = "FAKE" if prob > 0.5 else "REAL"
  confidence = prob if prediction == "FAKE" else 1.0 - prob
  if confidence < CONFIG["CONFIDENCE_THRESHOLD"]:
      result = "UNCERTAIN"
  ```
- **Config:** `CONFIG["CONFIDENCE_THRESHOLD"]` = 0.6
- **Full output dict keys:** `verdict`, `prob`, `logit`, `confidence`, `spatial_weight`, `freq_weight`, `faces_detected`, `faces_total`, `gradcam_image`, `analysis_time`

---

### A2 · Cross-Dataset Evaluation (auxiliary, dashed)

Methodology annotation: train on one dataset, test on another to measure generalisation.

> **Status: IMPLEMENTED for Celeb-DF v2.** `cross_eval.py` evaluates on the official 518-video Celeb-DF v2 test list (default) or a custom sample (`--protocol sample`). T4-20 official result: Acc 0.7317 / AUC 0.8049. DFDC not yet implemented.

---

## Ablation Configurations (not in diagram, but relevant)

| ID | Flag | Effect | Code |
|---|---|---|---|
| B4 | `CONFIG["DCT_ENABLED"]=True` | Swaps FFT frequency transform for 2D DCT-II | `src/model.py:55–57` |
| B6 | `CONFIG["BASELINE_MODE"]=True` | Removes frequency branch + Transformer; spatial-only with mean-pool | `src/model.py:176–180` |

Override with `override_config()` context manager — `src/config.py:91`.

---

## Key Discrepancy Summary

| Module | Diagram says | Corrected to |
|---|---|---|
| M2 | "10–15 FPS" | "N evenly-spaced frames (N=10)" — no FPS resampling |
| M3 | "(RetinaFace/MediaPipe)" | "(InsightFace SCRFD)" |
| A1 | "H.264/H.265, bitrate reduction, resize, fps drop" | "JPEG compression, Gaussian blur, random resize, Gaussian noise" |
| A2 | "Cross-Dataset Evaluation" | Implemented for Celeb-DF v2 (`cross_eval.py`); DFDC pending |
| M12 | — | Add: "2 layers, 8 heads, d=512, d_ff=2048" |
| M15 | "Real/Fake + Confidence Score" | "FAKE/REAL/UNCERTAIN + prob + gate weights + Grad-CAM + faces metadata" |
| (missing) | — | Add Compression-Aware Augmentation step between M4 and Stage 2 |

---

## File Map

| File | Purpose |
|---|---|
| `src/config.py` | All hyperparameters and paths |
| `src/model.py` | `DeepfakeDetector` class, `frequency_transform()`, `fusion_gate_net()` |
| `src/preprocess.py` | `extract_faces()` — frame sampling + InsightFace detection + crop |
| `src/detector.py` | InsightFace SCRFD `face_app` singleton |
| `src/augmentation.py` | `CompressionAwareAugmentation` — JPEG/blur/resize/noise |
| `src/inference.py` | `probe_video()`, `run_pipeline()`, `batch_infer()` — full inference loop |
| `src/dataset.py` | `DeepfakeDataset` — loads cached `.pt` face tensors for training |
| `src/train.py` | Training loop with AUC metrics |
| `src/gradcam.py` | Grad-CAM heatmap generation |
| `src/analysis.py` | Ablation tools (B3 gate analysis, B4 DCT/FFT, B6 baseline) |
| `evaluate.py` | Holdout evaluation (accuracy, AUC, F1, confusion matrix) |
| `website/backend/main.py` | FastAPI endpoint calling `probe_video()` |
