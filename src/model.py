"""DeepfakeDetector v7.0 — dual-stream spatial/frequency EfficientNet,
gated fusion, Transformer temporal reasoning, MLP head.

Import: from src.model import DeepfakeDetector, frequency_transform
"""

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CONFIG, TRAIN_CONFIG

# ImageNet normalisation for the pretrained EfficientNet spatial branch.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ── Frequency transforms ─────────────────────────────────────────


def _dct_1d(x, dim=-1):
    """Orthonormal 1D DCT-II via FFT (Makhoul-style, O(n log n))."""
    n = x.shape[dim]
    x = x.transpose(dim, -1)
    xpad = torch.cat([x, torch.zeros_like(x)], dim=-1)  # length 2n
    X = torch.fft.fft(xpad, dim=-1)[..., :n]  # bins 0..n-1
    k = torch.arange(n, device=x.device)
    tw = torch.exp(-0.5j * math.pi * k / n)  # e^{-jπk/2n}
    dct = (X * tw).real
    w = torch.ones(n, device=x.device) * math.sqrt(2.0 / n)
    w[0] = math.sqrt(1.0 / n)
    return (dct * w).transpose(dim, -1)


def _dct_2d(x):
    out = _dct_1d(x, dim=-1)
    return _dct_1d(out, dim=-2)


# R5's residual operator. These two constants must match tools/freq_probe.py's feat_residual
# (`cv2.GaussianBlur(g, (5, 5), 1.0)`), because R1's probe score on that exact representation —
# 0.674 against the spectrum's 0.561 — is the entire evidential basis for building R5. Changing
# them here silently detaches the experiment from the measurement that motivated it.
_RESIDUAL_KSIZE = 5
_RESIDUAL_SIGMA = 1.0


def _gaussian_kernel_1d(ksize, sigma, device, dtype):
    """Normalised 1D Gaussian, matching cv2.getGaussianKernel(ksize, sigma).

    cv2 only uses its small hard-coded table when sigma <= 0; given an explicit positive sigma it
    evaluates exp(-(i - c)^2 / (2 sigma^2)) and normalises, which is what this reproduces.
    """
    i = torch.arange(ksize, device=device, dtype=dtype)
    c = (ksize - 1) / 2.0
    k = torch.exp(-((i - c) ** 2) / (2.0 * sigma * sigma))
    return k / k.sum()


def highpass_residual(gray, ksize=_RESIDUAL_KSIZE, sigma=_RESIDUAL_SIGMA):
    """High-frequency spatial residual of a (B,1,H,W) map: the image minus its Gaussian blur.

    Separable and differentiable, with 'reflect' padding to match cv2's default
    BORDER_REFLECT_101, so the operator agrees with the probe that motivated R5 rather than
    merely resembling it.
    """
    k = _gaussian_kernel_1d(ksize, sigma, gray.device, gray.dtype)
    pad = ksize // 2
    blur = F.conv2d(F.pad(gray, (pad, pad, 0, 0), mode="reflect"), k.view(1, 1, 1, ksize))
    blur = F.conv2d(F.pad(blur, (0, 0, pad, pad), mode="reflect"), k.view(1, 1, ksize, 1))
    return gray - blur


def frequency_transform(
    x: torch.Tensor,
    mode: str | None = None,
    norm: str | None = None,
    freq_input: str | None = None,
) -> torch.Tensor:
    """
    Convert an image tensor (B, 3, H, W) in [0,1] to a single-channel
    frequency map (B, 1, H, W).

    `norm` selects between two behaviours (audit-fyp2.md §4.3, FIX_PLAN P3.1):

      "none" (legacy, the default)
          FFT : grayscale → fft2 → fftshift → log(|FFT| + eps)
          DCT : grayscale → |2D DCT-II|, with NO log
        These differ from each other by a log and by orders of magnitude in
        scale, which is exactly why the FYP-I "FFT vs DCT" comparison was not
        like-for-like. Kept unchanged so every FYP-1 checkpoint reproduces.

      "per_image" (FYP-2 runs)
          Both modes take the log, then are standardised per image to zero
          mean and unit variance. The spatial stream receives ImageNet-
          normalised input, so without this the frequency branch was the only
          stream fed un-normalised values whose scale varied per image — the
          leading explanation for its near-zero measured contribution.

    `freq_input` selects what the branch looks at at all (R5, audit §20.3):

      "spectrum" (the default)
          Everything described above — a log-magnitude FFT or DCT map.

      "residual"
          The high-frequency spatial residual, `gray - GaussianBlur(gray)`, instead of any
          Fourier transform. `mode` is then irrelevant and ignored. This exists because a
          spectrum is the wrong input for this backbone: absolute position carries the meaning
          in a Fourier magnitude image, so convolution's translation equivariance and ImageNet's
          pretrained filters are both mismatched to it (cause C1). A residual map is an ordinary
          image and suffers neither problem.

    Grayscale weights are the standard luminance ones, 0.299/0.587/0.114.
    They assume channel 0 is red, which holds for the RGB cache v2. Legacy
    checkpoints were trained on BGR with these same weights, so the expression
    is deliberately not "corrected" for them: it is what those weights learned.
    """
    mode = mode or ("dct" if CONFIG["DCT_ENABLED"] else "fft")
    norm = norm or CONFIG["FREQ_NORM"]
    freq_input = freq_input or CONFIG.get("FREQ_INPUT", "spectrum")

    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]  # (B,1,H,W)

    if freq_input == "residual":
        # .float() for the same reason as below: the transform must not run in bf16 autocast.
        res = highpass_residual(gray.float())
        if norm == "per_image":
            mu = res.mean(dim=(-2, -1), keepdim=True)
            sd = res.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
            res = (res - mu) / sd
        # With norm="none" the residual is left raw, which is poorly scaled for a CNN (values
        # are a few hundredths of the [0,1] pixel range). train.py selects per_image for this
        # input precisely so a badly scaled run is not mistaken for a failed hypothesis.
        return res

    if norm == "none":  # legacy path, bit-identical
        if mode == "dct":
            return torch.abs(_dct_2d(gray[:, 0]).unsqueeze(1))
        mag = torch.fft.fftshift(torch.fft.fft2(gray, dim=(-2, -1)), dim=(-2, -1))
        return torch.log(torch.abs(mag) + CONFIG["FREQ_EPS"])

    # .float() guards the transform against bf16 autocast during training.
    if mode == "dct":
        mag = torch.abs(_dct_2d(gray[:, 0].float()).unsqueeze(1))
    else:
        mag = torch.abs(
            torch.fft.fftshift(torch.fft.fft2(gray.float(), dim=(-2, -1)), dim=(-2, -1))
        )

    spec = torch.log(mag + CONFIG["FREQ_EPS"])  # log for BOTH modes
    if norm == "per_image":
        mu = spec.mean(dim=(-2, -1), keepdim=True)
        sd = spec.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        spec = (spec - mu) / sd
    return spec


def _efficientnet_freq_backbone():
    """
    EfficientNet-B0 taking a single-channel frequency map.
    The stem Conv2d(3→32) is replaced with Conv2d(1→32) initialised by
    averaging the three pretrained RGB channel weights [6.2].
    """
    m = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
    stem = m.conv_stem
    new_stem = nn.Conv2d(
        1,
        stem.out_channels,
        stem.kernel_size,
        stem.stride,
        stem.padding,
        bias=stem.bias is not None,
    )
    with torch.no_grad():
        new_stem.weight.copy_(stem.weight.mean(dim=1, keepdim=True))
        if stem.bias is not None:
            new_stem.bias.copy_(stem.bias)
    m.conv_stem = new_stem
    return m


def _spatial_normalize(x):
    """(x - mean) / std with ImageNet stats for the pretrained backbone."""
    device = x.device
    return (x - _IMAGENET_MEAN.to(device)) / _IMAGENET_STD.to(device)


def _masked_mean(x, valid):
    """Mean of (B, N, D) over valid positions; valid is (B, N, 1) float."""
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


# ── Fusion gate network ──────────────────────────────────────────


def fusion_gate_net() -> nn.Module:
    """
    Gate network: g = sigmoid(W₂·ReLU(W₁·[s; f])).
    Gate hidden size is 256 — model lands at 16,097,465 params
    (the 18,811,450 target from the FYP-1 report was ABANDONED
    2026-09-16; canonical = 16.1M T4-20).

    s, f ∈ ℝ¹²⁸⁰ → concat 2560 → hidden → 1280 → sigmoid.
    """
    gate_hidden = 256
    return nn.Sequential(
        nn.Linear(2560, gate_hidden),
        nn.ReLU(inplace=True),
        nn.Linear(gate_hidden, 1280),
        nn.Sigmoid(),
    )


# ── Full model ───────────────────────────────────────────────────


class DeepfakeDetector(nn.Module):
    """
    Stage 2–4 of the pipeline (see FYP1 report §6.2–6.4):

        spatial stream : EfficientNet-B0 (RGB, ImageNet stats)
        frequency stream: EfficientNet-B0 (log-magnitude FFT map, 1ch stem)
        gated fusion   : F = g⊙s + (1-g)⊙f
        projection     : 1280 → 512, LayerNorm, GELU
        temporal       : + learnable positional embeddings → 2-layer
                         8-head Transformer → mean pool
        head           : 512 → 256 → 1 (sigmoid applied by caller)

    Baseline mode (BASELINE_MODE=True) uses only the spatial stream and
    mean-pools the projections (no Transformer, no frequency branch).

    Args/frames: expects video frames (B, N, 3, 224, 224) — or a single
    video (N, 3, 224, 224), which is treated as batch 1.
    """

    def __init__(self):
        super().__init__()
        d_model = TRAIN_CONFIG["D_MODEL"]
        nhead = TRAIN_CONFIG["NHEAD"]
        d_ff = TRAIN_CONFIG["D_FF"]
        n_layers = TRAIN_CONFIG["N_LAYERS"]
        dropout = 0.1

        self.spatial = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.freq = _efficientnet_freq_backbone()

        self.gate = fusion_gate_net()

        self.project = nn.Sequential(
            nn.Linear(1280, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.pos_emb = nn.Parameter(torch.zeros(1, CONFIG["SAMPLE_COUNT"], d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(TRAIN_CONFIG["DROPOUT"]),
            nn.Linear(256, 1),
        )

    def _split_frames(self, frames):
        if frames.dim() == 4:  # (N, 3, 224, 224)
            frames = frames.unsqueeze(0)
        b, n, c, h, w = frames.shape
        return frames.view(b * n, c, h, w), (b, n)  # (B*N, 3, 224, 224), (b, n)

    def _spatial_features(self, flat):
        return self.spatial(_spatial_normalize(flat))  # (B*N, 1280)

    def _frequency_features(self, flat):
        return self.freq(frequency_transform(flat))  # (B*N, 1280)

    def forward(
        self,
        frames: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        freq_frames: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        pad_mask: optional (B, N) or (N,) bool, True = padded frame (cache v2
        zero padding). Padded frames are ignored by the Transformer and by
        pooling, so their content cannot change the logit. With no mask, or a
        mask with no padded frames, the computation is exactly the pre-mask one
        (legacy checkpoints reproduce bit for bit).

        freq_frames: optional separate view of the same clip for the frequency branch (R4,
        PLAN_BINDER.md Tab 5). It defaults to `frames`, so every caller that passes only
        (frames, pad_mask) — which is all nine evaluation and inference scripts — is unaffected
        and legacy checkpoints still reproduce bit for bit. Only the training loop passes it,
        and only when FREQ_AUGMENT is not "full".
        """
        flat, (b, n) = self._split_frames(frames)
        flat_freq = flat if freq_frames is None else self._split_frames(freq_frames)[0]
        if pad_mask is not None:
            pad_mask = pad_mask.to(flat.device).view(b, n).bool()
            if not pad_mask.any():
                pad_mask = None
        valid = None if pad_mask is None else (~pad_mask).unsqueeze(-1).to(flat.dtype)

        s = self._spatial_features(flat)  # (B*N, 1280)

        if CONFIG["BASELINE_MODE"]:
            proj = self.project(s).view(b, n, -1)  # (B, N, 512)
            pooled = proj.mean(dim=1) if valid is None else _masked_mean(proj, valid)
            logit = self.head(pooled)
            return logit, None

        # Ablation switches (FIX_PLAN P3.2). Both default to True, which is the
        # full model and the exact pre-ablation computation. Disabled branches
        # are skipped in the forward pass, not removed from the module, so a
        # checkpoint's key set never changes and every script that builds the
        # model before reading the checkpoint keeps working.
        use_freq = CONFIG["FREQ_ENABLED"]
        use_transformer = CONFIG["TRANSFORMER_ENABLED"]

        if use_freq:
            f = self._frequency_features(flat_freq)  # (B*N, 1280)
            g = self.gate(torch.cat([s, f], dim=-1))  # (B*N, 1280)
            fused = g * s + (1.0 - g) * f  # (B*N, 1280)
            gate_mean = g.view(b, n, 1280).mean(dim=-1)  # (B, N)
        else:
            fused, gate_mean = s, None  # A1: spatial only

        proj = self.project(fused).view(b, n, -1)  # (B, N, 512)

        if use_transformer:
            if TRAIN_CONFIG["POSITIONAL_EMB"]:
                proj = proj + self.pos_emb[:, :n, :]
            ctx = (
                self.transformer(proj)
                if valid is None
                else self.transformer(proj, src_key_padding_mask=pad_mask)
            )
        else:
            ctx = proj

        pooled = ctx.mean(dim=1) if valid is None else _masked_mean(ctx, valid)

        if valid is not None and gate_mean is not None:
            # Padded positions report the video's valid-frame mean, so any
            # downstream gate_mean.mean(dim=1) equals the valid-only mean.
            v = valid.squeeze(-1)
            per_video = (gate_mean * v).sum(1, keepdim=True) / v.sum(1, keepdim=True).clamp_min(1)
            gate_mean = torch.where(pad_mask, per_video.expand_as(gate_mean), gate_mean)

        logit = self.head(pooled)  # (B, 1)

        return logit, gate_mean


# ── Checkpoint architecture ──────────────────────────────────────

_ARCH_KEYS = ("BASELINE_MODE", "DCT_ENABLED", "FREQ_ENABLED", "TRANSFORMER_ENABLED")
_LEGACY_FREQ_NORM = "none"
#: What every checkpoint trained before R5 looked at. A checkpoint with no "freq_input" in its
#: arch block predates the switch, so it must be read as a spectrum model however this process
#: happens to be configured — otherwise evaluating an older run inside an R5 session would feed
#: its frequency backbone an input it has never seen and silently report a wrong number.
_LEGACY_FREQ_INPUT = "spectrum"


def apply_checkpoint_arch(ckpt: dict) -> str:
    """
    Set BASELINE_MODE / DCT_ENABLED from the architecture recorded inside a
    checkpoint, so the forward pass matches how the weights were trained.
    Without this a baseline checkpoint runs through its untrained frequency /
    gate / Transformer modules (audit-fyp2.md F2). Returns a short mode label.

    FYP-2 checkpoints carry an explicit "arch" block (FIX_PLAN P3.2). Older
    ones only have the whole CONFIG dump, which is read as a fallback so every
    FYP-1 checkpoint keeps loading exactly as before.
    """
    arch = ckpt.get("arch")
    if isinstance(arch, dict) and "baseline_mode" in arch:
        CONFIG["BASELINE_MODE"] = bool(arch["baseline_mode"])
        CONFIG["DCT_ENABLED"] = str(arch.get("freq_mode", "fft")).lower() == "dct"
        CONFIG["FREQ_ENABLED"] = bool(arch.get("freq_enabled", True))
        CONFIG["TRANSFORMER_ENABLED"] = bool(arch.get("transformer_enabled", True))
        CONFIG["FREQ_NORM"] = str(arch.get("freq_norm", _LEGACY_FREQ_NORM))
        CONFIG["FREQ_INPUT"] = str(arch.get("freq_input", _LEGACY_FREQ_INPUT))
        return arch_label()

    # Legacy checkpoints predate the ablation switches and the frequency
    # normalisation; they were all trained with both branches on and an
    # un-normalised frequency input, so restore that rather than inheriting
    # whatever the current process happens to have set.
    CONFIG["FREQ_ENABLED"] = True
    CONFIG["TRANSFORMER_ENABLED"] = True
    CONFIG["FREQ_NORM"] = _LEGACY_FREQ_NORM
    CONFIG["FREQ_INPUT"] = _LEGACY_FREQ_INPUT
    saved = ckpt.get("config", {}).get("CONFIG", {})
    for key in _ARCH_KEYS:
        if key in saved:
            CONFIG[key] = bool(saved[key])
    return arch_label()


def checkpoint_channel_order(ckpt: dict) -> str:
    """ "RGB" or "BGR" — the channel order these weights were trained on (PLAN task F1).

    This matters because the two data paths disagree. `src/face_cache.py` converts crops to RGB
    before caching, so every model trained on the v2 cache (M1, A1, A2, A4 …) learned RGB. But
    `src/preprocess.py` crops straight out of an OpenCV frame and never converts, so the live
    inference path produces **BGR**. Feeding a v2 model BGR silently swaps its red and blue
    channels: the model still returns a confident-looking probability, which is the worst kind of
    wrong.

    FYP-1 checkpoints have no `arch` block and were trained on the v1 BGR cache, so the fallback is
    BGR and T4-20 keeps behaving exactly as it always has.
    """
    arch = ckpt.get("arch")
    if isinstance(arch, dict) and arch.get("channel_order"):
        return str(arch["channel_order"]).upper()
    return "BGR"


def to_channel_order(frames: torch.Tensor, current: str, wanted: str) -> torch.Tensor:
    """Reorder a (..., 3, H, W) tensor's colour channels. A no-op when they already match."""
    if current.upper() == wanted.upper():
        return frames
    return frames.flip(-3)  # RGB <-> BGR is a reversal of the channel axis


def arch_label() -> str:
    """Human-readable name of the architecture selected by the current ``CONFIG``."""
    if CONFIG["BASELINE_MODE"]:
        return "BASELINE (spatial-only, no Transformer)"
    if CONFIG["FREQ_ENABLED"]:
        # The residual input makes freq_mode meaningless, so naming FFT/DCT there would be a
        # label that contradicts the forward pass.
        if CONFIG.get("FREQ_INPUT", "spectrum") == "residual":
            freq = "freq=RESIDUAL"
        else:
            freq = f"freq={'DCT' if CONFIG['DCT_ENABLED'] else 'FFT'}"
        if CONFIG["FREQ_NORM"] != _LEGACY_FREQ_NORM:
            freq += f"({CONFIG['FREQ_NORM']})"
    else:
        freq = "freq=OFF"
    if CONFIG["FREQ_ENABLED"] and CONFIG["TRANSFORMER_ENABLED"]:
        return f"FULL | {freq}"
    tf = "transformer=ON" if CONFIG["TRANSFORMER_ENABLED"] else "transformer=OFF"
    return f"ABLATED | {freq} | {tf}"


# ── Parameter audit ──────────────────────────────────────────────


def count_params(model: nn.Module) -> int:
    """Total number of parameters in ``model``."""
    return sum(p.numel() for p in model.parameters())


class ConstGate(nn.Module):
    """Gate that emits a constant — ablation: force spatial or freq fusion."""

    def __init__(self, value):
        super().__init__()
        self.register_buffer("v", torch.tensor(value, dtype=torch.float32))

    def forward(self, x):
        """Return the constant gate value, shaped like one half of ``x``."""
        return torch.full_like(x[..., : x.shape[-1] // 2], self.v.item())


def force_gate(model: DeepfakeDetector, value: float) -> DeepfakeDetector:
    """Replace the learned fusion gate with a constant (for ablations)."""
    model.gate = ConstGate(value)
    return model


if __name__ == "__main__":
    m = DeepfakeDetector()
    n = count_params(m)
    print(f"DeepfakeDetector params: {n:,}")
    print("Canonical (T4-20)      : 16,097,465")
    print(f"delta                  : {n - 16_097_465:+,}")
