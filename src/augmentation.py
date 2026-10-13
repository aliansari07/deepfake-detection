"""CompressionAwareAugmentation — simulates social-media-grade
distortions so the model learns to tolerate real-world pipelines.

Transformations (each applied with p = 0.5, item 6.5):
  - JPEG compression (quality 30–95)
  - Gaussian blur (kernel 3x3/5x5/7x7)
  - Random resize (scale 50–90%, then back to 224)
  - Gaussian noise (sigma 1–5% of pixel range)

Two sampling granularities:
  aug(frame) / augment_batch(t)  → parameters drawn per FRAME (legacy).
  aug.augment_video(frames)      → one parameter set per VIDEO, shared by
                                   every frame except the noise, which stays
                                   per frame (FIX_PLAN P3.3 / audit §5.3).
A real compression pipeline re-encodes a whole clip at one quality, so the
per-video form is what training uses; the per-frame form is kept for the
inference-time TTA path and the demo.

Import:
  from src.augmentation import CompressionAwareAugmentation
"""

import random

import cv2
import numpy as np
import torch

from src.config import CONFIG

# Transform names in a fixed canonical order. _sample_params() always draws
# every parameter, whether or not the transform ends up being applied, so the
# number of RNG draws per video is constant and a seed reproduces exactly.
TRANSFORMS = ("jpeg", "blur", "resize", "noise")

#: Augmentation strength profiles (PLAN_BINDER.md Tab 5, R4).
#:
#: "full" is the pre-R4 policy and must stay byte-for-byte what it was, so seeded streams and the
#: M1/A1/A2/A4 runs remain reproducible. "gentle" exists because R2 measured that this pipeline is
#: what collapses the fusion gate: `resize` to half scale and `blur` to sigma 2.0 are low-pass
#: filters, and they delete exactly the high-frequency content the frequency branch is meant to
#: detect (audit-fyp2.md section 15.2). "gentle" therefore drops both and keeps only the
#: compression-like corruptions the branch must genuinely tolerate.
PROFILES = {
    "full": {
        "transforms": TRANSFORMS,
        "p": 0.5,
        "jpeg_q": (30, 95),
        "blur_sigma": (0.5, 2.0),
        "scale": (0.5, 0.9),
        "noise_sigma": (0.01, 0.05),
    },
    "gentle": {
        "transforms": ("jpeg", "noise"),
        "p": 0.3,
        "jpeg_q": (70, 95),
        "blur_sigma": (0.5, 1.0),
        "scale": (0.8, 1.0),
        "noise_sigma": (0.01, 0.03),
    },
    # "none" leaves the frequency branch's view of the clip completely clean while the spatial
    # branch keeps the full profile. It is the sharpest test of cause C2: if the branch cannot
    # earn its place even on undamaged input, augmentation was never what held it back.
    "none": {
        "transforms": (),
        "p": 0.0,
        "jpeg_q": (70, 95),
        "blur_sigma": (0.5, 1.0),
        "scale": (0.8, 1.0),
        "noise_sigma": (0.01, 0.03),
    },
}


def _tensor_to_bgr(t):
    """(3,H,W) float [0,1] tensor → BGR uint8 frame."""
    arr = np.transpose(t.detach().cpu().numpy(), (1, 2, 0))  # CHW→HWC
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bgr_to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


class CompressionAwareAugmentation:
    """
    Applies stochastic compression-aware distortions to face tensors.

    Call styles:
      aug(frame)                  → augmented (3,H,W) tensor      (per-frame)
      aug.augment_batch(tensor)   → (aug_tensor, frame_logs)      (per-frame)
      aug.augment_video(tensor)   → (k,3,H,W) tensor              (per-video)
      aug.reseed(seed)            → restart the parameter stream
    """

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def reseed(self, seed):
        """Restart the parameter stream from `seed`.

        Called from the DataLoader's worker_init_fn so that each worker draws a
        different, reproducible sequence. Without it every worker inherits the
        same seed from TRAIN_CONFIG and augments identically (audit §5.2).
        """
        self.rng = random.Random(seed)
        return self

    # ── Parameter sampling ────────────────────────────────────────
    def _sample_params(self, profile="full"):
        """One parameter set, drawn in a fixed order so seeds reproduce.

        Every parameter is drawn whatever the profile, and the per-transform coin is flipped for
        all four names even when a profile forbids one, so the structure of the draw does not
        depend on which transforms a profile allows. A seed therefore replays a run exactly
        (FIX_PLAN P3.3).

        Note that two different profiles do NOT consume identical entropy — `randint(30, 95)` and
        `randint(70, 95)` draw different numbers of bits — so streams are comparable within a
        profile, not across profiles. That is fine: a profile is fixed for the whole of a run.
        """
        order = list(TRANSFORMS)
        self.rng.shuffle(order)

        if profile == "full":
            # Unchanged from the pre-R4 code, deliberately: the M1/A1/A2/A4 runs and the seeded
            # reproduction tests depend on these exact draws.
            return {
                "profile": "full",
                "order": order,
                "apply": {name: self.rng.random() < 0.5 for name in TRANSFORMS},
                "jpeg_q": self.rng.randint(30, 95),
                "blur_k": self.rng.choice([3, 5, 7]),
                "blur_sigma": self.rng.uniform(0.5, 2.0),
                "scale": self.rng.uniform(0.5, 0.9),
                "noise_sigma": self.rng.uniform(0.01, 0.05),
            }

        spec = PROFILES[profile]
        allowed, p = spec["transforms"], spec["p"]
        return {
            "profile": profile,
            "order": order,
            # The draw comes first so it happens for every name; the membership test only decides
            # whether the result is used. Reversing these would short-circuit and desynchronise
            # the stream.
            "apply": {name: (self.rng.random() < p) and (name in allowed) for name in TRANSFORMS},
            "jpeg_q": self.rng.randint(*spec["jpeg_q"]),
            "blur_k": self.rng.choice([3, 5, 7]),
            "blur_sigma": self.rng.uniform(*spec["blur_sigma"]),
            "scale": self.rng.uniform(*spec["scale"]),
            "noise_sigma": self.rng.uniform(*spec["noise_sigma"]),
        }

    # ── Individual distortions (parameterised) ────────────────────
    @staticmethod
    def _jpeg_p(frame, quality):
        bgr = _tensor_to_bgr(frame)
        ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            return frame
        return _bgr_to_tensor(cv2.imdecode(enc, cv2.IMREAD_COLOR))

    @staticmethod
    def _blur_p(frame, k, sigma):
        bgr = _tensor_to_bgr(frame)
        return _bgr_to_tensor(cv2.GaussianBlur(bgr, (int(k), int(k)), sigma))

    @staticmethod
    def _resize_p(frame, scale):
        bgr = _tensor_to_bgr(frame)
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))))
        back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return _bgr_to_tensor(back)

    @staticmethod
    def _noise_p(frame, sigma):
        # Drawn from the torch RNG, which the DataLoader seeds per worker, so
        # the sensor-noise term stays independent per frame.
        return (frame + torch.randn_like(frame) * sigma).clamp(0, 1)

    def _apply(self, frame, p):
        """Apply one sampled parameter set to a single (3,H,W) frame."""
        out = frame
        for name in p["order"]:
            if not p["apply"][name]:
                continue
            if name == "jpeg":
                out = self._jpeg_p(out, p["jpeg_q"])
            elif name == "blur":
                out = self._blur_p(out, p["blur_k"], p["blur_sigma"])
            elif name == "resize":
                out = self._resize_p(out, p["scale"])
            elif name == "noise":
                out = self._noise_p(out, p["noise_sigma"])
        return out

    # ── Per-frame API (legacy; inference TTA and the demo) ─────────
    def _jpeg(self, frame):
        return self._jpeg_p(frame, self.rng.randint(30, 95))

    def _blur(self, frame):
        return self._blur_p(frame, self.rng.choice([3, 5, 7]), self.rng.uniform(0.5, 2.0))

    def _resize(self, frame):
        return self._resize_p(frame, self.rng.uniform(0.5, 0.9))

    def _noise(self, frame):
        return self._noise_p(frame, self.rng.uniform(0.01, 0.05))

    def __call__(self, frame):
        """Augment a single (3,H,W) float [0,1] tensor, parameters per frame."""
        out, _ = self.augment_batch_frame(frame)
        return out

    def augment_batch_frame(self, frame):
        """Augment one frame (3,H,W); returns (tensor, [names])."""
        transforms = [self._jpeg, self._blur, self._resize, self._noise]
        self.rng.shuffle(transforms)
        applied, out = [], frame
        for t in transforms:
            if self.rng.random() < 0.5:
                out = t(out)
                applied.append(t.__name__.replace("_", ""))
        return out, applied

    def augment_batch(self, tensor):
        """
        Augment a batch of frames (N,3,224,224) float [0,1], parameters drawn
        per frame.

        Returns:
            aug_tensor : (N,3,224,224) augmented frames.
            aug_logs   : list of applied-transform names per frame.
        """
        frames, logs = [], []
        for i in range(tensor.shape[0]):
            aug, applied = self.augment_batch_frame(tensor[i])
            frames.append(aug)
            logs.append(applied)
        return torch.stack(frames), logs

    # ── Per-video API (training) ──────────────────────────────────
    def augment_video(self, frames, return_params=False, profile="full"):
        """
        Augment every frame of one clip with a single sampled parameter set.

        Args:
            frames        : (k,3,H,W) float [0,1] tensor of the clip's real frames.
            return_params : also return the parameter dict (used by the tests).
            profile       : a key of PROFILES; "full" is the pre-R4 policy.

        Returns:
            (k,3,H,W) tensor, or (tensor, params) when return_params is True.
        """
        p = self._sample_params(profile)
        out = frames if frames.shape[0] == 0 else self._apply_params(frames, p)
        return (out, p) if return_params else out

    def _apply_params(self, frames, p):
        return torch.stack([self._apply(frames[i], p) for i in range(frames.shape[0])])

    def augment_video_pair(self, frames, freq_profile, return_params=False):
        """Two augmented views of one clip: the spatial branch's and the frequency branch's (R4).

        The spatial view always uses the "full" profile, so the compression robustness the S3
        claim rests on is unchanged. The frequency view uses `freq_profile`, normally "gentle".

        Both parameter sets come from the same per-video RNG in a fixed order, so the pair is
        reproducible under the existing per-worker reseeding and a seed still replays a run
        exactly.
        """
        p_spatial = self._sample_params("full")
        p_freq = self._sample_params(freq_profile)
        if frames.shape[0] == 0:
            out_s = out_f = frames
        else:
            out_s = self._apply_params(frames, p_spatial)
            out_f = self._apply_params(frames, p_freq)
        if return_params:
            return out_s, out_f, p_spatial, p_freq
        return out_s, out_f


if __name__ == "__main__":
    from src.logger import log

    demo = torch.rand(CONFIG["SAMPLE_COUNT"], 3, 224, 224)
    aug = CompressionAwareAugmentation(seed=7)
    out, params = aug.augment_video(demo, return_params=True)
    applied = [n for n in params["order"] if params["apply"][n]]
    log(
        "AUGMENT",
        f"demo: {tuple(out.shape)} — applied: "
        f"{', '.join(applied) or '(none)'} (jpeg q={params['jpeg_q']})",
    )
