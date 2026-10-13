"""Logging helpers used across all modules.
Import: from src.logger import log, debug, section
"""

from src.config import CONFIG

_MUTED = False


def mute(flag=True):
    """Globally silence log/detail/section output (batch scripts)."""
    global _MUTED
    _MUTED = bool(flag)


def log(stage, msg):
    """Stage-labelled line. Always visible."""
    if _MUTED:
        return
    print(f"  [{stage:^14s}]  {msg}")


def detail(msg):
    """Indented sub-detail. Always visible."""
    if _MUTED:
        return
    print(f"  {' ':14s}   > {msg}")


def debug(msg):
    """Only prints when LOG_MODE = 'debug'."""
    if _MUTED:
        return
    if CONFIG["LOG_MODE"] == "debug":
        print(f"  {' ':14s}   > {msg}")


def tensor_stats(t, name="tensor"):
    """Return a string summarising a tensor's shape and values."""
    return (
        f"{name}: shape={list(t.shape)}, "
        f"min={t.min().item():.4f}, max={t.max().item():.4f}, "
        f"mean={t.mean().item():.4f}, std={t.std().item():.4f}"
    )


def log_gpu():
    """Print GPU memory. Only active in debug mode."""
    import torch

    if CONFIG["LOG_MODE"] == "debug" and torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        resrv = torch.cuda.memory_reserved() / 1024**2
        debug(f"GPU memory: {alloc:.1f}MB allocated / {resrv:.1f}MB reserved")


def section(title):
    """Print a visual section divider."""
    print(f"\n  {'─' * 60}")
    print(f"  {title}")
    print(f"  {'─' * 60}")
