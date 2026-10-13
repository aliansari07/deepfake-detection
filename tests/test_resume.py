"""Resuming must not let a worse epoch overwrite the best checkpoint (audit-fyp2 §4.5)."""

import os

import torch
import torch.nn as nn

import src.dataset
import src.train as T
from src.config import override_config


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(1, 1)

    def forward(self, x):
        return self.w(x), None


HISTORY = {
    "train_loss": [0.3, 0.2, 0.2],
    "val_loss": [0.3, 0.2, 0.2],
    "train_acc": [0.8, 0.9, 0.9],
    "val_acc": [0.8, 0.9, 0.9],
    "val_auc": [0.90, 0.99, 0.95],
    "val_f1": [0.8, 0.9, 0.9],
    "lr": [1e-4] * 3,
    "gate": [0.5, 0.55, 0.6],
    "time": [1, 1, 1],
}


def _resume_once(tmp_path, monkeypatch, next_auc):
    monkeypatch.setattr(T, "DeepfakeDetector", _Tiny)
    monkeypatch.setattr(src.dataset, "build_dataloaders", lambda: ([], []))
    monkeypatch.setattr(T, "_train_epoch", lambda *a, **k: {"loss": 0.1, "acc": 0.9, "gate": 0.6})
    # Mirror what _validate() really returns: the checkpoint metrics block is
    # built from this dict and now requires threshold + balanced_acc.
    monkeypatch.setattr(
        T,
        "_validate",
        lambda *a, **k: {
            "loss": 0.1,
            "acc": 0.9,
            "f1": 0.9,
            "auc": next_auc,
            "gate": 0.6,
            "threshold": 0.62,
            "balanced_acc": 0.88,
        },
    )
    monkeypatch.setattr(T, "_plot_training", lambda *a, **k: None)
    monkeypatch.setattr(T, "_plot_gate_drift", lambda *a, **k: None)

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    with override_config(CHECKPOINT_DIR=str(ckpt_dir), EPOCHS=4):
        m = _Tiny()
        opt = torch.optim.AdamW(m.parameters())
        # latest.pt from epoch 3 stores THAT epoch's AUC (0.95), exactly as train.py does
        T._save_ckpt(m, opt, 3, HISTORY, {"auc": 0.95, "epoch": 3}, str(ckpt_dir / "R-4.latest.pt"))
        best = ckpt_dir / "R-4.best.pt"
        T._save_ckpt(m, opt, 2, HISTORY, {"auc": 0.99, "epoch": 2}, str(best))
        before = os.path.getmtime(best)
        _, history = T.train_model(resume=True, run_id="R-4")
        best_metrics = torch.load(best, weights_only=True)["metrics"]
        return os.path.getmtime(best) == before, history, best_metrics


def test_best_from_history():
    assert T._best_from_history({"val_auc": [0.90, 0.99, 0.95]}) == (0.99, 2, 1)
    assert T._best_from_history({"val_auc": []}) == (0.0, -1, 0)


def test_worse_epoch_after_resume_keeps_best(tmp_path, monkeypatch):
    unchanged, history, best = _resume_once(tmp_path, monkeypatch, next_auc=0.97)
    assert unchanged
    assert best["auc"] == 0.99 and best["epoch"] == 2
    assert history["val_auc"] == [0.90, 0.99, 0.95, 0.97]


def test_better_epoch_after_resume_replaces_best(tmp_path, monkeypatch):
    unchanged, _, best = _resume_once(tmp_path, monkeypatch, next_auc=0.995)
    assert not unchanged
    assert best["auc"] == 0.995 and best["epoch"] == 4
