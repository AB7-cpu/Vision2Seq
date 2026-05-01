"""
Utility functions:

1. Tracking metrics (loss, accuracy, etc.)  
2. Saving and Loading checkpoints
3. Plotting training curves
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any

import torch
import matplotlib.pyplot as plt



class AverageMeter:
    """Track a running average metric"""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


def get_lr(optimiser: torch.optim.Optimizer) -> float:
    return optimiser.param_groups[0]["lr"]


#     Checkpoint Saving and Loading:

def save_checkpoint(
    state    : dict[str, Any],    # Dictionary with completed epoch, encoder, decoder, optimiser, bleu4
    save_dir : Path | str,        # Save checkpoint directory
    filename : str,               # Checkpoint file name
    is_best  : bool = False,      # If the current epoch checkpoint is best then copy as best_model.pth
) -> Path:
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / filename
    torch.save(state, path)
    print(f"[Checkpoint] Saved → {path}")

    if is_best:
        best_path = save_dir / "best_model.pth"
        torch.save(state, best_path)
        print(f"[Checkpoint] ★ New best model → {best_path}")

    return path


def load_checkpoint(path: Path | str, device: torch.device) -> dict[str, Any]:
    path = Path(path)
    assert path.exists(), f"Checkpoint not found: {path}"
    state = torch.load(path, map_location=device)
    print(f"[Checkpoint] Loaded from {path}  (epoch {state.get('epoch', '?')})")
    return state


#     Visualisation:

def plot_training_curves(
    history  : dict[str, list[float]],
    save_path: Path | str | None = None,
) -> None:
    
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#4C72B0")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="#DD8452")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    # BLEU-4
    ax2.plot(epochs, history["bleu4"], label="BLEU-4", color="#55A868", marker="o")
    ax2.set_title("Validation BLEU-4 over Epochs")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("BLEU-4 Score")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[Plot] Saved training curves → {save_path}")
    else:
        plt.show()

    plt.close(fig)


def save_history(history: dict, path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[History] Saved → {path}")


def format_time(seconds: float) -> str:
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"
