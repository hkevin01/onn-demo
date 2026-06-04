"""
# ID: TRAIN
# Purpose: Training loop for both ONN and baseline models.
#          Supports operator parameter snapshotting for trajectory visualization.
#
# Requirement: Train loop must record per-epoch metrics and optionally checkpoint.
# Inputs: model, train_loader, config dict
# Outputs: dict with 'train_losses', 'train_accs', 'val_losses', 'val_accs'
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional

from .utils import accuracy, format_metrics, get_device, save_checkpoint
from .model.onn_model import ONNModel


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    snapshot: bool = False,
) -> Dict[str, float]:
    """
    # ID: TRAIN_EPOCH
    # Purpose: Run one full training epoch.
    # Inputs:
    #   model     - nn.Module
    #   loader    - DataLoader
    #   optimizer - torch optimizer
    #   criterion - loss function
    #   device    - torch.device
    #   snapshot  - bool, if True and model is ONNModel, snapshot operator params
    # Outputs: dict with 'loss' and 'acc' averages
    # Side effects: updates model weights in-place
    """
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(logits.detach(), y)
        n_batches += 1

    if snapshot and isinstance(model, ONNModel):
        model.snapshot_params()

    return {"loss": total_loss / n_batches, "acc": total_acc / n_batches}


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    # ID: EVAL_EPOCH
    # Purpose: Evaluate model on a DataLoader without gradient computation.
    # Inputs: model, loader, criterion, device
    # Outputs: dict with 'loss' and 'acc'
    """
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item()
            total_acc += accuracy(logits, y)
            n_batches += 1

    return {"loss": total_loss / n_batches, "acc": total_acc / n_batches}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    snapshot_every: int = 5,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    # ID: TRAIN_MODEL
    # Purpose: Full training routine with early stopping, LR scheduling, and
    #          optional operator parameter snapshots.
    #
    # Inputs:
    #   model         - nn.Module (ONNModel or baseline)
    #   train_loader  - DataLoader
    #   val_loader    - DataLoader
    #   epochs        - int
    #   lr            - float, initial learning rate
    #   weight_decay  - float, L2 regularization
    #   snapshot_every - int, snapshot operator params every N epochs (ONN only)
    #   save_path     - str or None, path to save best checkpoint
    #   verbose       - bool, print progress
    # Outputs: history dict with train/val loss and accuracy lists
    # Side effects: modifies model weights, optionally writes checkpoint
    """
    device = get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_losses": [], "train_accs": [],
        "val_losses": [], "val_accs": [],
    }
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        do_snapshot = (epoch % snapshot_every == 0)
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, snapshot=do_snapshot)
        val_metrics = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_losses"].append(train_metrics["loss"])
        history["train_accs"].append(train_metrics["acc"])
        history["val_losses"].append(val_metrics["loss"])
        history["val_accs"].append(val_metrics["acc"])

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            if save_path:
                save_checkpoint(model, save_path, {"epoch": epoch, "best_val_acc": best_val_acc})

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train: {format_metrics(train_metrics)}  "
                f"val: {format_metrics(val_metrics)}"
            )

    history["best_val_acc"] = best_val_acc
    return history
