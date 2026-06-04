"""
# ID: EVALUATE
# Purpose: Evaluation utilities - compute metrics, confusion matrix, per-class accuracy.
#
# Requirement: All metrics must be computed on held-out test data only.
# Inputs: trained model, test DataLoader or tensors
# Outputs: dict of metrics
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple

from .utils import get_device, accuracy


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
) -> Dict[str, float]:
    """
    # ID: EVALUATE_MODEL
    # Purpose: Compute loss, accuracy, and per-class accuracy on test set.
    # Inputs:
    #   model       - nn.Module (eval mode will be set internally)
    #   test_loader - DataLoader
    # Outputs: dict with 'loss', 'accuracy', 'per_class_acc' (list)
    # Preconditions: model and data must be on same device (handled internally)
    """
    device = get_device()
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds, all_labels, all_logits = [], [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item()
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_logits.append(logits.cpu())
            n_batches += 1

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    overall_acc = (all_preds == all_labels).float().mean().item()
    n_classes = int(all_labels.max().item()) + 1

    per_class = []
    for c in range(n_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class.append((all_preds[mask] == all_labels[mask]).float().mean().item())
        else:
            per_class.append(float("nan"))

    return {
        "loss": total_loss / n_batches,
        "accuracy": overall_acc,
        "per_class_acc": per_class,
        "n_correct": int((all_preds == all_labels).sum().item()),
        "n_total": len(all_labels),
    }


def compare_models(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
) -> Dict[str, Dict]:
    """
    # ID: COMPARE_MODELS
    # Purpose: Evaluate multiple models and return a combined results dict.
    # Inputs:
    #   models      - dict {model_name: nn.Module}
    #   test_loader - shared DataLoader
    # Outputs: dict {model_name: metrics_dict}
    """
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, test_loader)
        print(
            f"{name:<20} acc={results[name]['accuracy']:.4f}  "
            f"loss={results[name]['loss']:.4f}  "
            f"params={sum(p.numel() for p in model.parameters()):,}"
        )
    return results


def predict_grid(
    model: nn.Module,
    x_range: Tuple[float, float] = (-3.5, 3.5),
    y_range: Tuple[float, float] = (-3.5, 3.5),
    resolution: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    # ID: PREDICT_GRID
    # Purpose: Generate decision boundary grid predictions for 2D models.
    # Inputs:
    #   model      - nn.Module accepting (batch, 2) input
    #   x_range    - (min, max) for x axis
    #   y_range    - (min, max) for y axis
    #   resolution - int, grid resolution
    # Outputs: (xx, yy, Z) where Z is predicted class probability or class index
    # Preconditions: model must accept 2D inputs
    """
    device = get_device()
    model = model.to(device)
    model.eval()

    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    grid_t = torch.from_numpy(grid).to(device)

    with torch.no_grad():
        logits = model(grid_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    Z = probs.reshape(xx.shape)
    return xx, yy, Z
