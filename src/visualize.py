"""
# ID: VISUALIZE
# Purpose: Visualization routines for decision boundaries, training curves,
#          operator parameter trajectories, and activation distributions.
#
# Requirement: All plots must save to outputs/ directory without blocking (use savefig).
# Inputs: trained models, histories, data tensors
# Outputs: PNG files saved to outputs/
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend - safe for headless environments
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Dict, List, Optional, Tuple

from .evaluate import predict_grid

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _savefig(name: str):
    """
    # ID: SAVEFIG
    # Purpose: Save current figure to outputs/ and close it.
    # Side effects: writes PNG file, closes matplotlib figure
    """
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_decision_boundaries(
    models: Dict[str, nn.Module],
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    title: str = "Decision Boundaries",
    filename: str = "decision_boundaries.png",
    resolution: int = 200,
):
    """
    # ID: PLOT_DECISION_BOUNDARIES
    # Purpose: Side-by-side decision boundary plots for multiple models.
    # Inputs:
    #   models    - dict {name: nn.Module}
    #   X_test    - (n, 2) float tensor
    #   y_test    - (n,) long tensor
    #   title     - str, overall figure title
    #   filename  - str, output filename
    #   resolution - int, grid resolution
    # Preconditions: models must accept 2D input
    # Side effects: saves PNG
    """
    X_np = X_test.numpy()
    y_np = y_test.numpy()
    x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14, fontweight="bold")

    cmap_bg = plt.cm.RdBu
    colors = ["#e74c3c", "#2980b9"]

    for ax, (name, model) in zip(axes, models.items()):
        xx, yy, Z = predict_grid(model, (x_min, x_max), (y_min, y_max), resolution)
        ax.contourf(xx, yy, Z, levels=50, cmap=cmap_bg, alpha=0.7, vmin=0, vmax=1)
        ax.contour(xx, yy, Z, levels=[0.5], colors="white", linewidths=1.5, linestyles="--")
        for cls, color in enumerate(colors):
            mask = y_np == cls
            ax.scatter(X_np[mask, 0], X_np[mask, 1], c=color, s=20, alpha=0.7,
                       edgecolors="white", linewidths=0.3, label=f"Class {cls}")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend(fontsize=8)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    _savefig(filename)


def plot_training_curves(
    histories: Dict[str, Dict],
    filename: str = "training_curves.png",
):
    """
    # ID: PLOT_TRAINING_CURVES
    # Purpose: Plot train/val loss and accuracy curves for multiple models.
    # Inputs:
    #   histories - dict {model_name: history_dict}
    # Side effects: saves PNG
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = sns.color_palette("tab10", len(histories))

    for (name, history), color in zip(histories.items(), colors):
        epochs = range(1, len(history["train_losses"]) + 1)
        axes[0].plot(epochs, history["train_losses"], color=color, linestyle="--", alpha=0.6, label=f"{name} train")
        axes[0].plot(epochs, history["val_losses"], color=color, linestyle="-", label=f"{name} val")
        axes[1].plot(epochs, history["train_accs"], color=color, linestyle="--", alpha=0.6, label=f"{name} train")
        axes[1].plot(epochs, history["val_accs"], color=color, linestyle="-", label=f"{name} val")

    axes[0].set_title("Loss over Epochs", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Accuracy over Epochs", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _savefig(filename)


def plot_operator_trajectories(
    onn_model,
    filename: str = "operator_trajectories.png",
):
    """
    # ID: PLOT_OPERATOR_TRAJECTORIES
    # Purpose: Visualize how operator parameters evolved during training
    #          by plotting their L2 norms across snapshots.
    # Inputs:
    #   onn_model - ONNModel with non-empty param_history
    # Preconditions: onn_model.param_history must have at least 2 entries
    # Side effects: saves PNG
    """
    if not onn_model.param_history:
        print("  No parameter history recorded - skipping trajectory plot.")
        return

    history = onn_model.param_history
    layer_keys = list(history[0].keys())
    n_layers = len(layer_keys)

    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    fig.suptitle(f"Operator Parameter Trajectories ({onn_model.operator_name})", fontsize=13)

    for ax, layer_key in zip(axes, layer_keys):
        param_names = list(history[0][layer_key].keys())
        snapshots = list(range(1, len(history) + 1))
        colors = sns.color_palette("husl", len(param_names))

        for param_name, color in zip(param_names, colors):
            norms = [history[t][layer_key][param_name].norm().item() for t in range(len(history))]
            ax.plot(snapshots, norms, marker="o", markersize=3, color=color, label=param_name)

        ax.set_title(layer_key)
        ax.set_xlabel("Snapshot (every N epochs)")
        ax.set_ylabel("L2 Norm")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    _savefig(filename)


def plot_operator_response(
    onn_model,
    layer_idx: int = 0,
    filename: str = "operator_response.png",
):
    """
    # ID: PLOT_OPERATOR_RESPONSE
    # Purpose: Plot the learned operator's input-output response curve
    #          for the first ONN layer over a range of input values.
    # Inputs:
    #   onn_model - ONNModel
    #   layer_idx - int, which ONN layer to visualize
    # Preconditions: model in_features >= 1
    # Side effects: saves PNG
    """
    device = next(onn_model.parameters()).device
    layer = onn_model.onn_layers[layer_idx]
    operator = layer.operator
    operator.eval()

    # Determine operator's in_features from the model
    in_f = onn_model.in_features
    x_scalar = torch.linspace(-3, 3, 300)  # (300,)
    # Build full input: vary dim 0, keep other dims at zero
    x_vals = torch.zeros(300, in_f)
    x_vals[:, 0] = x_scalar
    x_vals = x_vals.to(device)

    with torch.no_grad():
        y_vals = operator(x_vals).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    for j in range(min(y_vals.shape[1], 8)):
        ax.plot(x_scalar.numpy(), y_vals[:, j], alpha=0.7, label=f"neuron {j}")

    ax.set_title(f"Operator Response - Layer {layer_idx} ({onn_model.operator_name})", fontsize=12)
    ax.set_xlabel("Input value (feature dim 0, others=0)")
    ax.set_ylabel("Operator output")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    _savefig(filename)


def plot_comparison_bar(
    results: Dict[str, Dict],
    filename: str = "comparison_bar.png",
):
    """
    # ID: PLOT_COMPARISON_BAR
    # Purpose: Bar chart comparing final test accuracy of all models.
    # Inputs:
    #   results - dict {model_name: metrics_dict}
    # Side effects: saves PNG
    """
    names = list(results.keys())
    accs = [results[n]["accuracy"] * 100 for n in names]

    colors = sns.color_palette("viridis", len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=1.2)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Model Comparison - Test Accuracy", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(filename)
