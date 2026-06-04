"""
# ID: MAIN
# Purpose: Entry point for the ONN demo. Trains ONN variants and MLP baseline on
#          a 2D toy dataset, compares performance, and produces all visualizations.
#
# Usage:
#   python -m src.main --dataset spiral --epochs 80 --operator polynomial
#   python -m src.main --dataset moons --epochs 60 --operator sinusoidal
#   python -m src.main --dataset circles --epochs 60 --operator gaussian
#
# Outputs: outputs/*.png, outputs/results_summary.txt
"""

import argparse
import os
import sys
import torch

from .data import load_dataset, get_full_tensors
from .model.onn_model import ONNModel
from .model.baseline_model import BaselineMLP
from .train import train_model
from .evaluate import compare_models
from .visualize import (
    plot_decision_boundaries,
    plot_training_curves,
    plot_operator_trajectories,
    plot_operator_response,
    plot_comparison_bar,
)
from .utils import set_seed, count_parameters, get_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ONN Demo")
    p.add_argument("--dataset", default="spiral", choices=["moons", "circles", "spiral"])
    p.add_argument("--n_samples", type=int, default=1200)
    p.add_argument("--noise", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden", type=int, nargs="+", default=[32, 32])
    p.add_argument("--operator", default="polynomial",
                   choices=["polynomial", "sinusoidal", "gaussian", "multiplicative"])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_checkpoints", action="store_true")
    return p


def main(args=None):
    parser = build_parser()
    args = parser.parse_args(args)

    set_seed(args.seed)
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  ONN Demo  |  dataset={args.dataset}  operator={args.operator}")
    print(f"  device={device}  epochs={args.epochs}  seed={args.seed}")
    print(f"{'='*60}\n")

    # --- Data ---
    print("[1/5] Loading data...")
    train_loader, test_loader, in_features, n_classes = load_dataset(
        name=args.dataset,
        n_samples=args.n_samples,
        noise=args.noise,
        batch_size=args.batch_size,
        random_state=args.seed,
    )
    X_train, X_test, y_train, y_test = get_full_tensors(
        name=args.dataset, n_samples=args.n_samples,
        noise=args.noise, random_state=args.seed,
    )
    print(f"  in_features={in_features}  n_classes={n_classes}  "
          f"train={len(train_loader.dataset)}  test={len(test_loader.dataset)}")

    # --- Models ---
    print("\n[2/5] Building models...")
    onn = ONNModel(
        in_features=in_features,
        hidden_dims=args.hidden,
        n_classes=n_classes,
        operator=args.operator,
        dropout=args.dropout,
    )
    mlp = BaselineMLP(
        in_features=in_features,
        hidden_dims=args.hidden,
        n_classes=n_classes,
        dropout=args.dropout,
    )
    print(f"  ONN params:      {count_parameters(onn):,}")
    print(f"  Baseline params: {count_parameters(mlp):,}")

    # --- Training ---
    print("\n[3/5] Training models...")

    onn_ckpt = "outputs/onn_best.pt" if args.save_checkpoints else None
    mlp_ckpt = "outputs/mlp_best.pt" if args.save_checkpoints else None

    print("  Training ONN...")
    onn_history = train_model(
        onn, train_loader, test_loader,
        epochs=args.epochs, lr=args.lr,
        snapshot_every=max(1, args.epochs // 10),
        save_path=onn_ckpt,
    )
    print("  Training Baseline MLP...")
    mlp_history = train_model(
        mlp, train_loader, test_loader,
        epochs=args.epochs, lr=args.lr,
        save_path=mlp_ckpt,
    )

    # --- Evaluation ---
    print("\n[4/5] Evaluating models...")
    models_dict = {f"ONN ({args.operator})": onn, "Baseline MLP": mlp}
    results = compare_models(models_dict, test_loader)

    os.makedirs("outputs", exist_ok=True)
    summary_path = "outputs/results_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Dataset: {args.dataset}  Operator: {args.operator}\n")
        f.write(f"Epochs: {args.epochs}  LR: {args.lr}\n\n")
        for name, res in results.items():
            f.write(f"{name}\n")
            f.write(f"  accuracy:  {res['accuracy']:.4f}\n")
            f.write(f"  loss:      {res['loss']:.4f}\n")
            f.write(f"  per_class: {[f'{a:.3f}' for a in res['per_class_acc']]}\n\n")
    print(f"  Summary saved: {summary_path}")

    # --- Visualizations ---
    print("\n[5/5] Generating visualizations...")

    plot_decision_boundaries(
        models=models_dict,
        X_test=X_test,
        y_test=y_test,
        title=f"Decision Boundaries - {args.dataset.capitalize()} ({args.operator})",
        filename="decision_boundaries.png",
    )

    plot_training_curves(
        histories={f"ONN ({args.operator})": onn_history, "Baseline MLP": mlp_history},
        filename="training_curves.png",
    )

    plot_operator_trajectories(onn, filename="operator_trajectories.png")
    plot_operator_response(onn, layer_idx=0, filename="operator_response.png")
    plot_comparison_bar(results, filename="comparison_bar.png")

    print(f"\n{'='*60}")
    print("  Done! All outputs saved to outputs/")
    print(f"  ONN best val acc:      {onn_history['best_val_acc']:.4f}")
    print(f"  Baseline best val acc: {mlp_history['best_val_acc']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
