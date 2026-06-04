"""
# ID: DATA
# Purpose: Dataset generation and loading utilities for ONN demo.
#          Provides toy 2D datasets (moons, circles, spirals) and MNIST subset.
#
# Requirement: All datasets must return (X_train, X_test, y_train, y_test) as tensors.
# Inputs: dataset name, n_samples, noise, random_state
# Outputs: train/test split tensors
"""

import torch
import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple


def make_spiral(n_samples: int = 1000, noise: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    # ID: MAKE_SPIRAL
    # Purpose: Generate 2-class spiral dataset - highly nonlinear, good ONN showcase.
    # Inputs:
    #   n_samples    - int, total samples (split equally between classes)
    #   noise        - float, Gaussian noise std
    #   random_state - int, RNG seed
    # Outputs: X (n_samples, 2), y (n_samples,) with labels 0/1
    """
    rng = np.random.RandomState(random_state)
    n = n_samples // 2
    theta = np.linspace(0, 4 * np.pi, n)

    X0 = np.stack([theta * np.cos(theta), theta * np.sin(theta)], axis=1)
    X1 = np.stack([-theta * np.cos(theta), -theta * np.sin(theta)], axis=1)

    X0 += rng.randn(n, 2) * noise
    X1 += rng.randn(n, 2) * noise

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)])
    return X.astype(np.float32), y.astype(np.int64)


def load_dataset(
    name: str = "moons",
    n_samples: int = 1000,
    noise: float = 0.15,
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    # ID: LOAD_DATASET
    # Purpose: Load a named toy dataset, scale it, and return DataLoaders.
    # Inputs:
    #   name         - str: 'moons' | 'circles' | 'spiral'
    #   n_samples    - int, total sample count
    #   noise        - float, label noise / feature noise
    #   test_size    - float, fraction for test split
    #   random_state - int, reproducibility seed
    #   batch_size   - int
    # Outputs: (train_loader, test_loader, in_features, n_classes)
    # Failure modes: unknown name raises ValueError
    """
    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        X, y = X.astype(np.float32), y.astype(np.int64)
    elif name == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
        X, y = X.astype(np.float32), y.astype(np.int64)
    elif name == "spiral":
        X, y = make_spiral(n_samples=n_samples, noise=noise, random_state=random_state)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose: moons, circles, spiral")

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_t = torch.from_numpy(X_train)
    X_test_t = torch.from_numpy(X_test)
    y_train_t = torch.from_numpy(y_train)
    y_test_t = torch.from_numpy(y_test)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    in_features = X.shape[1]
    n_classes = len(np.unique(y))
    return train_loader, test_loader, in_features, n_classes


def get_full_tensors(
    name: str = "moons",
    n_samples: int = 1000,
    noise: float = 0.15,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    # ID: GET_FULL_TENSORS
    # Purpose: Return raw tensors (X_train, X_test, y_train, y_test) for visualization.
    # Outputs: four tensors for train/test split
    """
    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        X, y = X.astype(np.float32), y.astype(np.int64)
    elif name == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
        X, y = X.astype(np.float32), y.astype(np.int64)
    elif name == "spiral":
        X, y = make_spiral(n_samples=n_samples, noise=noise, random_state=random_state)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return (
        torch.from_numpy(X_train),
        torch.from_numpy(X_test),
        torch.from_numpy(y_train),
        torch.from_numpy(y_test),
    )
