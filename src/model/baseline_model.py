"""
# ID: BASELINE_MODEL
# Purpose: Standard MLP and CNN baselines for comparison against ONN.
#
# Requirement: Architecturally equivalent parameter count to ONN for fair comparison.
# Inputs: x - tensor (batch, in_features) for MLP, (batch, C, H, W) for CNN
# Outputs: tensor (batch, n_classes)
"""

import torch
import torch.nn as nn
from typing import List


class BaselineMLP(nn.Module):
    """
    # ID: BASELINE_MLP
    # Purpose: Standard MLP with ReLU activations, batch norm, and dropout.
    #          Serves as comparison baseline for ONN on 2D classification tasks.
    #
    # Inputs:
    #   in_features  - int
    #   hidden_dims  - List[int]
    #   n_classes    - int
    #   dropout      - float (default 0.1)
    #   use_bn       - bool (default True)
    # Postconditions: returns logits tensor (batch, n_classes)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int],
        n_classes: int = 2,
        dropout: float = 0.1,
        use_bn: bool = True,
    ):
        super().__init__()
        layers = []
        prev = in_features
        for i, hidden in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, hidden))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            if dropout > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev = hidden

        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class BaselineCNN(nn.Module):
    """
    # ID: BASELINE_CNN
    # Purpose: Small CNN for image classification (e.g., MNIST 28x28).
    #          Used as baseline comparison when ONN operates on flattened images.
    #
    # Inputs:
    #   in_channels  - int, number of input channels (default 1 for grayscale)
    #   n_classes    - int, number of output classes
    #   dropout      - float (default 0.25)
    # Preconditions: input image must be >= 7x7 spatially
    # Postconditions: returns logits (batch, n_classes)
    """

    def __init__(self, in_channels: int = 1, n_classes: int = 10, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 14x14

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 7x7

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
