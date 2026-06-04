"""
# ID: ONN_MODEL
# Purpose: Full ONN classifier/regressor built from stacked ONNLayers.
#
# Requirement: Model must support configurable depth, operator types, and task heads.
# Inputs: x - tensor (batch, in_features)
# Outputs: tensor (batch, n_classes) for classification, (batch, 1) for regression
# Preconditions: config lists must be consistent
# References: Kiranyaz et al., "Operational Neural Networks" (2021)
"""

import torch
import torch.nn as nn
from typing import List, Optional
from .onn_layer import ONNLayer


class ONNModel(nn.Module):
    """
    # ID: ONN_MODEL_CLASS
    # Purpose: Stacked ONN with a linear classification/regression head.
    #
    # Inputs:
    #   in_features    - int, input dimension
    #   hidden_dims    - List[int], sizes of hidden layers
    #   n_classes      - int, number of output classes (1 for regression)
    #   operator       - str or List[str], operator(s) to use per layer
    #   dropout        - float, dropout rate (default 0.1)
    #   use_bn         - bool, batch norm on hidden layers (default True)
    #   operator_kwargs - dict, forwarded to each operator
    #
    # Preconditions: len(hidden_dims) >= 1
    # Postconditions: model is fully initialized and forward-ready
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int],
        n_classes: int = 2,
        operator: str = "polynomial",
        dropout: float = 0.1,
        use_bn: bool = True,
        operator_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        self.operator_name = operator

        operator_kwargs = operator_kwargs or {}
        layers = []
        prev = in_features
        for i, hidden in enumerate(hidden_dims):
            activation = nn.ReLU() if i < len(hidden_dims) - 1 else nn.Tanh()
            layers.append(
                ONNLayer(
                    in_features=prev,
                    out_features=hidden,
                    operator=operator,
                    activation=activation,
                    use_bn=use_bn,
                    dropout=dropout if i < len(hidden_dims) - 1 else 0.0,
                    operator_kwargs=operator_kwargs,
                )
            )
            prev = hidden

        self.onn_layers = nn.ModuleList(layers)
        self.head = nn.Linear(prev, n_classes)

        # Storage for parameter trajectory snapshots (filled by trainer)
        self.param_history: List[dict] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.onn_layers:
            x = layer(x)
        return self.head(x)

    def snapshot_params(self):
        """
        # ID: SNAPSHOT_PARAMS
        # Purpose: Capture current operator parameters for trajectory visualization.
        # Side effects: appends to self.param_history
        """
        snap = {}
        for i, layer in enumerate(self.onn_layers):
            snap[f"layer_{i}"] = layer.get_operator_params()
        self.param_history.append(snap)

    def get_layer_outputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        # ID: GET_LAYER_OUTPUTS
        # Purpose: Return intermediate activations for analysis.
        # Outputs: list of tensors, one per ONN layer
        """
        activations = []
        for layer in self.onn_layers:
            x = layer(x)
            activations.append(x.detach())
        return activations
