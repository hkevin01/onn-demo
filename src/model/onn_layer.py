"""
# ID: ONN_LAYER
# Purpose: Single ONN layer that applies a learnable operator to inputs,
#          followed by an optional activation. Replaces a standard nn.Linear.
#
# Requirement: Layer must support all operators in OPERATOR_REGISTRY.
# Inputs: x - tensor (batch, in_features)
# Outputs: tensor (batch, out_features)
# References: Kiranyaz et al., "Operational Neural Networks" (2021)
"""

import torch
import torch.nn as nn
from .operators import build_operator, OPERATOR_REGISTRY


class ONNLayer(nn.Module):
    """
    # ID: ONN_LAYER_CLASS
    # Purpose: Wraps a learnable operator neuron into a standard nn.Module layer.
    #          Supports batch normalization and dropout for regularization.
    #
    # Inputs:
    #   in_features   - int, input dimensionality
    #   out_features  - int, output dimensionality
    #   operator      - str, name of operator ('polynomial','sinusoidal','gaussian','multiplicative')
    #   activation    - nn.Module or None, applied after operator (default ReLU)
    #   use_bn        - bool, apply BatchNorm1d after operator (default True)
    #   dropout       - float, dropout rate (default 0.0)
    #   operator_kwargs - dict, extra kwargs forwarded to operator constructor
    #
    # Preconditions: in_features > 0, out_features > 0
    # Postconditions: returns tensor (batch, out_features)
    # Failure modes: invalid operator name raises ValueError
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        operator: str = "polynomial",
        activation: nn.Module = None,
        use_bn: bool = True,
        dropout: float = 0.0,
        operator_kwargs: dict = None,
    ):
        super().__init__()
        operator_kwargs = operator_kwargs or {}
        self.operator = build_operator(operator, in_features, out_features, **operator_kwargs)
        self.activation = activation if activation is not None else nn.ReLU()
        self.bn = nn.BatchNorm1d(out_features) if use_bn else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.operator_name = operator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.operator(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.drop(x)
        return x

    def get_operator_params(self) -> dict:
        """
        # ID: GET_OP_PARAMS
        # Purpose: Return a snapshot of learnable operator parameters for visualization.
        # Outputs: dict mapping param name -> cloned detached tensor
        """
        return {
            name: param.detach().clone()
            for name, param in self.operator.named_parameters()
        }
