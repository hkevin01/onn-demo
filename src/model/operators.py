"""
# ID: OPERATORS
# Purpose: Learnable operator neurons for Operational Neural Networks.
# Each operator replaces the standard linear transform y = W*x + b with a
# parameterized nonlinear function phi(x; theta) that is learned during training.
#
# Requirement: Each operator must be differentiable and support autograd.
# Inputs: x - tensor of shape (batch, in_features)
# Outputs: tensor of shape (batch, out_features)
# References: Kiranyaz et al., "Operational Neural Networks" (2021)
"""

import torch
import torch.nn as nn
import math


class PolynomialOperator(nn.Module):
    """
    # ID: OP_POLY
    # Purpose: Applies a learnable polynomial transformation per output neuron.
    # y_j = sum_i (a_ij * x_i^p_ij + b_ij * x_i) + bias_j
    # where a_ij and p_ij are learned parameters.
    #
    # Inputs:
    #   in_features  - int, number of input dimensions
    #   out_features - int, number of output neurons
    #   degree       - int, max polynomial degree (default 3)
    # Preconditions: in_features > 0, out_features > 0, degree >= 1
    # Postconditions: returns tensor of shape (batch, out_features)
    # Failure modes: NaN from large exponent - mitigated by input clamping
    """

    def __init__(self, in_features: int, out_features: int, degree: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        # Learnable coefficient matrix: (out, in, degree+1) - one coeff per degree
        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, degree + 1) * 0.1
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        # Clamp to avoid numerical explosion in power ops
        x_safe = x.clamp(-10.0, 10.0)

        # Build polynomial basis: (batch, in_features, degree+1)
        batch = x_safe.shape[0]
        basis = torch.stack(
            [x_safe ** d for d in range(self.degree + 1)], dim=2
        )  # (batch, in, degree+1)

        # Weighted sum: coeffs (out, in, deg+1), basis (batch, in, deg+1)
        # -> (batch, out)
        out = torch.einsum("bid,oid->bo", basis, self.coeffs) + self.bias
        return out


class SinusoidalOperator(nn.Module):
    """
    # ID: OP_SIN
    # Purpose: Applies a learnable sinusoidal transformation.
    # y_j = sum_i w_ij * sin(freq_ij * x_i + phase_ij) + bias_j
    #
    # Inputs:
    #   in_features  - int
    #   out_features - int
    # Preconditions: in_features > 0, out_features > 0
    # Postconditions: returns tensor (batch, out_features)
    # Side effects: none
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.1
        )
        self.freq = nn.Parameter(
            torch.ones(out_features, in_features) + torch.randn(out_features, in_features) * 0.1
        )
        self.phase = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        # sin_vals: (batch, out, in) via broadcasting
        sin_vals = torch.sin(
            self.freq.unsqueeze(0) * x.unsqueeze(1) + self.phase.unsqueeze(0)
        )  # (batch, out, in)
        out = (self.weight.unsqueeze(0) * sin_vals).sum(dim=2) + self.bias
        return out


class GaussianOperator(nn.Module):
    """
    # ID: OP_GAUSS
    # Purpose: Applies a learnable Gaussian (RBF) transformation.
    # y_j = sum_i w_ij * exp(-sigma_ij * (x_i - mu_ij)^2) + bias_j
    #
    # Inputs:
    #   in_features  - int
    #   out_features - int
    # Preconditions: in_features > 0, out_features > 0
    # Postconditions: returns tensor (batch, out_features), values in (-inf, inf)
    # Failure modes: sigma going negative - mitigated with softplus
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.5
        )
        self.mu = nn.Parameter(torch.randn(out_features, in_features) * 0.5)
        # log_sigma so we can apply softplus to keep sigma positive
        self.log_sigma = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.nn.functional.softplus(self.log_sigma) + 1e-6
        # diff: (batch, out, in)
        diff = x.unsqueeze(1) - self.mu.unsqueeze(0)
        gauss = torch.exp(-sigma.unsqueeze(0) * diff ** 2)
        out = (self.weight.unsqueeze(0) * gauss).sum(dim=2) + self.bias
        return out


class MultiplicativeOperator(nn.Module):
    """
    # ID: OP_MULT
    # Purpose: Learns pairwise multiplicative interactions between inputs.
    # y_j = sum_i w_ij * x_i + sum_{i<k} v_ijk * x_i * x_k + bias_j
    # Approximated efficiently via factored form (rank-r factorization).
    #
    # Inputs:
    #   in_features  - int
    #   out_features - int
    #   rank         - int, factorization rank (default 4)
    # Preconditions: in_features > 0, out_features > 0, rank >= 1
    # Postconditions: returns tensor (batch, out_features)
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Factored interaction: U (out, rank, in), V (out, rank, in)
        self.U = nn.Parameter(torch.randn(out_features, rank, in_features) * 0.05)
        self.V = nn.Parameter(torch.randn(out_features, rank, in_features) * 0.05)
        self.bias_mult = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_part = self.linear(x)
        # Interaction: sum_r (U_r @ x) * (V_r @ x) -> (batch, out)
        u_proj = torch.einsum("ori,bi->bor", self.U, x)  # (batch, out, rank)
        v_proj = torch.einsum("ori,bi->bor", self.V, x)  # (batch, out, rank)
        interaction = (u_proj * v_proj).sum(dim=2) + self.bias_mult
        return linear_part + interaction


# Registry for easy instantiation by name
OPERATOR_REGISTRY = {
    "polynomial": PolynomialOperator,
    "sinusoidal": SinusoidalOperator,
    "gaussian": GaussianOperator,
    "multiplicative": MultiplicativeOperator,
}


def build_operator(name: str, in_features: int, out_features: int, **kwargs) -> nn.Module:
    """
    # ID: BUILD_OP
    # Purpose: Factory function to instantiate an operator by name.
    # Inputs: name - str key in OPERATOR_REGISTRY
    # Outputs: nn.Module operator instance
    # Failure modes: KeyError if name not in registry
    """
    if name not in OPERATOR_REGISTRY:
        raise ValueError(f"Unknown operator '{name}'. Choose from: {list(OPERATOR_REGISTRY.keys())}")
    return OPERATOR_REGISTRY[name](in_features, out_features, **kwargs)
