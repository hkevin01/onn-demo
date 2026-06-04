"""
# ID: TEST_OPERATORS
# Purpose: Unit tests for all operator implementations.
# Requirement: Each operator must produce correct output shapes and gradients.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
from src.model.operators import (
    PolynomialOperator,
    SinusoidalOperator,
    GaussianOperator,
    MultiplicativeOperator,
    build_operator,
    OPERATOR_REGISTRY,
)


BATCH = 16
IN_F = 4
OUT_F = 8


@pytest.mark.parametrize("op_name", list(OPERATOR_REGISTRY.keys()))
def test_operator_output_shape(op_name):
    op = build_operator(op_name, IN_F, OUT_F)
    x = torch.randn(BATCH, IN_F)
    y = op(x)
    assert y.shape == (BATCH, OUT_F), f"{op_name}: expected ({BATCH},{OUT_F}), got {y.shape}"


@pytest.mark.parametrize("op_name", list(OPERATOR_REGISTRY.keys()))
def test_operator_gradients(op_name):
    op = build_operator(op_name, IN_F, OUT_F)
    x = torch.randn(BATCH, IN_F, requires_grad=True)
    y = op(x).sum()
    y.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any(), f"{op_name}: NaN gradients"


def test_polynomial_degree_1():
    op = PolynomialOperator(2, 4, degree=1)
    x = torch.randn(8, 2)
    y = op(x)
    assert y.shape == (8, 4)


def test_build_operator_invalid():
    with pytest.raises(ValueError):
        build_operator("invalid_op", 2, 4)


def test_operator_no_nan_output():
    for name in OPERATOR_REGISTRY:
        op = build_operator(name, IN_F, OUT_F)
        x = torch.randn(BATCH, IN_F)
        y = op(x)
        assert not torch.isnan(y).any(), f"{name}: NaN in output"
