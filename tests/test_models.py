"""
# ID: TEST_MODELS
# Purpose: Integration tests for ONNModel, BaselineMLP, and ONNLayer.
# Requirement: Models must forward-pass without error and produce correct shapes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
from src.model.onn_layer import ONNLayer
from src.model.onn_model import ONNModel
from src.model.baseline_model import BaselineMLP, BaselineCNN
from src.model.operators import OPERATOR_REGISTRY


BATCH = 8
IN_F = 2
N_CLASSES = 2


@pytest.mark.parametrize("op_name", list(OPERATOR_REGISTRY.keys()))
def test_onn_layer_forward(op_name):
    layer = ONNLayer(in_features=IN_F, out_features=16, operator=op_name)
    x = torch.randn(BATCH, IN_F)
    y = layer(x)
    assert y.shape == (BATCH, 16)


@pytest.mark.parametrize("op_name", list(OPERATOR_REGISTRY.keys()))
def test_onn_model_forward(op_name):
    model = ONNModel(IN_F, [16, 16], n_classes=N_CLASSES, operator=op_name)
    x = torch.randn(BATCH, IN_F)
    out = model(x)
    assert out.shape == (BATCH, N_CLASSES)


def test_onn_model_snapshot():
    model = ONNModel(IN_F, [16], n_classes=N_CLASSES, operator="polynomial")
    assert len(model.param_history) == 0
    model.snapshot_params()
    assert len(model.param_history) == 1
    snap = model.param_history[0]
    assert "layer_0" in snap


def test_onn_layer_get_params():
    layer = ONNLayer(IN_F, 8, operator="sinusoidal")
    params = layer.get_operator_params()
    assert len(params) > 0
    for v in params.values():
        assert isinstance(v, torch.Tensor)


def test_baseline_mlp_forward():
    mlp = BaselineMLP(IN_F, [16, 16], n_classes=N_CLASSES)
    x = torch.randn(BATCH, IN_F)
    out = mlp(x)
    assert out.shape == (BATCH, N_CLASSES)


def test_baseline_cnn_forward():
    cnn = BaselineCNN(in_channels=1, n_classes=10)
    x = torch.randn(4, 1, 28, 28)
    out = cnn(x)
    assert out.shape == (4, 10)


def test_onn_model_get_layer_outputs():
    model = ONNModel(IN_F, [16, 8], n_classes=N_CLASSES)
    x = torch.randn(BATCH, IN_F)
    activations = model.get_layer_outputs(x)
    assert len(activations) == 2
    assert activations[0].shape == (BATCH, 16)
    assert activations[1].shape == (BATCH, 8)
