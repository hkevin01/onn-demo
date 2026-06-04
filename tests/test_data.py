"""
# ID: TEST_DATA
# Purpose: Tests for data loading utilities.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
from src.data import load_dataset, get_full_tensors, make_spiral


@pytest.mark.parametrize("name", ["moons", "circles", "spiral"])
def test_load_dataset_shapes(name):
    train_loader, test_loader, in_f, n_cls = load_dataset(name=name, n_samples=200)
    assert in_f == 2
    assert n_cls == 2
    for X, y in train_loader:
        assert X.shape[1] == 2
        assert y.dtype == torch.int64
        break


@pytest.mark.parametrize("name", ["moons", "circles", "spiral"])
def test_get_full_tensors(name):
    X_train, X_test, y_train, y_test = get_full_tensors(name=name, n_samples=200)
    assert X_train.shape[1] == 2
    assert X_test.shape[1] == 2
    assert y_train.dtype == torch.int64


def test_make_spiral():
    X, y = make_spiral(n_samples=100)
    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert set(y.tolist()) == {0, 1}


def test_load_dataset_invalid():
    with pytest.raises(ValueError):
        load_dataset(name="unknown_dataset")
