"""
# ID: UTILS
# Purpose: Shared utility functions: seeding, metrics, model parameter counting,
#          checkpoint save/load, and logging helpers.
"""

import os
import torch
import numpy as np
import random
from typing import Dict, Any


def set_seed(seed: int = 42):
    """
    # ID: SET_SEED
    # Purpose: Set all RNG seeds for full reproducibility.
    # Inputs: seed - int
    # Side effects: modifies global RNG state for torch, numpy, random, and CUDA
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    # ID: COUNT_PARAMS
    # Purpose: Count total trainable parameters in a model.
    # Outputs: int - number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """
    # ID: GET_DEVICE
    # Purpose: Return the best available compute device.
    # Outputs: torch.device - 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model: torch.nn.Module, path: str, metadata: Dict[str, Any] = None):
    """
    # ID: SAVE_CKPT
    # Purpose: Save model state dict and optional metadata to disk.
    # Inputs:
    #   model    - nn.Module
    #   path     - str, output file path (.pt)
    #   metadata - dict, extra info to store (epoch, accuracy, etc.)
    # Side effects: creates file at path
    # Failure modes: IOError if directory does not exist
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if metadata:
        payload.update(metadata)
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: str) -> Dict[str, Any]:
    """
    # ID: LOAD_CKPT
    # Purpose: Load model state dict from checkpoint file.
    # Inputs: model - nn.Module, path - str
    # Outputs: metadata dict (everything except 'state_dict')
    # Failure modes: FileNotFoundError if path does not exist
    """
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload.pop("state_dict"))
    return payload


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    # ID: ACCURACY
    # Purpose: Compute classification accuracy from logits.
    # Inputs:
    #   logits - (batch, n_classes) float tensor
    #   labels - (batch,) long tensor
    # Outputs: float in [0, 1]
    """
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    # ID: FORMAT_METRICS
    # Purpose: Format a metrics dict as a readable string for logging.
    # Inputs: metrics - dict {name: float}
    # Outputs: str like 'loss=0.123 acc=0.987'
    """
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
