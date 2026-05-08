"""Shared optimizer helpers for training scripts."""

from __future__ import annotations

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    TORCH_AVAILABLE = False


def create_optimizer(
    optimizer_name: str,
    parameters,
    *,
    lr: float,
    weight_decay: float,
):
    if not TORCH_AVAILABLE:
        raise ModuleNotFoundError("torch is required to create an optimizer.")

    name = optimizer_name.lower()
    if name == "adagrad":
        return torch.optim.Adagrad(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")
