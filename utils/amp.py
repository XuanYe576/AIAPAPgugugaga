"""Shared CUDA AMP helpers for training scripts."""

from __future__ import annotations

from contextlib import nullcontext

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    TORCH_AVAILABLE = False


class NullGradScaler:
    """Fallback scaler used when AMP is unavailable or disabled."""

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer) -> None:
        return None

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


def cuda_amp_enabled(device: str, use_amp: bool) -> bool:
    return bool(
        TORCH_AVAILABLE
        and use_amp
        and device == "cuda"
        and torch.cuda.is_available()
    )


def build_grad_scaler(device: str, use_amp: bool):
    if cuda_amp_enabled(device, use_amp):
        # GH200-class CUDA targets are best served by bfloat16 autocast,
        # which does not need gradient scaling.
        return NullGradScaler()
    return NullGradScaler()


def autocast_context(device: str, use_amp: bool):
    if cuda_amp_enabled(device, use_amp):
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()
