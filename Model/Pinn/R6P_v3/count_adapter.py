from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


class DipCountAdapter(nn.Module):
    """Predicts how many distinct resonance dips an antenna has.

    Count is a whole-antenna property, not tied to any one frequency bin, so
    this reads the pooled geometry embedding z_g rather than the per-bin
    decoder tokens the dip experts use. Output is classification logits over
    {0, 1, ..., max_count} rather than a scalar regression, since the target
    is a small non-negative integer and cross-entropy gives a much sharper
    training signal than MSE would here.
    """

    def __init__(self, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.max_count = int(model_cfg.max_dip_count)
        self.net = nn.Sequential(
            nn.Linear(model_cfg.d_model, model_cfg.count_adapter_hidden),
            nn.GELU(),
            nn.Linear(model_cfg.count_adapter_hidden, self.max_count + 1),
        )

    def forward(self, z_g: torch.Tensor) -> torch.Tensor:
        return self.net(z_g)
