from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FusionOutputs:
    final_curve: torch.Tensor
    gate: torch.Tensor


def _topk_mask(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Binary mask selecting, per sample, the k bins with the highest logit.

    Rank-based selection only depends on the relative order of bins, so it
    stays correct even if the whole logit distribution is shifted below a
    fixed absolute threshold (e.g. an under-calibrated presence head) —
    unlike `sigmoid(logits) > 0.5`, it always selects something when k > 0.
    """
    batch_size, seq_len = logits.shape
    order = logits.argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    arange = torch.arange(seq_len, device=logits.device).unsqueeze(0).expand(batch_size, seq_len)
    ranks.scatter_(1, order, arange)
    k = k.clamp(0, seq_len).unsqueeze(1)
    return (ranks < k).float()


class SoftDipFusion(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = float(max(temperature, 1.0e-6))

    def forward(
        self,
        coarse_line: torch.Tensor,
        dip_curve: torch.Tensor,
        dip_presence_logits: torch.Tensor,
        *,
        count_logits: torch.Tensor | None = None,
        hard_inference: bool = False,
    ) -> FusionOutputs:
        soft_gate = torch.sigmoid(dip_presence_logits / self.temperature)
        if hard_inference:
            if count_logits is not None:
                predicted_count = count_logits.argmax(dim=-1)
                gate = _topk_mask(dip_presence_logits, predicted_count)
            else:
                gate = (soft_gate > 0.5).float()
        else:
            gate = soft_gate
        final_curve = gate * dip_curve + (1.0 - gate) * coarse_line
        return FusionOutputs(final_curve=final_curve, gate=gate)
