from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, PhysicsConfig
from .physics_adapter import AdapterOutputs


@dataclass
class DipExpertOutputs:
    dip_presence_logits: torch.Tensor
    dip_offset_ghz: torch.Tensor
    dip_depth_db: torch.Tensor
    dip_curve: torch.Tensor
    expert_gates: torch.Tensor


class ResBlock1D(nn.Module):
    """Pre-norm residual MLP block applied per frequency position, (B, T, C) -> (B, T, C)."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class DownBlock1D(nn.Module):
    """Halves the sequence length (adaptive avg-pool) then applies a residual block."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.block = ResBlock1D(channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_len = max(1, x.shape[1] // 2)
        pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), target_len).transpose(1, 2)
        return self.block(pooled)


class UpBlock1D(nn.Module):
    """Upsamples to a skip connection's length, fuses via concat+linear, then a residual block."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.fuse = nn.Linear(channels * 2, channels)
        self.block = ResBlock1D(channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = F.interpolate(x.transpose(1, 2), size=skip.shape[1], mode="linear", align_corners=False)
        fused = self.fuse(torch.cat([upsampled.transpose(1, 2), skip], dim=-1))
        return self.block(fused)


class DipResUNet(nn.Module):
    """Lightweight 1D residual U-Net over the frequency axis, with skip connections.

    Locating an arbitrary number of resonance dips is a dense-segmentation
    problem (which bins are dips), not a single-location one. A single global
    self-attention block has no bias toward that, so this replaces it with an
    encoder/decoder whose skip connections let multiple, independently
    confident dips of different widths coexist in the output instead of
    competing for one shared representation.
    """

    def __init__(self, channels: int, depth: int, dropout: float) -> None:
        super().__init__()
        self.depth = max(0, int(depth))
        self.down_blocks = nn.ModuleList([DownBlock1D(channels, dropout) for _ in range(self.depth)])
        self.bottleneck = ResBlock1D(channels, dropout)
        self.up_blocks = nn.ModuleList([UpBlock1D(channels, dropout) for _ in range(self.depth)])
        self.out_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [x]
        h = x
        for down in self.down_blocks:
            h = down(h)
            skips.append(h)
        h = self.bottleneck(h)
        for level, up in enumerate(self.up_blocks):
            skip = skips[self.depth - 1 - level]
            h = up(h, skip)
        return self.out_norm(h)


class AttentionMoEDipExperts(nn.Module):
    def __init__(
        self,
        seq_len: int,
        freq_axis_ghz: torch.Tensor,
        model_cfg: ModelConfig,
        physics_cfg: PhysicsConfig,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.max_offset_bins = float(model_cfg.max_offset_bins)
        self.dip_sigma_bins = float(model_cfg.dip_sigma_bins)
        self.register_buffer("freq_axis_ghz", freq_axis_ghz.float().reshape(1, self.seq_len), persistent=False)
        cond_dim = int(physics_cfg.conditioning_dim)
        input_dim = model_cfg.d_model + 1 + cond_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_cfg.d_model),
            nn.GELU(),
            nn.Linear(model_cfg.d_model, model_cfg.d_model),
        )
        self.unet = DipResUNet(model_cfg.d_model, model_cfg.dip_unet_depth, model_cfg.dropout)
        self.gate_head = nn.Sequential(
            nn.Linear(model_cfg.d_model, model_cfg.expert_hidden),
            nn.GELU(),
            nn.Linear(model_cfg.expert_hidden, model_cfg.num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(model_cfg.d_model, model_cfg.expert_hidden),
                    nn.GELU(),
                    nn.Linear(model_cfg.expert_hidden, 3),
                )
                for _ in range(model_cfg.num_experts)
            ]
        )
        # Learnable, not a fixed constant: initialized negative to reflect
        # how rare true dip bins are, but training must be free to walk it
        # back once the MoE head has enough separation, or every logit gets
        # stuck under the sigmoid>0.5 threshold no matter how well the rest
        # of the network learns to rank bins.
        self.presence_bias = nn.Parameter(torch.tensor(-2.0))

    def _build_dip_curve(
        self,
        coarse_line: torch.Tensor,
        dip_presence_logits: torch.Tensor,
        dip_offset_ghz: torch.Tensor,
        dip_depth_db: torch.Tensor,
    ) -> torch.Tensor:
        freq_axis = self.freq_axis_ghz.to(coarse_line.device)
        if freq_axis.shape[1] > 1:
            bin_width = float((freq_axis[0, 1] - freq_axis[0, 0]).item())
        else:
            bin_width = 1.0
        width_ghz = max(1.0e-4, self.dip_sigma_bins * abs(bin_width))
        # Independent per-bin confidence (not a softmax over the whole axis) so
        # several real, simultaneous resonances can each be confidently
        # predicted without diluting a shared probability budget.
        anchor_weights = torch.sigmoid(dip_presence_logits)
        centers = freq_axis + dip_offset_ghz
        delta = (freq_axis.unsqueeze(1) - centers.unsqueeze(-1)) / width_ghz
        notch_basis = torch.exp(-0.5 * delta.pow(2))
        dip_magnitude = anchor_weights.unsqueeze(-1) * dip_depth_db.unsqueeze(-1) * notch_basis
        dip_residual = dip_magnitude.sum(dim=1)
        return coarse_line - dip_residual

    def forward(
        self,
        stage_features: torch.Tensor,
        coarse_line: torch.Tensor,
        adapter_outputs: AdapterOutputs,
    ) -> DipExpertOutputs:
        conditioned_stage = stage_features * (1.0 + adapter_outputs.film_scale) + adapter_outputs.film_shift
        dip_input = torch.cat(
            [conditioned_stage, coarse_line.unsqueeze(-1), adapter_outputs.conditioning],
            dim=-1,
        )
        token_features = self.input_proj(dip_input)
        context = self.unet(token_features)
        expert_gates = torch.softmax(self.gate_head(context), dim=-1)
        expert_outputs = torch.stack([expert(context) for expert in self.experts], dim=2)
        mixed = (expert_gates.unsqueeze(-1) * expert_outputs).sum(dim=2)
        dip_presence_logits = mixed[..., 0] + self.presence_bias
        max_offset_ghz = self.max_offset_bins * abs(float(self.freq_axis_ghz[0, 1] - self.freq_axis_ghz[0, 0])) if self.seq_len > 1 else 1.0
        dip_offset_ghz = torch.tanh(mixed[..., 1]) * max_offset_ghz
        dip_depth_db = F.softplus(mixed[..., 2])
        dip_curve = self._build_dip_curve(coarse_line, dip_presence_logits, dip_offset_ghz, dip_depth_db)
        return DipExpertOutputs(
            dip_presence_logits=dip_presence_logits,
            dip_offset_ghz=dip_offset_ghz,
            dip_depth_db=dip_depth_db,
            dip_curve=dip_curve,
            expert_gates=expert_gates,
        )
