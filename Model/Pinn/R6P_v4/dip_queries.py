from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .physics_adapter import AdapterOutputs


def compute_local_curve_features(coarse_line: torch.Tensor) -> torch.Tensor:
    """Extract local extrema and gradient features from the Stage-1 coarse line.

    Returns a (B, T, 4) tensor with:
        0: first derivative (gradient, captures sudden changes)
        1: second derivative (curvature, captures sharpness of dips/peaks)
        2: local-min indicator (1 where the line has a local minimum)
        3: local-max indicator (1 where the line has a local maximum)
    """
    # Pad to keep length T after differencing
    first = F.pad(coarse_line[:, 1:] - coarse_line[:, :-1], (0, 1), mode="replicate")
    second = F.pad(first[:, 1:] - first[:, :-1], (0, 1), mode="replicate")

    # Local extrema: sign change in first derivative
    prev_sign = torch.sign(F.pad(first[:, :-1], (1, 0), mode="replicate"))
    curr_sign = torch.sign(first)
    local_min = ((prev_sign < 0) & (curr_sign > 0)).float()
    local_max = ((prev_sign > 0) & (curr_sign < 0)).float()

    return torch.stack([first, second, local_min, local_max], dim=-1)


@dataclass
class DipQueryOutputs:
    existence_logits: torch.Tensor  # (B, Q) — sigmoid > 0.5 == "this slot is a real dip"
    location: torch.Tensor  # (B, Q) in [0, 1], normalized position along the frequency axis
    depth: torch.Tensor  # (B, Q) positive dB magnitude
    dip_curve: torch.Tensor  # (B, T) coarse_line with existence-weighted notches subtracted


def _build_dip_curve_from_queries(
    coarse_line: torch.Tensor,
    existence_logits: torch.Tensor,
    location: torch.Tensor,
    depth: torch.Tensor,
    freq_axis_ghz: torch.Tensor,
    dip_sigma_bins: float,
    *,
    hard_inference: bool,
) -> torch.Tensor:
    seq_len = coarse_line.shape[1]
    freq_min = freq_axis_ghz[0]
    freq_max = freq_axis_ghz[-1]
    bin_width_ghz = (freq_max - freq_min) / max(seq_len - 1, 1)
    width_ghz = max(1.0e-4, dip_sigma_bins * float(bin_width_ghz.item()))

    existence_prob = torch.sigmoid(existence_logits)
    if hard_inference:
        existence_prob = (existence_prob > 0.5).float()

    center_freq = freq_min + location * (freq_max - freq_min)  # (B, Q)
    delta = (freq_axis_ghz.view(1, 1, -1) - center_freq.unsqueeze(-1)) / width_ghz  # (B, Q, T)
    notch = torch.exp(-0.5 * delta.pow(2))
    magnitude = existence_prob.unsqueeze(-1) * depth.unsqueeze(-1) * notch  # (B, Q, T)
    residual = magnitude.sum(dim=1)  # (B, T)
    return coarse_line - residual


class DipQueryDecoder(nn.Module):
    """DETR-style set-prediction head: a small fixed number of learned query
    slots cross-attend over the Stage-1 decoder features and each emit
    (existence, location, depth). Trained with Hungarian matching against the
    variable-size set of true dips, so there is no dense per-bin classification
    and no background/foreground imbalance to tune around — a slot is simply
    matched or it isn't.
    """

    def __init__(self, model_cfg: ModelConfig, physics_cfg) -> None:
        super().__init__()
        self.num_queries = int(model_cfg.num_queries)
        self.dip_sigma_bins = float(model_cfg.dip_sigma_bins)
        d_model = model_cfg.d_model

        self.query_embed = nn.Parameter(torch.randn(self.num_queries, d_model) * 0.02)
        self.local_feature_proj = nn.Linear(4, d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, model_cfg.query_attention_heads, dropout=model_cfg.dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, model_cfg.query_attention_heads, dropout=model_cfg.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, model_cfg.query_ffn_hidden),
            nn.GELU(),
            nn.Linear(model_cfg.query_ffn_hidden, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, model_cfg.query_head_hidden),
            nn.GELU(),
            nn.Linear(model_cfg.query_head_hidden, 3),
        )

    def forward(
        self,
        stage_features: torch.Tensor,
        coarse_line: torch.Tensor,
        adapter_outputs: AdapterOutputs,
        freq_axis_ghz: torch.Tensor,
        *,
        hard_inference: bool = False,
    ) -> DipQueryOutputs:
        memory = stage_features * (1.0 + adapter_outputs.film_scale) + adapter_outputs.film_shift
        # Inject local extrema/gradient features so queries can attend to sharp
        # changes in the coarse line instead of only smooth global context.
        local_feats = compute_local_curve_features(coarse_line)
        memory = memory + self.local_feature_proj(local_feats)
        batch_size = memory.shape[0]

        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        attn_out, _ = self.cross_attn(queries, memory, memory, need_weights=False)
        queries = self.norm1(queries + attn_out)
        self_attn_out, _ = self.self_attn(queries, queries, queries, need_weights=False)
        queries = self.norm2(queries + self_attn_out)
        queries = self.norm3(queries + self.ffn(queries))

        raw = self.head(queries)  # (B, Q, 3)
        existence_logits = raw[..., 0]
        location = torch.sigmoid(raw[..., 1])
        depth = F.softplus(raw[..., 2])

        dip_curve = _build_dip_curve_from_queries(
            coarse_line,
            existence_logits,
            location,
            depth,
            freq_axis_ghz,
            self.dip_sigma_bins,
            hard_inference=hard_inference,
        )
        return DipQueryOutputs(
            existence_logits=existence_logits,
            location=location,
            depth=depth,
            dip_curve=dip_curve,
        )
