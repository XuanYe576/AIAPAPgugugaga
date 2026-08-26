from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import ModelConfig


def build_local_attention_mask(
    seq_len: int,
    window: int,
    *,
    causal: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    if window <= 0 or window >= seq_len:
        return torch.zeros((seq_len, seq_len), device=device)
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    distance = j - i
    if causal:
        allowed = (distance >= 0) & (distance <= window)
    else:
        allowed = distance.abs() <= window
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = mask.masked_fill(allowed, 0.0)
    return mask


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class FrequencyTokenExpander(nn.Module):
    def __init__(self, d_model: int, seq_len: int) -> None:
        super().__init__()
        freq_positions = torch.linspace(0.0, 1.0, seq_len, dtype=torch.float32).unsqueeze(-1)
        self.register_buffer("freq_positions", freq_positions, persistent=False)
        self.position_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, z_g: torch.Tensor) -> torch.Tensor:
        batch_size = z_g.shape[0]
        base_tokens = z_g.unsqueeze(1).expand(batch_size, self.freq_positions.shape[0], z_g.shape[-1])
        position_tokens = self.position_mlp(self.freq_positions).unsqueeze(0)
        return base_tokens + position_tokens


class SpectralBlock(nn.Module):
    def __init__(self, d_model: int, max_modes: int, dropout: float) -> None:
        super().__init__()
        self.max_modes = max_modes
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Linear(d_model, d_model)
        self.weight_real = nn.Parameter(torch.randn(max_modes, d_model) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(max_modes, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        freq_repr = torch.fft.rfft(x, dim=1, norm="ortho")
        modes = min(self.max_modes, freq_repr.shape[1])
        real = freq_repr[:, :modes].real
        imag = freq_repr[:, :modes].imag
        wr = self.weight_real[:modes].unsqueeze(0)
        wi = self.weight_imag[:modes].unsqueeze(0)
        mixed_real = real * wr - imag * wi
        mixed_imag = real * wi + imag * wr
        mixed = torch.complex(mixed_real, mixed_imag)
        full = torch.zeros_like(freq_repr)
        full[:, :modes] = mixed
        out = torch.fft.irfft(full, n=x.shape[1], dim=1, norm="ortho")
        return residual + self.dropout(self.residual_proj(out))


class WindowedTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_hidden: int,
        dropout: float,
        seq_len: int,
        attention_window: int,
        causal: bool,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        mask = build_local_attention_mask(seq_len, attention_window, causal=causal)
        self.register_buffer("attention_mask", mask, persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # need_weights=True keeps PyTorch on the legacy math attention path.
        # Some CUDA/cuDNN builds fail inside scaled_dot_product_attention here.
        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            need_weights=True,
            attn_mask=self.attention_mask.to(x.device),
        )
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


@dataclass
class DecoderOutputs:
    decoder_features: torch.Tensor
    coarse_line: torch.Tensor


class Stage1SpectralDecoder(nn.Module):
    def __init__(self, seq_len: int, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.expander = FrequencyTokenExpander(model_cfg.d_model, self.seq_len)
        self.position_encoding = SinusoidalPositionEncoding(model_cfg.d_model, self.seq_len)
        self.spectral_blocks = nn.ModuleList(
            [
                SpectralBlock(model_cfg.d_model, model_cfg.spectral_modes, model_cfg.dropout)
                for _ in range(model_cfg.num_spectral_blocks)
            ]
        )
        self.transformer_layers = nn.ModuleList(
            [
                WindowedTransformerLayer(
                    d_model=model_cfg.d_model,
                    num_heads=model_cfg.transformer_heads,
                    ffn_hidden=model_cfg.ffn_hidden,
                    dropout=model_cfg.dropout,
                    seq_len=self.seq_len,
                    attention_window=model_cfg.attention_window,
                    causal=model_cfg.attention_causal,
                )
                for _ in range(model_cfg.num_transformer_layers)
            ]
        )
        self.line_head = nn.Linear(model_cfg.d_model, 1)

    def forward(self, z_g: torch.Tensor) -> DecoderOutputs:
        x = self.expander(z_g)
        x = self.position_encoding(x)
        blocks = max(len(self.spectral_blocks), len(self.transformer_layers))
        for block_index in range(blocks):
            if block_index < len(self.spectral_blocks):
                x = self.spectral_blocks[block_index](x)
            if block_index < len(self.transformer_layers):
                x = self.transformer_layers[block_index](x)
        coarse_line = self.line_head(x).squeeze(-1)
        return DecoderOutputs(decoder_features=x, coarse_line=coarse_line)
