"""Patch Antenna AI R5.

Full R5 physics-informed surrogate aligned with the design intent captured in
`old/C.md`:

- geometry -> k-NN graph with geometry-derived node and edge features
- 3-layer GATv2-style graph encoder with global attention pooling
- geometry token + learned frequency tokens -> geometry-conditioned FEDformer
- complex-valued output head predicting [Re(Gamma), Im(Gamma)] per frequency
- physics-informed loss with complex MSE, dB loss, notch weighting, passivity,
  smoothness, and resonance-aware auxiliary heads

The script keeps backward compatibility with legacy datasets through explicit
layout flags, but the default target structure is the full 122-point complex
response.

For the uploaded magnitude-only raw data split into geometry catalogs and
per-patch S11 curves, use `Model/PINN/R5PINN_perF.py` or the surface
`main.py --usepinn` entry point.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset

GRID_HEIGHT = 10
GRID_WIDTH = 10
NUM_CELLS = GRID_HEIGHT * GRID_WIDTH


def _default_csv_path() -> Path:
    candidates = [
        REPO_ROOT / "Data" / "Full_122ComplexData.csv",
        REPO_ROOT / "Data" / "TrainData.csv",
        REPO_ROOT / "old" / "data" / "Full_1000Data.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _default_output_dir() -> Path:
    return REPO_ROOT / "results" / "patch_antenna_ai_r5"


@dataclass
class Config:
    csv_path: Path = field(default_factory=_default_csv_path)
    input_dim: int = NUM_CELLS
    seq_len: int = 122
    output_mode: str = "complex_122"
    batch_size: int = 32
    lr: float = 2e-3
    optimizer_name: str = "adamw"
    weight_decay: float = 1e-4
    min_lr_scale: float = 0.05
    epochs: int = 300
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    dmodel: int = 256
    geometry_embedding_dim: int = 256
    gnn_hidden_dim: int = 128
    gnn_layers: int = 3
    gnn_heads: int = 4
    edge_mlp_hidden: int = 64
    node_update_hidden: int = 128
    graph_k: int = 8
    decoder_heads: int = 4
    decoder_blocks: int = 1
    refinement_layers: int = 2
    decoder_ffn_hidden: int = 512
    dropout: float = 0.10
    freq_hz_start: float = 1.0e9
    freq_hz_stop: float = 6.0e9
    bit_flip_prob: float = 0.01
    block_jitter_prob: float = 0.15
    max_block_size: int = 2
    use_augmentation: bool = True
    random_seed: int = 42
    patience: int = 30
    gradient_clip: float = 1.0
    log_every_batches: int = 10
    warmup_ratio: float = 0.10
    use_amp: bool = True
    curriculum_epochs: int = 80
    curriculum_stride: int = 2
    loss_complex: float = 1.0
    loss_db: float = 0.25
    loss_notch: float = 0.10
    loss_passive: float = 0.10
    loss_smooth: float = 0.02
    notch_sigma_db: float = 4.0
    passive_limit: float = 1.0
    device: str = "auto"
    eval_split: str = ""
    output_dir: Path = field(default_factory=_default_output_dir)
    weights_filename: str = "gnn_fedformer_r5_full_best.pt"
    checkpoint_filename: str = "gnn_fedformer_r5_full_best.ckpt"
    history_filename: str = "history.csv"
    loss_plot_filename: str = "loss_curve.png"
    summary_filename: str = "summary.json"
    config_filename: str = "config.json"

    @property
    def weights_path(self) -> Path:
        return self.output_dir / self.weights_filename

    @property
    def history_path(self) -> Path:
        return self.output_dir / self.history_filename

    @property
    def loss_plot_path(self) -> Path:
        return self.output_dir / self.loss_plot_filename

    @property
    def summary_path(self) -> Path:
        return self.output_dir / self.summary_filename

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / self.checkpoint_filename


@dataclass(frozen=True)
class LayoutInfo:
    name: str
    seq_len: int
    is_complex: bool
    expected_cols: int


@dataclass(frozen=True)
class GraphStructure:
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    incoming_edges: tuple[torch.Tensor, ...]
    node_static_features: torch.Tensor
    num_nodes: int


@dataclass
class LossBreakdown:
    total: torch.Tensor
    complex_mse: torch.Tensor
    db_mae: torch.Tensor
    notch: torch.Tensor
    passive: torch.Tensor
    smooth: torch.Tensor


SUPPORTED_LAYOUTS: dict[str, LayoutInfo] = {
    "complex_122": LayoutInfo("complex_122", 122, True, NUM_CELLS + 2 * 122),
    "complex_61": LayoutInfo("complex_61", 61, True, NUM_CELLS + 2 * 61),
    "mag_only": LayoutInfo("mag_only", 61, False, NUM_CELLS + 61),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def create_optimizer(cfg: Config, parameters: list[torch.nn.Parameter] | iter) -> torch.optim.Optimizer:
    name = cfg.optimizer_name.lower()
    if name == "adagrad":
        return torch.optim.Adagrad(
            parameters,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer_name}")


def infer_layout(column_count: int, requested_mode: str) -> LayoutInfo:
    if requested_mode != "auto":
        layout = SUPPORTED_LAYOUTS[requested_mode]
        if column_count < layout.expected_cols:
            hint = (
                f"Requested {layout.name} expects at least {layout.expected_cols} columns "
                f"but found {column_count}. "
            )
            if column_count == SUPPORTED_LAYOUTS["mag_only"].expected_cols:
                hint += (
                    "This looks like a legacy 61-point magnitude CSV. "
                    "Use --output-mode mag_only for that file or provide a 122-point "
                    "complex dataset to run the full R5 target."
                )
            raise ValueError(hint)
        return layout

    for name in ("complex_122", "complex_61", "mag_only"):
        layout = SUPPORTED_LAYOUTS[name]
        if column_count >= layout.expected_cols:
            return layout

    raise ValueError(
        f"Unsupported CSV layout with {column_count} columns. "
        "Expected one of: 344 (100 + 122x2), 222 (100 + 61x2), or 161 (100 + 61)."
    )


class AntennaDataset(Dataset):
    """Dataset for flattened 10x10 geometry and complex S11 targets."""

    def __init__(
        self,
        csv_path: Path,
        input_dim: int = NUM_CELLS,
        output_mode: str = "complex_122",
    ) -> None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path, header=None)
        values = df.values.astype(np.float32)
        if values.ndim != 2:
            raise ValueError("CSV must be a 2D tabular array.")

        layout = infer_layout(values.shape[1], output_mode)

        self.csv_path = csv_path
        self.input_dim = input_dim
        self.seq_len = layout.seq_len
        self.output_mode = layout.name
        self.layout = layout

        x = values[:, :input_dim]
        if layout.is_complex:
            y_real = values[:, input_dim : input_dim + layout.seq_len]
            y_imag = values[
                :, input_dim + layout.seq_len : input_dim + 2 * layout.seq_len
            ]
        else:
            y_real = values[:, input_dim : input_dim + layout.seq_len]
            y_imag = np.zeros_like(y_real)

        self.X = torch.from_numpy((x > 0.5).astype(np.float32))
        self.Y = torch.stack(
            [torch.from_numpy(y_real), torch.from_numpy(y_imag)],
            dim=-1,
        )

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


def build_geometry_graph(
    height: int = GRID_HEIGHT,
    width: int = GRID_WIDTH,
    k_neighbors: int = 8,
    device: str = "cpu",
) -> GraphStructure:
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    coords = np.stack([rows.reshape(-1), cols.reshape(-1)], axis=1).astype(np.float32)
    x01 = coords[:, 1:2] / max(1, width - 1)
    y01 = coords[:, 0:1] / max(1, height - 1)
    x_center = 2.0 * x01 - 1.0
    y_center = 2.0 * y01 - 1.0
    radius = np.sqrt(x_center**2 + y_center**2)
    node_static = np.concatenate([x01, y01, x_center, y_center, radius], axis=1)

    pairwise = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    edge_pairs: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []
    incoming_edges: list[list[int]] = [[] for _ in range(height * width)]

    for dst in range(height * width):
        order = np.argsort(pairwise[dst])
        neighbors = [src for src in order if src != dst][:k_neighbors]
        neighbors.insert(0, dst)
        for src in neighbors:
            dx = float(x01[src] - x01[dst])
            dy = float(y01[src] - y01[dst])
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 1e-12:
                cos_theta = dx / dist
                sin_theta = dy / dist
            else:
                cos_theta = 0.0
                sin_theta = 0.0
            edge_pairs.append((src, dst))
            edge_features.append([dx, dy, dist, cos_theta, sin_theta])
            incoming_edges[dst].append(len(edge_pairs) - 1)

    edge_index = torch.tensor(edge_pairs, dtype=torch.long, device=device)
    edge_attr = torch.tensor(edge_features, dtype=torch.float32, device=device)
    incoming = tuple(
        torch.tensor(idx_list, dtype=torch.long, device=device) for idx_list in incoming_edges
    )
    node_static_features = torch.tensor(node_static, dtype=torch.float32, device=device)
    return GraphStructure(
        edge_index=edge_index,
        edge_attr=edge_attr,
        incoming_edges=incoming,
        node_static_features=node_static_features,
        num_nodes=height * width,
    )


class GATv2EdgeLayer(nn.Module):
    """Small GATv2-style layer with explicit edge features."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        edge_dim: int,
        edge_hidden: int,
        update_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError("out_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.query = nn.Linear(in_dim, out_dim, bias=False)
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden),
            nn.GELU(),
            nn.Linear(edge_hidden, out_dim),
        )
        self.attn_vector = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, update_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(update_hidden, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, graph: GraphStructure) -> torch.Tensor:
        batch, num_nodes, _ = x.shape
        src = graph.edge_index[:, 0]
        dst = graph.edge_index[:, 1]
        edge_context = self.edge_mlp(graph.edge_attr).view(
            graph.edge_attr.size(0), self.num_heads, self.head_dim
        )

        q = self.query(x).view(batch, num_nodes, self.num_heads, self.head_dim)
        k = self.key(x).view(batch, num_nodes, self.num_heads, self.head_dim)
        v = self.value(x).view(batch, num_nodes, self.num_heads, self.head_dim)

        edge_repr = q[:, dst] + k[:, src] + edge_context.unsqueeze(0)
        edge_repr = F.gelu(edge_repr)
        logits = (
            edge_repr * self.attn_vector.view(1, 1, self.num_heads, self.head_dim)
        ).sum(dim=-1) / math.sqrt(self.head_dim)
        messages = v[:, src] + edge_context.unsqueeze(0)

        aggregated = torch.zeros(
            batch,
            num_nodes,
            self.num_heads,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        for node_idx, incoming in enumerate(graph.incoming_edges):
            attn = torch.softmax(logits[:, incoming, :], dim=1)
            node_messages = messages[:, incoming, :, :]
            aggregated[:, node_idx] = (
                attn.unsqueeze(-1) * node_messages
            ).sum(dim=1)

        aggregated = aggregated.reshape(batch, num_nodes, -1)
        updated = self.out_proj(aggregated)
        x = self.norm1(self.residual(x) + updated)
        return self.norm2(x + self.ffn(x))


class GeometryGraphEncoder(nn.Module):
    def __init__(self, cfg: Config, graph: GraphStructure) -> None:
        super().__init__()
        self.graph = graph
        self.node_feat_dim = graph.node_static_features.size(1) + 2
        layers: list[nn.Module] = []
        in_dim = self.node_feat_dim
        for _ in range(cfg.gnn_layers):
            layers.append(
                GATv2EdgeLayer(
                    in_dim=in_dim,
                    out_dim=cfg.gnn_hidden_dim,
                    num_heads=cfg.gnn_heads,
                    edge_dim=graph.edge_attr.size(1),
                    edge_hidden=cfg.edge_mlp_hidden,
                    update_hidden=cfg.node_update_hidden,
                    dropout=cfg.dropout,
                )
            )
            in_dim = cfg.gnn_hidden_dim
        self.layers = nn.ModuleList(layers)
        self.pool_gate = nn.Linear(cfg.gnn_hidden_dim, 1)
        self.pool_value = nn.Linear(cfg.gnn_hidden_dim, cfg.geometry_embedding_dim)
        self.pool_norm = nn.LayerNorm(cfg.geometry_embedding_dim)

    def build_node_features(self, geom_bits: torch.Tensor) -> torch.Tensor:
        batch = geom_bits.size(0)
        geom_bits = geom_bits.view(batch, self.graph.num_nodes, 1)
        void_bits = 1.0 - geom_bits
        static = self.graph.node_static_features.unsqueeze(0).expand(batch, -1, -1)
        return torch.cat([geom_bits, void_bits, static], dim=-1)

    def forward(self, geom_bits: torch.Tensor) -> torch.Tensor:
        x = self.build_node_features(geom_bits)
        for layer in self.layers:
            x = layer(x, self.graph)
        pool_weights = torch.softmax(self.pool_gate(x).squeeze(-1), dim=1)
        pooled = (pool_weights.unsqueeze(-1) * self.pool_value(x)).sum(dim=1)
        return self.pool_norm(pooled)


class FullSpectrumSpectralBlock(nn.Module):
    """Spectral mixing block without low-frequency truncation."""

    def __init__(self, d_model: int, seq_len: int, ffn_hidden: int, dropout: float) -> None:
        super().__init__()
        self.freq_bins = seq_len // 2 + 1
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.weight_real = nn.Parameter(torch.randn(self.freq_bins, d_model) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(self.freq_bins, d_model) * 0.02)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.norm1(x).transpose(1, 2)
        spectrum = torch.fft.rfft(z, dim=-1)
        weights = torch.complex(self.weight_real, self.weight_imag).transpose(0, 1)
        mixed = spectrum * weights.unsqueeze(0)
        restored = torch.fft.irfft(mixed, n=x.size(1), dim=-1).transpose(1, 2)
        x = residual + restored
        return x + self.ffn(self.norm2(x))


class GeometryConditionedFEDBlock(nn.Module):
    def __init__(self, cfg: Config, seq_len: int) -> None:
        super().__init__()
        self.spectral = FullSpectrumSpectralBlock(
            d_model=cfg.dmodel,
            seq_len=seq_len,
            ffn_hidden=cfg.decoder_ffn_hidden,
            dropout=cfg.dropout,
        )
        self.self_norm = nn.LayerNorm(cfg.dmodel)
        self.cross_norm = nn.LayerNorm(cfg.dmodel)
        self.ff_norm = nn.LayerNorm(cfg.dmodel)
        self.self_attn = nn.MultiheadAttention(
            cfg.dmodel,
            cfg.decoder_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            cfg.dmodel,
            cfg.decoder_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(cfg.dmodel, cfg.decoder_ffn_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.decoder_ffn_hidden, cfg.dmodel),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        geom_token: torch.Tensor,
        freq_tokens: torch.Tensor,
    ) -> torch.Tensor:
        x = self.spectral(freq_tokens)
        self_attn_out, _ = self.self_attn(
            self.self_norm(x),
            self.self_norm(x),
            self.self_norm(x),
            need_weights=False,
        )
        x = x + self_attn_out
        cross_attn_out, _ = self.cross_attn(
            self.cross_norm(x),
            geom_token,
            geom_token,
            need_weights=False,
        )
        x = x + cross_attn_out
        return x + self.ffn(self.ff_norm(x))


class FEDformerDecoder(nn.Module):
    def __init__(self, cfg: Config, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.freq_embedding = nn.Embedding(seq_len, cfg.dmodel)
        self.blocks = nn.ModuleList(
            [GeometryConditionedFEDBlock(cfg, seq_len) for _ in range(cfg.decoder_blocks)]
        )
        refinement_layer = nn.TransformerEncoderLayer(
            d_model=cfg.dmodel,
            nhead=cfg.decoder_heads,
            dim_feedforward=cfg.decoder_ffn_hidden,
            batch_first=True,
            activation="gelu",
            norm_first=True,
            dropout=cfg.dropout,
        )
        self.refinement = nn.TransformerEncoder(
            refinement_layer,
            num_layers=cfg.refinement_layers,
        )
        self.out_norm = nn.LayerNorm(cfg.dmodel)
        self.output_head = nn.Linear(cfg.dmodel, 2)
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(cfg.dmodel * 2),
            nn.Linear(cfg.dmodel * 2, cfg.dmodel),
            nn.GELU(),
        )
        self.f0_head = nn.Linear(cfg.dmodel, 1)
        self.bw_head = nn.Linear(cfg.dmodel, 1)

    def forward(
        self,
        geom_token: torch.Tensor,
        freq_axis_hz: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = geom_token.size(0)
        freq_idx = torch.arange(self.seq_len, device=geom_token.device)
        freq_tokens = self.freq_embedding(freq_idx).unsqueeze(0).expand(batch, -1, -1)
        seq = torch.cat([geom_token, freq_tokens], dim=1)
        geom_context = seq[:, :1, :]
        freq_states = seq[:, 1:, :]

        for block in self.blocks:
            freq_states = block(geom_context, freq_states)

        freq_states = self.refinement(freq_states)
        freq_states = self.out_norm(freq_states)
        gamma = self.output_head(freq_states)

        summary = torch.cat([geom_context.squeeze(1), freq_states.mean(dim=1)], dim=-1)
        summary = self.summary_proj(summary)
        freq_min = float(freq_axis_hz[0].item())
        freq_span = float((freq_axis_hz[-1] - freq_axis_hz[0]).item())
        f0_hz = freq_min + torch.sigmoid(self.f0_head(summary)).squeeze(-1) * freq_span
        bw_hz = torch.sigmoid(self.bw_head(summary)).squeeze(-1) * freq_span

        return {
            "gamma": gamma,
            "f0_hz": f0_hz,
            "bw_hz": bw_hz,
        }


class Geometry2ComplexGamma(nn.Module):
    def __init__(self, cfg: Config, graph: GraphStructure) -> None:
        super().__init__()
        self.encoder = GeometryGraphEncoder(cfg, graph)
        self.geom_token_proj = nn.Sequential(
            nn.LayerNorm(cfg.geometry_embedding_dim),
            nn.Linear(cfg.geometry_embedding_dim, cfg.dmodel),
            nn.GELU(),
        )
        self.decoder = FEDformerDecoder(cfg, cfg.seq_len)

    def forward(
        self,
        geom_bits: torch.Tensor,
        freq_axis_hz: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        geom_embedding = self.encoder(geom_bits)
        geom_token = self.geom_token_proj(geom_embedding).unsqueeze(1)
        outputs = self.decoder(geom_token, freq_axis_hz)
        outputs["geometry_embedding"] = geom_embedding
        return outputs


def frequency_axis(cfg: Config, device: str) -> torch.Tensor:
    return torch.linspace(
        cfg.freq_hz_start,
        cfg.freq_hz_stop,
        cfg.seq_len,
        device=device,
        dtype=torch.float32,
    )


def gamma_magnitude(gamma: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return torch.sqrt(torch.clamp((gamma**2).sum(dim=-1), min=eps))


def gamma_db(gamma: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return 20.0 * torch.log10(torch.clamp(gamma_magnitude(gamma, eps), min=eps))


def resonance_weights(target_gamma: torch.Tensor, sigma_db: float) -> torch.Tensor:
    target_db = gamma_db(target_gamma)
    minima = target_db.min(dim=1, keepdim=True).values
    return 1.0 + 2.0 * torch.exp(-((target_db - minima) / sigma_db) ** 2)


def derive_notch_targets(
    target_gamma: torch.Tensor,
    freq_axis_hz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_db = gamma_db(target_gamma)
    min_indices = target_db.argmin(dim=1)
    f0_hz = freq_axis_hz[min_indices]

    bandwidths: list[torch.Tensor] = []
    for sample_idx in range(target_db.size(0)):
        mask = target_db[sample_idx] <= -10.0
        active = torch.where(mask)[0]
        if active.numel() == 0:
            bandwidths.append(torch.zeros((), device=target_gamma.device))
        else:
            bandwidths.append(freq_axis_hz[active[-1]] - freq_axis_hz[active[0]])
    bw_hz = torch.stack(bandwidths)
    return f0_hz, bw_hz


def smoothness_penalty(pred_gamma: torch.Tensor) -> torch.Tensor:
    if pred_gamma.size(1) < 3:
        return pred_gamma.new_zeros(())
    second_diff = pred_gamma[:, 2:, :] - 2.0 * pred_gamma[:, 1:-1, :] + pred_gamma[:, :-2, :]
    return second_diff.pow(2).mean()


def curriculum_slice(cfg: Config, epoch: int) -> slice:
    if cfg.curriculum_stride > 1 and epoch <= cfg.curriculum_epochs:
        return slice(None, None, cfg.curriculum_stride)
    return slice(None, None, 1)


def physics_informed_loss(
    outputs: dict[str, torch.Tensor],
    target_gamma: torch.Tensor,
    freq_axis_hz: torch.Tensor,
    cfg: Config,
    epoch: int,
) -> LossBreakdown:
    pred_gamma = outputs["gamma"]
    freq_subset = curriculum_slice(cfg, epoch)
    pred_used = pred_gamma[:, freq_subset, :]
    target_used = target_gamma[:, freq_subset, :]

    weights = resonance_weights(target_used, sigma_db=cfg.notch_sigma_db)
    sq_error = (pred_used - target_used).pow(2).sum(dim=-1)
    denom = target_used.pow(2).sum(dim=-1).clamp_min(1e-6)
    complex_mse = (weights * (sq_error / denom)).mean()

    db_error = torch.abs(gamma_db(pred_used) - gamma_db(target_used))
    db_mae = (weights * db_error).mean()

    f0_target, bw_target = derive_notch_targets(target_gamma, freq_axis_hz)
    freq_span = float((freq_axis_hz[-1] - freq_axis_hz[0]).item())
    notch = (
        torch.abs(outputs["f0_hz"] - f0_target).mean()
        + torch.abs(outputs["bw_hz"] - bw_target).mean()
    ) / max(freq_span, 1.0)

    passive = torch.relu(gamma_magnitude(pred_gamma) - cfg.passive_limit).pow(2).mean()
    smooth = smoothness_penalty(pred_gamma)

    total = (
        cfg.loss_complex * complex_mse
        + cfg.loss_db * db_mae
        + cfg.loss_notch * notch
        + cfg.loss_passive * passive
        + cfg.loss_smooth * smooth
    )
    return LossBreakdown(
        total=total,
        complex_mse=complex_mse,
        db_mae=db_mae,
        notch=notch,
        passive=passive,
        smooth=smooth,
    )


def augment_geometry_batch(x: torch.Tensor, cfg: Config) -> torch.Tensor:
    x = x.clone()
    if cfg.bit_flip_prob > 0.0:
        flip_mask = torch.rand_like(x) < cfg.bit_flip_prob
        x = torch.where(flip_mask, 1.0 - x, x)

    if cfg.block_jitter_prob > 0.0 and cfg.max_block_size > 0:
        grid = x.view(-1, GRID_HEIGHT, GRID_WIDTH)
        for batch_idx in range(grid.size(0)):
            if torch.rand(1, device=grid.device).item() >= cfg.block_jitter_prob:
                continue
            block_h = torch.randint(
                1,
                cfg.max_block_size + 1,
                (1,),
                device=grid.device,
            ).item()
            block_w = torch.randint(
                1,
                cfg.max_block_size + 1,
                (1,),
                device=grid.device,
            ).item()
            row = torch.randint(
                0,
                GRID_HEIGHT - block_h + 1,
                (1,),
                device=grid.device,
            ).item()
            col = torch.randint(
                0,
                GRID_WIDTH - block_w + 1,
                (1,),
                device=grid.device,
            ).item()
            patch = grid[batch_idx, row : row + block_h, col : col + block_w]
            grid[batch_idx, row : row + block_h, col : col + block_w] = 1.0 - patch
        x = grid.view(-1, NUM_CELLS)
    return x


def infer_family_key(bits: torch.Tensor) -> tuple[int, ...]:
    grid = bits.view(GRID_HEIGHT, GRID_WIDTH)
    metal = int(grid.sum().item() // 5)
    quadrants = (
        int(grid[:5, :5].sum().item() // 3),
        int(grid[:5, 5:].sum().item() // 3),
        int(grid[5:, :5].sum().item() // 3),
        int(grid[5:, 5:].sum().item() // 3),
    )
    row_bands = tuple(int(grid[r * 2 : (r + 1) * 2, :].sum().item() // 4) for r in range(5))
    col_bands = tuple(int(grid[:, c * 2 : (c + 1) * 2].sum().item() // 4) for c in range(5))
    horizontal_sym = int((grid == torch.flip(grid, dims=[1])).sum().item() // 10)
    vertical_sym = int((grid == torch.flip(grid, dims=[0])).sum().item() // 10)
    return (metal, *quadrants, *row_bands, *col_bands, horizontal_sym, vertical_sym)


def family_aware_split(
    dataset: AntennaDataset,
    cfg: Config,
) -> tuple[Subset, Subset, Subset]:
    families: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        families[infer_family_key(dataset.X[idx])] .append(idx)

    family_keys = list(families.keys())
    rng = random.Random(cfg.random_seed)
    rng.shuffle(family_keys)

    total = len(dataset)
    train_target = int(total * cfg.train_ratio)
    val_target = int(total * cfg.val_ratio)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for key in family_keys:
        members = families[key]
        if len(train_idx) < train_target:
            train_idx.extend(members)
        elif len(val_idx) < val_target:
            val_idx.extend(members)
        else:
            test_idx.extend(members)

    if not val_idx:
        cutoff = max(1, len(train_idx) // 10)
        val_idx = train_idx[-cutoff:]
        train_idx = train_idx[:-cutoff]
    if not test_idx:
        cutoff = max(1, len(val_idx) // 2)
        test_idx = val_idx[-cutoff:]
        val_idx = val_idx[:-cutoff]
    if not test_idx:
        cutoff = max(1, len(train_idx) // 10)
        test_idx = train_idx[-cutoff:]
        train_idx = train_idx[:-cutoff]

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def build_dataloaders(
    dataset: AntennaDataset,
    cfg: Config,
    device: str,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = family_aware_split(dataset, cfg)
    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def create_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int) -> LambdaLR:
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr_scale + (1.0 - cfg.min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def sync_config_from_dataset(cfg: Config, dataset: AntennaDataset) -> None:
    cfg.seq_len = dataset.seq_len
    cfg.output_mode = dataset.output_mode


def inspect_dataset(dataset: AntennaDataset) -> None:
    x0, y0 = dataset[0]
    print(f"Dataset: {dataset.csv_path}")
    print(f"Samples: {len(dataset)}")
    print(f"Detected layout: {dataset.layout.name}")
    print(f"Input sample shape: {tuple(x0.shape)}")
    print(f"Target sample shape: {tuple(y0.shape)}")
    print(f"First 10 geometry bits: {x0[:10].tolist()}")
    print(f"First 3 target rows: {y0[:3].tolist()}")


def build_model(cfg: Config, device: str) -> Geometry2ComplexGamma:
    graph = build_geometry_graph(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        k_neighbors=cfg.graph_k,
        device=device,
    )
    return Geometry2ComplexGamma(cfg, graph).to(device)


def inspect_model(cfg: Config, device: str) -> None:
    model = build_model(cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy_x = torch.randint(0, 2, (2, cfg.input_dim), dtype=torch.float32, device=device)
    freq_axis_hz = frequency_axis(cfg, device)
    with torch.no_grad():
        outputs = model(dummy_x, freq_axis_hz)
    print(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Dummy input shape: {tuple(dummy_x.shape)}")
    print(f"Dummy gamma shape: {tuple(outputs['gamma'].shape)}")


def save_config(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    for key, value in asdict(cfg).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    with cfg.output_dir.joinpath(cfg.config_filename).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_summary(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_checkpoint(
    cfg: Config,
    model: Geometry2ComplexGamma,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_val_total": best_val,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in asdict(cfg).items()
            },
        },
        cfg.checkpoint_path,
    )
    torch.save(model.state_dict(), cfg.weights_path)


def load_model_weights(
    model: Geometry2ComplexGamma,
    checkpoint_path: Path,
    device: str,
) -> dict[str, object]:
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
        return payload
    model.load_state_dict(payload)
    return {}


def evaluate_model(
    model: Geometry2ComplexGamma,
    loader: DataLoader,
    cfg: Config,
    device: str,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    freq_axis_hz = frequency_axis(cfg, device)
    totals = {
        "total": 0.0,
        "complex_mse": 0.0,
        "db_mae": 0.0,
        "notch": 0.0,
        "passive": 0.0,
        "smooth": 0.0,
    }
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb, freq_axis_hz)
            losses = physics_informed_loss(outputs, yb, freq_axis_hz, cfg, epoch)
            batch_size = xb.size(0)
            totals["total"] += losses.total.item() * batch_size
            totals["complex_mse"] += losses.complex_mse.item() * batch_size
            totals["db_mae"] += losses.db_mae.item() * batch_size
            totals["notch"] += losses.notch.item() * batch_size
            totals["passive"] += losses.passive.item() * batch_size
            totals["smooth"] += losses.smooth.item() * batch_size
            count += batch_size

    return {key: value / max(1, count) for key, value in totals.items()}


def train_model(cfg: Config) -> tuple[dict[str, float], list[dict[str, float]]]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg)

    device = get_device(cfg.device)
    dataset = AntennaDataset(cfg.csv_path, input_dim=cfg.input_dim, output_mode=cfg.output_mode)
    sync_config_from_dataset(cfg, dataset)
    train_loader, val_loader, test_loader = build_dataloaders(dataset, cfg, device)
    model = build_model(cfg, device)
    optimizer = create_optimizer(cfg, model.parameters())
    scheduler = create_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and cfg.use_amp))
    freq_axis_hz = frequency_axis(cfg, device)

    history_rows: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = 0
    patience_left = cfg.patience

    with cfg.history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_total",
                "train_complex_mse",
                "train_db_mae",
                "train_notch",
                "train_passive",
                "train_smooth",
                "val_total",
                "val_complex_mse",
                "val_db_mae",
                "val_notch",
                "val_passive",
                "val_smooth",
                "lr",
            ]
        )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_totals = {
            "total": 0.0,
            "complex_mse": 0.0,
            "db_mae": 0.0,
            "notch": 0.0,
            "passive": 0.0,
            "smooth": 0.0,
        }
        count = 0
        amp_enabled = device == "cuda" and cfg.use_amp

        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            if cfg.use_augmentation:
                xb = augment_geometry_batch(xb, cfg)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(xb, freq_axis_hz)
                losses = physics_informed_loss(outputs, yb, freq_axis_hz, cfg, epoch)

            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_size = xb.size(0)
            train_totals["total"] += losses.total.item() * batch_size
            train_totals["complex_mse"] += losses.complex_mse.item() * batch_size
            train_totals["db_mae"] += losses.db_mae.item() * batch_size
            train_totals["notch"] += losses.notch.item() * batch_size
            train_totals["passive"] += losses.passive.item() * batch_size
            train_totals["smooth"] += losses.smooth.item() * batch_size
            count += batch_size

            if cfg.log_every_batches > 0 and batch_idx % cfg.log_every_batches == 0:
                running = {key: value / max(1, count) for key, value in train_totals.items()}
                print(
                    "Epoch "
                    f"{epoch:03d} | "
                    f"batch {batch_idx:04d}/{len(train_loader):04d} | "
                    f"loss {running['total']:.6f} | "
                    f"cmse {running['complex_mse']:.6f} | "
                    f"db {running['db_mae']:.6f} | "
                    f"lr {optimizer.param_groups[0]['lr']:.6e}"
                )

        train_metrics = {key: value / max(1, count) for key, value in train_totals.items()}
        val_metrics = evaluate_model(model, val_loader, cfg, device, epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_total": train_metrics["total"],
            "train_complex_mse": train_metrics["complex_mse"],
            "train_db_mae": train_metrics["db_mae"],
            "train_notch": train_metrics["notch"],
            "train_passive": train_metrics["passive"],
            "train_smooth": train_metrics["smooth"],
            "val_total": val_metrics["total"],
            "val_complex_mse": val_metrics["complex_mse"],
            "val_db_mae": val_metrics["db_mae"],
            "val_notch": val_metrics["notch"],
            "val_passive": val_metrics["passive"],
            "val_smooth": val_metrics["smooth"],
            "lr": current_lr,
        }
        history_rows.append(row)

        with cfg.history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch,
                    row["train_total"],
                    row["train_complex_mse"],
                    row["train_db_mae"],
                    row["train_notch"],
                    row["train_passive"],
                    row["train_smooth"],
                    row["val_total"],
                    row["val_complex_mse"],
                    row["val_db_mae"],
                    row["val_notch"],
                    row["val_passive"],
                    row["val_smooth"],
                    current_lr,
                ]
            )

        print(
            "Epoch "
            f"{epoch:03d} | "
            f"train {row['train_total']:.6f} | "
            f"val {row['val_total']:.6f} | "
            f"cmse {row['val_complex_mse']:.6f} | "
            f"db {row['val_db_mae']:.6f}"
        )

        if row["val_total"] < best_val - 1e-6:
            best_val = row["val_total"]
            best_epoch = epoch
            save_checkpoint(cfg, model, optimizer, epoch, best_val)
            patience_left = cfg.patience
            print(f"  saved checkpoint: {cfg.checkpoint_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    best_model = build_model(cfg, device)
    load_model_weights(best_model, cfg.checkpoint_path, device)
    test_metrics = evaluate_model(best_model, test_loader, cfg, device, best_epoch)
    summary = {
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "test_total": test_metrics["total"],
        "test_complex_mse": test_metrics["complex_mse"],
        "test_db_mae": test_metrics["db_mae"],
        "test_notch": test_metrics["notch"],
        "test_passive": test_metrics["passive"],
        "test_smooth": test_metrics["smooth"],
        "layout": dataset.layout.name,
        "seq_len": dataset.seq_len,
    }
    write_summary(cfg.summary_path, summary)
    return summary, history_rows


def run_saved_evaluation(cfg: Config, split: str) -> dict[str, float]:
    device = get_device(cfg.device)
    dataset = AntennaDataset(cfg.csv_path, input_dim=cfg.input_dim, output_mode=cfg.output_mode)
    sync_config_from_dataset(cfg, dataset)
    train_loader, val_loader, test_loader = build_dataloaders(dataset, cfg, device)
    loader = val_loader if split == "val" else test_loader
    model = build_model(cfg, device)
    load_model_weights(model, cfg.checkpoint_path, device)
    metrics = evaluate_model(model, loader, cfg, device, cfg.epochs)
    metrics["split"] = split
    metrics["checkpoint"] = str(cfg.checkpoint_path)
    return metrics


def plot_loss_curve(history_rows: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history_rows]
    train_loss = [row["train_total"] for row in history_rows]
    val_loss = [row["val_total"] for row in history_rows]
    plt.figure()
    plt.plot(epochs, train_loss, label="Train total loss")
    plt.plot(epochs, val_loss, label="Validation total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("R5 Physics-Informed Training Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Train full R5 physics-informed surrogate.")
    parser.add_argument("--csv-path", type=Path, default=cfg.csv_path)
    parser.add_argument(
        "--output-mode",
        choices=["complex_122", "complex_61", "mag_only", "auto"],
        default=cfg.output_mode,
    )
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--optimizer", choices=["adamw", "adagrad"], default=cfg.optimizer_name)
    parser.add_argument("--device", type=str, default=cfg.device)
    parser.add_argument("--output-dir", type=Path, default=cfg.output_dir)
    parser.add_argument("--seed", type=int, default=cfg.random_seed)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--graph-k", type=int, default=cfg.graph_k)
    parser.add_argument("--log-every-batches", type=int, default=cfg.log_every_batches)
    parser.add_argument("--curriculum-off", action="store_true")
    parser.add_argument("--seq-len", type=int, default=cfg.seq_len)
    parser.add_argument("--eval-split", choices=["val", "test"])
    args = parser.parse_args()

    cfg.csv_path = args.csv_path
    cfg.output_mode = args.output_mode
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.optimizer_name = args.optimizer
    cfg.device = args.device
    cfg.output_dir = args.output_dir
    cfg.random_seed = args.seed
    cfg.use_augmentation = not args.no_augmentation
    cfg.graph_k = args.graph_k
    cfg.log_every_batches = args.log_every_batches
    cfg.seq_len = args.seq_len
    cfg.eval_split = args.eval_split or ""
    if args.curriculum_off:
        cfg.curriculum_epochs = 0
        cfg.curriculum_stride = 1
    return cfg


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.random_seed)

    eval_split = getattr(cfg, "eval_split", None)
    if eval_split:
        metrics = run_saved_evaluation(cfg, eval_split)
        print(f"Evaluation split: {eval_split}")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        return

    print("Configuration:")
    for key, value in asdict(cfg).items():
        print(f"  {key}: {value}")

    dataset = AntennaDataset(cfg.csv_path, input_dim=cfg.input_dim, output_mode=cfg.output_mode)
    sync_config_from_dataset(cfg, dataset)
    inspect_dataset(dataset)

    device = get_device(cfg.device)
    print(f"Using device: {device}")
    inspect_model(cfg, device)

    summary, history_rows = train_model(cfg)
    plot_loss_curve(history_rows, cfg.loss_plot_path)
    print(f"Loss curve saved to: {cfg.loss_plot_path}")
    print("Training summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
