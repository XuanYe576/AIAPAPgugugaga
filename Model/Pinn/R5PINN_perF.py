"""
1. train a per-frequency physics-informed model
   `geometry + frequency -> dB(S11)`
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    TORCH_AVAILABLE = False

    class _Module:
        pass

    class _Dataset:
        pass

    class _DataLoader:
        pass

    class _NNNamespace:
        Module = _Module

    class _FFallback:
        def __getattr__(self, name: str) -> object:
            raise ModuleNotFoundError("torch is required for training or inspection.")

    nn = _NNNamespace()
    F = _FFallback()
    DataLoader = _DataLoader
    Dataset = _Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GRID_HEIGHT = 10
GRID_WIDTH = 10
NUM_CELLS = GRID_HEIGHT * GRID_WIDTH


def _default_raw_data_root() -> Path:
    candidates = [
        REPO_ROOT / "Data" / "NotprocessedData",
        REPO_ROOT / "Data" / "Patch Antennas 6,001-30,000-selected",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


RAW_DATA_ROOT = _default_raw_data_root()

from metrics.prediction_graphs import save_scalar_prediction_graphs
from utils.dataP import preprocess_uploaded_dataset as build_processed_dataset


def _default_geometry_catalog_path() -> Path:
    candidates = [
        RAW_DATA_ROOT / "60000 Patch Antenna File" / "patch_antennas_updated5b.csv",
        RAW_DATA_ROOT / "30000 Patch Antenna File" / "patch_antennas_updated4.csv",
        RAW_DATA_ROOT / "19001-20000" / "patch_antennas_updated3.csv",
        RAW_DATA_ROOT / "15000 Patch Antenna File" / "patch_antennas_updated2.csv",
        REPO_ROOT / "patch_antennas_updated.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _default_processed_dir() -> Path:
    return REPO_ROOT / "Data" / "processed"


def _default_processed_csv() -> Path:
    return _default_processed_dir() / "Full_60000Data_61dB.csv"


def _default_processed_meta() -> Path:
    return _default_processed_dir() / "Full_60000Data_61dB.meta.json"


def _default_results_dir() -> Path:
    return REPO_ROOT / "results" / "R5PINN_perF"


@dataclass
class Config:
    command: str = "train"
    geometry_catalog_path: Path = field(default_factory=_default_geometry_catalog_path)
    curves_root: Path = RAW_DATA_ROOT
    processed_csv_path: Path = field(default_factory=_default_processed_csv)
    processed_meta_path: Path = field(default_factory=_default_processed_meta)
    results_dir: Path = field(default_factory=_default_results_dir)
    input_dim: int = NUM_CELLS
    batch_size: int = 256
    lr: float = 2e-3
    optimizer_name: str = "adamw"
    weight_decay: float = 1e-4
    epochs: int = 80
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    gnn_hidden_dim: int = 128
    gnn_layers: int = 3
    gnn_heads: int = 4
    edge_mlp_hidden: int = 64
    node_update_hidden: int = 128
    geometry_embedding_dim: int = 256
    freq_hidden_dim: int = 96
    fusion_hidden_dim: int = 256
    graph_k: int = 8
    dropout: float = 0.10
    loss_passive: float = 0.10
    resonance_loss_weight: float = 0.10
    notch_sigma_db: float = 4.0
    patience: int = 12
    gradient_clip: float = 1.0
    log_every_batches: int = 10
    use_amp: bool = True
    random_seed: int = 42
    max_antennas: int = 0
    device: str = "auto"
    eval_split: str = ""
    prediction_plot_count: int = 12
    overwrite_processed: bool = False
    weights_filename: str = "r5pinn_perF_best.pt"
    checkpoint_filename: str = "r5pinn_perF_best.ckpt"
    history_filename: str = "history.csv"
    loss_per_freq_filename: str = "loss_per_freq.csv"
    summary_filename: str = "summary.json"
    config_filename: str = "config.json"

    @property
    def weights_path(self) -> Path:
        return self.results_dir / self.weights_filename

    @property
    def history_path(self) -> Path:
        return self.results_dir / self.history_filename

    @property
    def loss_per_freq_path(self) -> Path:
        return self.results_dir / self.loss_per_freq_filename

    @property
    def summary_path(self) -> Path:
        return self.results_dir / self.summary_filename

    @property
    def config_path(self) -> Path:
        return self.results_dir / self.config_filename

    @property
    def checkpoint_path(self) -> Path:
        return self.results_dir / self.checkpoint_filename

    @property
    def prediction_graphical_dir(self) -> Path:
        return self.results_dir / "weight" / "graphical"


@dataclass(frozen=True)
class GraphStructure:
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    incoming_edges: tuple[torch.Tensor, ...]
    node_static_features: torch.Tensor
    num_nodes: int


@dataclass
class PointLossBreakdown:
    total: torch.Tensor
    data: torch.Tensor
    passive: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> str:
    if not TORCH_AVAILABLE:
        return "cpu"
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def require_torch(command_name: str) -> None:
    if not TORCH_AVAILABLE:
        raise ModuleNotFoundError(
            f"`torch` is required for `{command_name}`. "
            "Preprocess can run without torch, but train/inspect cannot."
        )


def create_optimizer(cfg: Config, parameters: object) -> object:
    require_torch("optimizer creation")
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


def save_config(cfg: Config) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    for key, value in asdict(cfg).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    with cfg.config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_checkpoint(
    cfg: Config,
    model: nn.Module,
    optimizer: object,
    epoch: int,
    best_selection_total: float,
) -> None:
    require_torch("checkpoint save")
    torch.save(
        {
            "epoch": epoch,
            "best_selection_total": best_selection_total,
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


def load_model_weights(model: nn.Module, checkpoint_path: Path, device: str) -> dict[str, object]:
    require_torch("checkpoint load")
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
        return payload
    model.load_state_dict(payload)
    return {}


def preprocess_uploaded_dataset(cfg: Config) -> dict[str, object]:
    return build_processed_dataset(
        geometry_catalog_path=cfg.geometry_catalog_path,
        curves_root=cfg.curves_root,
        processed_csv_path=cfg.processed_csv_path,
        processed_meta_path=cfg.processed_meta_path,
        overwrite_processed=cfg.overwrite_processed,
        max_antennas=cfg.max_antennas,
        grid_height=GRID_HEIGHT,
        grid_width=GRID_WIDTH,
    )


class ProcessedCurveDataset:
    def __init__(self, csv_path: Path, meta_path: Path, sigma_db: float) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Processed CSV not found: {csv_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Processed metadata not found: {meta_path}")

        with meta_path.open(encoding="utf-8") as handle:
            meta = json.load(handle)
        values = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[1] < NUM_CELLS + 1:
            raise ValueError(
                f"Processed CSV must have at least {NUM_CELLS + 1} columns, "
                f"found {values.shape[1]}."
            )

        self.csv_path = csv_path
        self.meta_path = meta_path
        self.meta = meta
        self.seq_len = int(meta["seq_len"])
        if values.shape[1] != NUM_CELLS + self.seq_len:
            raise ValueError(
                f"Processed CSV column count mismatch. Expected {NUM_CELLS + self.seq_len}, "
                f"found {values.shape[1]}."
            )

        self.geometry = torch.from_numpy((values[:, :NUM_CELLS] > 0.5).astype(np.float32))
        self.curves_db = torch.from_numpy(values[:, NUM_CELLS:])
        self.freq_axis_ghz = torch.tensor(meta["freq_axis_ghz"], dtype=torch.float32)
        freq_min = float(self.freq_axis_ghz.min().item())
        freq_max = float(self.freq_axis_ghz.max().item())
        denom = max(freq_max - freq_min, 1e-6)
        self.freq_axis_norm = (self.freq_axis_ghz - freq_min) / denom
        self.sample_weights = resonance_weights_db(self.curves_db, sigma_db=sigma_db)
        self.antenna_ids = meta.get(
            "matched_antenna_ids",
            list(range(1, self.geometry.size(0) + 1)),
        )

    def __len__(self) -> int:
        return self.geometry.size(0)


def resonance_weights_db(curves_db: torch.Tensor, sigma_db: float) -> torch.Tensor:
    minima = curves_db.min(dim=1, keepdim=True).values
    return 1.0 + 2.0 * torch.exp(-((curves_db - minima) / sigma_db) ** 2)


def extract_patch_length(bits: torch.Tensor) -> torch.Tensor:
    cell_size_m = 15.0e-3
    grid = bits.view(-1, GRID_HEIGHT, GRID_WIDTH)
    metal_mask = grid > 0.5

    row_presence = metal_mask.any(dim=2)
    col_presence = metal_mask.any(dim=1)
    row_indices = torch.arange(GRID_HEIGHT, device=grid.device).view(1, GRID_HEIGHT)
    col_indices = torch.arange(GRID_WIDTH, device=grid.device).view(1, GRID_WIDTH)

    r_min = torch.where(
        row_presence,
        row_indices,
        torch.full_like(row_indices, GRID_HEIGHT),
    ).min(dim=1).values
    r_max = torch.where(row_presence, row_indices, torch.zeros_like(row_indices)).max(dim=1).values
    c_min = torch.where(
        col_presence,
        col_indices,
        torch.full_like(col_indices, GRID_WIDTH),
    ).min(dim=1).values
    c_max = torch.where(col_presence, col_indices, torch.zeros_like(col_indices)).max(dim=1).values

    has_metal = metal_mask.flatten(1).any(dim=1)
    lx_cells = c_max - c_min + 1
    ly_cells = r_max - r_min + 1
    resonant_cells = torch.maximum(lx_cells, ly_cells)
    resonant_cells = torch.where(has_metal, resonant_cells, torch.ones_like(resonant_cells))
    return resonant_cells.to(dtype=torch.float32) * cell_size_m


def theoretical_resonant_freq(L_m: torch.Tensor) -> torch.Tensor:
    c_m_per_s = 3.0e8
    h_m = 1.57e-3
    er = 4.4
    L_m = L_m.to(dtype=torch.float32).clamp_min(1.0e-6)
    u = 1.0 + 12.0 * h_m / L_m
    ee = (er + 1.0) / 2.0 + (er - 1.0) / (2.0 * torch.sqrt(u))
    return c_m_per_s / (2.0 * L_m * torch.sqrt(ee)) / 1.0e9


def resonance_frequency_loss(
    geom_bits: torch.Tensor,
    pred_curves_db: torch.Tensor,
    freq_axis_ghz: torch.Tensor,
) -> torch.Tensor:
    curve_mask = torch.isfinite(pred_curves_db).all(dim=1)
    if not curve_mask.any():
        return pred_curves_db.new_zeros(())

    pred_curves_db = pred_curves_db[curve_mask]
    geom_bits = geom_bits[curve_mask]
    pred_notch_idx = pred_curves_db.argmin(dim=1)
    pred_notch_ghz = freq_axis_ghz[pred_notch_idx]
    theory_notch_ghz = theoretical_resonant_freq(extract_patch_length(geom_bits))
    finite = torch.isfinite(pred_notch_ghz) & torch.isfinite(theory_notch_ghz)
    if not finite.any():
        return pred_curves_db.new_zeros(())
    return (pred_notch_ghz[finite] - theory_notch_ghz[finite]).pow(2).mean()


class PerFrequencyDataset(Dataset):
    def __init__(self, base: ProcessedCurveDataset, antenna_indices: list[int]) -> None:
        self.base = base
        self.antenna_indices = antenna_indices
        self.seq_len = base.seq_len

    def __len__(self) -> int:
        return len(self.antenna_indices) * self.seq_len

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        antenna_slot, freq_idx = divmod(idx, self.seq_len)
        antenna_idx = self.antenna_indices[antenna_slot]
        geom_bits = self.base.geometry[antenna_idx]
        freq_norm = self.base.freq_axis_norm[freq_idx].view(1)
        target_db = self.base.curves_db[antenna_idx, freq_idx]
        weight = self.base.sample_weights[antenna_idx, freq_idx]
        return geom_bits, freq_norm, target_db, weight, freq_idx, antenna_slot


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
            dx = float(x01[src, 0] - x01[dst, 0])
            dy = float(y01[src, 0] - y01[dst, 0])
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


class FrequencyFeatureEncoder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, cfg.freq_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.freq_hidden_dim, cfg.freq_hidden_dim),
            nn.GELU(),
        )

    def forward(self, freq_norm: torch.Tensor) -> torch.Tensor:
        f = freq_norm.squeeze(-1)
        feats = torch.stack(
            [
                f,
                f * f,
                torch.sin(math.pi * f),
                torch.cos(math.pi * f),
                torch.sin(2.0 * math.pi * f),
                torch.cos(2.0 * math.pi * f),
                torch.sin(4.0 * math.pi * f),
            ],
            dim=-1,
        )
        return self.net(feats)


class R5PINNPerFrequency(nn.Module):
    def __init__(self, cfg: Config, graph: GraphStructure) -> None:
        super().__init__()
        self.encoder = GeometryGraphEncoder(cfg, graph)
        self.freq_encoder = FrequencyFeatureEncoder(cfg)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.geometry_embedding_dim + cfg.freq_hidden_dim),
            nn.Linear(cfg.geometry_embedding_dim + cfg.freq_hidden_dim, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden_dim, cfg.fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden_dim // 2, 1),
        )

    def forward(self, geom_bits: torch.Tensor, freq_norm: torch.Tensor) -> torch.Tensor:
        geom_embedding = self.encoder(geom_bits)
        freq_embedding = self.freq_encoder(freq_norm)
        fused = torch.cat([geom_embedding, freq_embedding], dim=-1)
        return self.head(fused).squeeze(-1)


def pointwise_physics_terms(
    pred_db: torch.Tensor,
    target_db: torch.Tensor,
    weights: torch.Tensor,
    cfg: Config,
) -> PointLossBreakdown:
    data = weights * F.smooth_l1_loss(pred_db, target_db, reduction="none")
    s11_lin = torch.pow(pred_db.new_tensor(10.0), pred_db / 20.0)
    passive = F.relu(s11_lin - 1.0).pow(2)
    total = data + cfg.loss_passive * passive
    return PointLossBreakdown(total=total, data=data, passive=passive)


def pointwise_physics_loss(
    pred_db: torch.Tensor,
    target_db: torch.Tensor,
    weights: torch.Tensor,
    cfg: Config,
) -> PointLossBreakdown:
    pointwise = pointwise_physics_terms(pred_db, target_db, weights, cfg)
    return PointLossBreakdown(
        total=pointwise.total.mean(),
        data=pointwise.data.mean(),
        passive=pointwise.passive.mean(),
    )


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


def family_aware_split_indices(
    base: ProcessedCurveDataset,
    cfg: Config,
) -> tuple[list[int], list[int], list[int]]:
    families: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for idx in range(len(base)):
        families[infer_family_key(base.geometry[idx])].append(idx)

    family_keys = list(families.keys())
    rng = random.Random(cfg.random_seed)
    rng.shuffle(family_keys)

    total = len(base)
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
    return train_idx, val_idx, test_idx


def build_dataloaders(
    base: ProcessedCurveDataset,
    cfg: Config,
    device: str,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_idx, val_idx, test_idx = family_aware_split_indices(base, cfg)
    pin_memory = device == "cuda"
    train_loader = DataLoader(
        PerFrequencyDataset(base, train_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        PerFrequencyDataset(base, val_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        PerFrequencyDataset(base, test_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def build_model(cfg: Config, device: str) -> R5PINNPerFrequency:
    graph = build_geometry_graph(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        k_neighbors=cfg.graph_k,
        device=device,
    )
    return R5PINNPerFrequency(cfg, graph).to(device)


def evaluate_model(
    model: R5PINNPerFrequency,
    loader: DataLoader,
    cfg: Config,
    device: str,
) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "data": 0.0, "passive": 0.0, "mae": 0.0}
    count = 0
    dataset = loader.dataset
    pred_curves = None
    geom_subset = None
    freq_axis_ghz = None
    if isinstance(dataset, PerFrequencyDataset):
        pred_curves = torch.full(
            (len(dataset.antenna_indices), dataset.seq_len),
            float("nan"),
            dtype=torch.float32,
        )
        geom_subset = dataset.base.geometry[dataset.antenna_indices].detach().cpu()
        freq_axis_ghz = dataset.base.freq_axis_ghz.detach().cpu()
    with torch.no_grad():
        for geom_bits, freq_norm, target_db, weights, freq_idx, antenna_slot in loader:
            geom_bits = geom_bits.to(device)
            freq_norm = freq_norm.to(device)
            target_db = target_db.to(device)
            weights = weights.to(device)
            pred_db = model(geom_bits, freq_norm)
            losses = pointwise_physics_loss(pred_db, target_db, weights, cfg)
            batch_size = geom_bits.size(0)
            totals["total"] += losses.total.item() * batch_size
            totals["data"] += losses.data.item() * batch_size
            totals["passive"] += losses.passive.item() * batch_size
            totals["mae"] += torch.abs(pred_db - target_db).mean().item() * batch_size
            count += batch_size

            if pred_curves is not None:
                pred_curves[
                    antenna_slot.to(dtype=torch.long),
                    freq_idx.to(dtype=torch.long),
                ] = pred_db.detach().cpu()

    metrics = {key: value / max(1, count) for key, value in totals.items()}
    if pred_curves is not None and geom_subset is not None and freq_axis_ghz is not None:
        metrics["resonance_loss"] = float(
            resonance_frequency_loss(geom_subset, pred_curves, freq_axis_ghz).item()
        )
    else:
        metrics["resonance_loss"] = 0.0
    metrics["selection_total"] = (
        metrics["total"] + cfg.resonance_loss_weight * metrics["resonance_loss"]
    )
    return metrics


def train_model(cfg: Config) -> dict[str, float]:
    require_torch("train")
    if (
        cfg.overwrite_processed
        or not cfg.processed_csv_path.exists()
        or not cfg.processed_meta_path.exists()
    ):
        preprocess_uploaded_dataset(cfg)

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg)

    device = get_device(cfg.device)
    base = ProcessedCurveDataset(
        cfg.processed_csv_path,
        cfg.processed_meta_path,
        sigma_db=cfg.notch_sigma_db,
    )
    train_loader, val_loader, test_loader = build_dataloaders(base, cfg, device)
    model = build_model(cfg, device)
    optimizer = create_optimizer(cfg, model.parameters())
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and cfg.use_amp))

    history_rows: list[dict[str, float]] = []
    best_selection_total = float("inf")
    best_val_total = float("inf")
    best_val_resonance = float("inf")
    best_epoch = 0
    patience_left = cfg.patience

    with cfg.history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_total",
                "train_data",
                "train_passive",
                "val_total",
                "val_data",
                "val_passive",
                "val_mae",
                "val_resonance_loss",
                "val_selection_total",
            ]
        )
    with cfg.loss_per_freq_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                *[
                    f"freq_{float(freq_ghz):.6f}_ghz"
                    for freq_ghz in base.freq_axis_ghz.tolist()
                ],
            ]
        )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_totals = {"total": 0.0, "data": 0.0, "passive": 0.0}
        per_freq_loss_sums = torch.zeros(base.seq_len, dtype=torch.float64)
        per_freq_loss_counts = torch.zeros(base.seq_len, dtype=torch.long)
        count = 0
        amp_enabled = device == "cuda" and cfg.use_amp

        for batch_idx, (geom_bits, freq_norm, target_db, weights, freq_idx, _antenna_slot) in enumerate(
            train_loader,
            start=1,
        ):
            geom_bits = geom_bits.to(device)
            freq_norm = freq_norm.to(device)
            target_db = target_db.to(device)
            weights = weights.to(device)
            freq_idx = freq_idx.to(dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred_db = model(geom_bits, freq_norm)
                pointwise = pointwise_physics_terms(pred_db, target_db, weights, cfg)
                losses = PointLossBreakdown(
                    total=pointwise.total.mean(),
                    data=pointwise.data.mean(),
                    passive=pointwise.passive.mean(),
                )

            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_size = geom_bits.size(0)
            train_totals["total"] += losses.total.item() * batch_size
            train_totals["data"] += losses.data.item() * batch_size
            train_totals["passive"] += losses.passive.item() * batch_size
            per_freq_loss_sums.index_add_(
                0,
                freq_idx,
                pointwise.total.detach().to(device="cpu", dtype=torch.float64),
            )
            per_freq_loss_counts.index_add_(
                0,
                freq_idx,
                torch.ones_like(freq_idx, dtype=torch.long),
            )
            count += batch_size

            if cfg.log_every_batches > 0 and batch_idx % cfg.log_every_batches == 0:
                running = {key: value / max(1, count) for key, value in train_totals.items()}
                print(
                    "Epoch "
                    f"{epoch:03d} | "
                    f"batch {batch_idx:04d}/{len(train_loader):04d} | "
                    f"loss {running['total']:.6f} | "
                    f"data {running['data']:.6f} | "
                    f"passive {running['passive']:.6f} | "
                    f"lr {optimizer.param_groups[0]['lr']:.6e}"
                )

        train_metrics = {key: value / max(1, count) for key, value in train_totals.items()}
        val_metrics = evaluate_model(model, val_loader, cfg, device)
        counts_np = per_freq_loss_counts.numpy()
        sums_np = per_freq_loss_sums.numpy()
        avg_loss_per_freq = np.full(base.seq_len, np.nan, dtype=np.float64)
        observed = counts_np > 0
        avg_loss_per_freq[observed] = sums_np[observed] / counts_np[observed]
        row = {
            "epoch": epoch,
            "train_total": train_metrics["total"],
            "train_data": train_metrics["data"],
            "train_passive": train_metrics["passive"],
            "val_total": val_metrics["total"],
            "val_data": val_metrics["data"],
            "val_passive": val_metrics["passive"],
            "val_mae": val_metrics["mae"],
            "val_resonance_loss": val_metrics["resonance_loss"],
            "val_selection_total": val_metrics["selection_total"],
        }
        history_rows.append(row)

        with cfg.history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    row["epoch"],
                    row["train_total"],
                    row["train_data"],
                    row["train_passive"],
                    row["val_total"],
                    row["val_data"],
                    row["val_passive"],
                    row["val_mae"],
                    row["val_resonance_loss"],
                    row["val_selection_total"],
                ]
            )
        with cfg.loss_per_freq_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([epoch, *avg_loss_per_freq.tolist()])

        print(
            "Epoch "
            f"{epoch:03d} | "
            f"train {row['train_total']:.6f} | "
            f"val {row['val_total']:.6f} | "
            f"res {row['val_resonance_loss']:.6f} | "
            f"select {row['val_selection_total']:.6f} | "
            f"mae {row['val_mae']:.6f}"
        )

        if row["val_selection_total"] < best_selection_total - 1e-6:
            best_selection_total = row["val_selection_total"]
            best_val_total = row["val_total"]
            best_val_resonance = row["val_resonance_loss"]
            best_epoch = epoch
            save_checkpoint(cfg, model, optimizer, epoch, best_selection_total)
            patience_left = cfg.patience
            print(f"  saved checkpoint: {cfg.checkpoint_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    best_model = build_model(cfg, device)
    load_model_weights(best_model, cfg.checkpoint_path, device)
    test_metrics = evaluate_model(best_model, test_loader, cfg, device)
    _, _, test_idx = family_aware_split_indices(base, cfg)
    graphical_dir = save_scalar_prediction_graphs(
        model=best_model,
        base=base,
        antenna_indices=test_idx,
        output_dir=cfg.prediction_graphical_dir / "test",
        split="test",
        plot_count=cfg.prediction_plot_count,
        device=device,
    )
    summary = {
        "best_epoch": best_epoch,
        "best_val_total": best_val_total,
        "best_val_resonance_loss": best_val_resonance,
        "best_val_selection_total": best_selection_total,
        "test_total": test_metrics["total"],
        "test_data": test_metrics["data"],
        "test_passive": test_metrics["passive"],
        "test_mae": test_metrics["mae"],
        "test_resonance_loss": test_metrics["resonance_loss"],
        "test_selection_total": test_metrics["selection_total"],
        "resonance_loss_weight": cfg.resonance_loss_weight,
        "num_antennas": len(base),
        "seq_len": base.seq_len,
        "prediction_graphical_dir": str(graphical_dir),
        "prediction_graph_count": min(cfg.prediction_plot_count, len(test_idx)),
    }
    with cfg.summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def run_saved_evaluation(cfg: Config, split: str) -> dict[str, float]:
    require_torch("evaluation")
    if (
        cfg.overwrite_processed
        or not cfg.processed_csv_path.exists()
        or not cfg.processed_meta_path.exists()
    ):
        preprocess_uploaded_dataset(cfg)

    device = get_device(cfg.device)
    base = ProcessedCurveDataset(
        cfg.processed_csv_path,
        cfg.processed_meta_path,
        sigma_db=cfg.notch_sigma_db,
    )
    _train_idx, val_idx, test_idx = family_aware_split_indices(base, cfg)
    train_loader, val_loader, test_loader = build_dataloaders(base, cfg, device)
    loader = val_loader if split == "val" else test_loader
    antenna_indices = val_idx if split == "val" else test_idx
    model = build_model(cfg, device)
    load_model_weights(model, cfg.checkpoint_path, device)
    metrics = evaluate_model(model, loader, cfg, device)
    graphical_dir = save_scalar_prediction_graphs(
        model=model,
        base=base,
        antenna_indices=antenna_indices,
        output_dir=cfg.prediction_graphical_dir / split,
        split=split,
        plot_count=cfg.prediction_plot_count,
        device=device,
    )
    metrics["split"] = split
    metrics["checkpoint"] = str(cfg.checkpoint_path)
    metrics["graphical_dir"] = str(graphical_dir)
    metrics["prediction_graph_count"] = min(cfg.prediction_plot_count, len(antenna_indices))
    return metrics


def inspect_processed_dataset(cfg: Config) -> None:
    require_torch("inspect")
    base = ProcessedCurveDataset(
        cfg.processed_csv_path,
        cfg.processed_meta_path,
        sigma_db=cfg.notch_sigma_db,
    )
    print(f"Processed dataset: {cfg.processed_csv_path}")
    print(f"Metadata: {cfg.processed_meta_path}")
    print(f"Antennas: {len(base)}")
    print(f"Sequence length: {base.seq_len}")
    print(f"Frequency axis first 5: {base.freq_axis_ghz[:5].tolist()}")
    print(f"First geometry bits: {base.geometry[0, :10].tolist()}")
    print(f"First curve values: {base.curves_db[0, :5].tolist()}")


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="Preprocess uploaded patch antenna data and train R5 PINN per frequency."
    )
    parser.add_argument(
        "command",
        choices=["preprocess", "inspect", "train"],
        nargs="?",
        default=cfg.command,
    )
    parser.add_argument("--geometry-catalog-path", type=Path, default=cfg.geometry_catalog_path)
    parser.add_argument("--curves-root", type=Path, default=cfg.curves_root)
    parser.add_argument("--processed-csv-path", type=Path, default=cfg.processed_csv_path)
    parser.add_argument("--processed-meta-path", type=Path, default=cfg.processed_meta_path)
    parser.add_argument("--results-dir", type=Path, default=cfg.results_dir)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--optimizer", choices=["adamw", "adagrad"], default=cfg.optimizer_name)
    parser.add_argument("--seed", type=int, default=cfg.random_seed)
    parser.add_argument("--max-antennas", type=int, default=cfg.max_antennas)
    parser.add_argument("--device", type=str, default=cfg.device)
    parser.add_argument("--log-every-batches", type=int, default=cfg.log_every_batches)
    parser.add_argument("--eval-split", choices=["val", "test"])
    parser.add_argument("--prediction-plot-count", type=int, default=cfg.prediction_plot_count)
    parser.add_argument("--resonance-loss-weight", type=float, default=cfg.resonance_loss_weight)
    parser.add_argument("--overwrite-processed", action="store_true")
    args = parser.parse_args()

    cfg.command = args.command
    cfg.geometry_catalog_path = args.geometry_catalog_path
    cfg.curves_root = args.curves_root
    cfg.processed_csv_path = args.processed_csv_path
    cfg.processed_meta_path = args.processed_meta_path
    cfg.results_dir = args.results_dir
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.optimizer_name = args.optimizer
    cfg.random_seed = args.seed
    cfg.max_antennas = args.max_antennas
    cfg.device = args.device
    cfg.log_every_batches = args.log_every_batches
    cfg.eval_split = args.eval_split or ""
    cfg.prediction_plot_count = max(0, args.prediction_plot_count)
    cfg.resonance_loss_weight = max(0.0, args.resonance_loss_weight)
    cfg.overwrite_processed = args.overwrite_processed
    return cfg


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.random_seed)

    if cfg.command == "preprocess":
        summary = preprocess_uploaded_dataset(cfg)
        compact = {
            "geometry_catalog_path": summary["geometry_catalog_path"],
            "curves_root": summary["curves_root"],
            "processed_csv_path": summary["processed_csv_path"],
            "num_geometries": summary["num_geometries"],
            "num_curve_files": summary["num_curve_files"],
            "num_duplicate_curve_ids": summary["num_duplicate_curve_ids"],
            "duplicate_curve_ids": summary["duplicate_curve_ids"],
            "missing_curve_ids": summary["missing_curve_ids"],
            "matched_antennas": summary["matched_antennas"],
            "seq_len": summary["seq_len"],
            "freq_axis_ghz_preview": (
                summary["freq_axis_ghz"][:3] + ["..."] + summary["freq_axis_ghz"][-3:]
            ),
        }
        print(json.dumps(compact, indent=2))
        return

    if cfg.command == "inspect":
        if not cfg.processed_csv_path.exists() or not cfg.processed_meta_path.exists():
            preprocess_uploaded_dataset(cfg)
        inspect_processed_dataset(cfg)
        return

    if cfg.eval_split:
        metrics = run_saved_evaluation(cfg, cfg.eval_split)
        print(f"Evaluation split: {cfg.eval_split}")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        return

    summary = train_model(cfg)
    print(f"Prediction graphs saved to: {summary['prediction_graphical_dir']}")
    print("Training summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
