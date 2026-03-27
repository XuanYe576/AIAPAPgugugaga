"""R5.5B-s: notebook-style R5 model with the current script interface.

This version intentionally switches the neural network back to the older
notebook R5 stack:

- fixed 10x10 four-neighbor grid adjacency
- 2-layer GraphConv encoder with mean pooling
- linear expansion from one geometry embedding to per-frequency tokens
- spectral blocks + Transformer decoder
- plain complex MSE objective with target-side augmentation

The surrounding CLI, output folders, checkpoints, history logging, and
prediction graph export remain aligned with the current repository workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from metrics.prediction_graphs import (
    plot_loss_curve,
    save_complex_prediction_graphs,
    save_real_channel_prediction_graphs,
)
from utils.adamw import create_optimizer as create_shared_optimizer

GRID_HEIGHT = 10
GRID_WIDTH = 10
NUM_CELLS = GRID_HEIGHT * GRID_WIDTH


def _default_csv_path() -> Path:
    candidates = [
        REPO_ROOT / "old" / "data" / "Full_1000Data.csv",
        REPO_ROOT / "Data" / "TrainData.csv",
        REPO_ROOT / "Data" / "Full_122ComplexData.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _default_output_dir() -> Path:
    return REPO_ROOT / "results" / "R5.5B-s"


@dataclass(frozen=True)
class LayoutInfo:
    name: str
    seq_len: int
    is_complex: bool
    expected_cols: int


SUPPORTED_LAYOUTS: dict[str, LayoutInfo] = {
    "complex_61": LayoutInfo("complex_61", 61, True, NUM_CELLS + 2 * 61),
    "mag_only": LayoutInfo("mag_only", 61, False, NUM_CELLS + 61),
}


@dataclass
class Config:
    csv_path: Path = field(default_factory=_default_csv_path)
    input_dim: int = NUM_CELLS
    seq_len: int = 61
    output_mode: str = "auto"
    batch_size: int = 64
    lr: float = 1e-3
    optimizer_name: str = "adamw"
    weight_decay: float = 1e-5
    epochs: int = 50
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    dmodel: int = 128
    nhead: int = 8
    ffn_hidden: int = 256
    num_transformer_layers: int = 2
    num_spectral_blocks: int = 2
    top_k_freq: int = 16
    transformer_dropout: float = 0.30
    freq_hz_start: float = 1.0e9
    freq_hz_stop: float = 6.0e9
    device: str = "auto"
    random_seed: int = 42
    patience: int = 10
    use_augmentation: bool = True
    aug_noise_std: float = 0.01
    aug_freq_mask_prob: float = 0.50
    aug_freq_mask_width: int = 8
    log_every_batches: int = 10
    graph_k: int = 8
    eval_split: str = ""
    prediction_plot_count: int = 12
    output_dir: Path = field(default_factory=_default_output_dir)
    weights_filename: str = "r55bs_best.pt"
    checkpoint_filename: str = "r55bs_best.ckpt"
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

    @property
    def prediction_graphical_dir(self) -> Path:
        return self.output_dir / "weight" / "graphical"


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


def infer_layout(column_count: int, requested_mode: str) -> LayoutInfo:
    if requested_mode == "auto":
        if column_count >= SUPPORTED_LAYOUTS["complex_61"].expected_cols:
            return SUPPORTED_LAYOUTS["complex_61"]
        if column_count >= SUPPORTED_LAYOUTS["mag_only"].expected_cols:
            return SUPPORTED_LAYOUTS["mag_only"]
        raise ValueError(
            f"Unsupported CSV layout with {column_count} columns. "
            "Expected 222 (100 + 61x2) or 161 (100 + 61)."
        )

    layout = SUPPORTED_LAYOUTS[requested_mode]
    if column_count < layout.expected_cols:
        raise ValueError(
            f"Requested {layout.name} expects at least {layout.expected_cols} columns "
            f"but found {column_count}."
        )
    return layout


class AntennaDataset(Dataset):
    """Dataset for flattened 10x10 geometry and 61-point complex targets."""

    def __init__(
        self,
        csv_path: Path,
        input_dim: int = NUM_CELLS,
        output_mode: str = "complex_61",
    ) -> None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path, header=None)
        values = df.values.astype(np.float32)
        if values.ndim != 2:
            raise ValueError("CSV must be a 2D tabular array.")

        layout = infer_layout(values.shape[1], output_mode)
        x = values[:, :input_dim]
        y_real = values[:, input_dim : input_dim + layout.seq_len]
        if layout.is_complex:
            y_imag = values[:, input_dim + layout.seq_len : input_dim + 2 * layout.seq_len]
        else:
            y_imag = np.zeros_like(y_real)

        self.csv_path = csv_path
        self.input_dim = input_dim
        self.seq_len = layout.seq_len
        self.output_mode = layout.name
        self.layout = layout
        self.X = torch.from_numpy((x > 0.5).astype(np.float32))
        self.Y = torch.stack(
            [torch.from_numpy(y_real), torch.from_numpy(y_imag)],
            dim=-1,
        )

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


def build_grid_adjacency(height: int = GRID_HEIGHT, width: int = GRID_WIDTH) -> torch.Tensor:
    adjacency = np.zeros((height * width, height * width), dtype=np.float32)

    def node_index(row: int, col: int) -> int:
        return row * width + col

    for row in range(height):
        for col in range(width):
            idx = node_index(row, col)
            if row > 0:
                adjacency[idx, node_index(row - 1, col)] = 1.0
            if row < height - 1:
                adjacency[idx, node_index(row + 1, col)] = 1.0
            if col > 0:
                adjacency[idx, node_index(row, col - 1)] = 1.0
            if col < width - 1:
                adjacency[idx, node_index(row, col + 1)] = 1.0

    adjacency += np.eye(height * width, dtype=np.float32)
    degree = adjacency.sum(axis=1)
    degree_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))
    normalized = degree_inv_sqrt @ adjacency @ degree_inv_sqrt
    return torch.from_numpy(normalized)


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        propagated = torch.einsum("ij,bjf->bif", self.adjacency, x)
        return F.relu(self.linear(propagated))


class GridGraphEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        adjacency: torch.Tensor,
        node_feat_dim: int = 1,
        hidden_dims: tuple[int, int] = (32, 64),
    ) -> None:
        super().__init__()
        self.gc1 = GraphConv(node_feat_dim, hidden_dims[0], adjacency)
        self.gc2 = GraphConv(hidden_dims[0], hidden_dims[1], adjacency)
        self.readout = nn.Linear(hidden_dims[1], d_model)

    def forward(self, geom_bits: torch.Tensor) -> torch.Tensor:
        x = geom_bits.view(geom_bits.size(0), NUM_CELLS, 1)
        h = self.gc1(x)
        h = self.gc2(h)
        pooled = h.mean(dim=1)
        return self.readout(pooled)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SpectralBlock(nn.Module):
    def __init__(self, d_model: int, seq_len: int, top_k: int = 16, ffn_hidden: int = 256) -> None:
        super().__init__()
        self.top_k = min(top_k, seq_len // 2 + 1)
        self.w_real = nn.Parameter(torch.randn(d_model, self.top_k) * 0.02)
        self.w_imag = nn.Parameter(torch.randn(d_model, self.top_k) * 0.02)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        normalized = self.norm1(x)
        x_channels = normalized.transpose(1, 2)
        spectrum = torch.fft.rfft(x_channels, dim=-1)
        idx = torch.arange(self.top_k, device=spectrum.device)
        spectrum_top = spectrum[..., idx]
        real = spectrum_top.real
        imag = spectrum_top.imag
        w_real = self.w_real.unsqueeze(0)
        w_imag = self.w_imag.unsqueeze(0)
        mod_real = real * w_real - imag * w_imag
        mod_imag = real * w_imag + imag * w_real
        modified = torch.complex(mod_real, mod_imag)
        rebuilt = torch.zeros_like(spectrum)
        rebuilt[..., idx] = modified
        time_domain = torch.fft.irfft(rebuilt, n=x.size(1), dim=-1).transpose(1, 2)
        x = residual + time_domain
        return x + self.ffn(self.norm2(x))


class FEDformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        nhead: int,
        ffn_hidden: int,
        num_transformer_layers: int,
        num_spectral_blocks: int,
        top_k: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.positional = PositionalEncoding(d_model, max_len=seq_len)
        self.spectral = nn.ModuleList(
            [
                SpectralBlock(d_model, seq_len, top_k=top_k, ffn_hidden=ffn_hidden)
                for _ in range(num_spectral_blocks)
            ]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_hidden,
            batch_first=True,
            activation="gelu",
            norm_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.positional(tokens)
        for block in self.spectral:
            x = block(x)
        x = self.transformer(x)
        return self.head(x)


class Geometry2SParam(nn.Module):
    def __init__(self, cfg: Config, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.seq_len = cfg.seq_len
        self.dmodel = cfg.dmodel
        self.encoder = GridGraphEncoder(
            d_model=cfg.dmodel,
            adjacency=adjacency,
            node_feat_dim=1,
            hidden_dims=(32, 64),
        )
        self.to_tokens = nn.Linear(cfg.dmodel, cfg.seq_len * cfg.dmodel)
        self.decoder = FEDformerDecoder(
            d_model=cfg.dmodel,
            seq_len=cfg.seq_len,
            nhead=cfg.nhead,
            ffn_hidden=cfg.ffn_hidden,
            num_transformer_layers=cfg.num_transformer_layers,
            num_spectral_blocks=cfg.num_spectral_blocks,
            top_k=cfg.top_k_freq,
            dropout=cfg.transformer_dropout,
        )

    def forward(
        self,
        geom_bits: torch.Tensor,
        freq_axis_hz: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        geom_repr = self.encoder(geom_bits)
        tokens = self.to_tokens(geom_repr).view(-1, self.seq_len, self.dmodel)
        gamma = self.decoder(tokens)
        return {"gamma": gamma}


def create_optimizer(cfg: Config, parameters) -> torch.optim.Optimizer:
    return create_shared_optimizer(
        cfg.optimizer_name,
        parameters,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )


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


def complex_mse(pred_gamma: torch.Tensor, target_gamma: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_gamma, target_gamma)


def augment_batch(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    y_aug = y.clone()
    batch_size, seq_len, _ = y_aug.shape

    if cfg.aug_noise_std > 0.0:
        noise = torch.randn(batch_size, seq_len, device=y_aug.device) * cfg.aug_noise_std
        y_aug = y_aug + noise.unsqueeze(-1)

    if cfg.aug_freq_mask_prob > 0.0 and cfg.aug_freq_mask_width > 0:
        max_width = min(cfg.aug_freq_mask_width, seq_len)
        for batch_idx in range(batch_size):
            if torch.rand(1, device=y_aug.device).item() >= cfg.aug_freq_mask_prob:
                continue
            width = torch.randint(1, max_width + 1, (1,), device=y_aug.device).item()
            start = torch.randint(0, seq_len - width + 1, (1,), device=y_aug.device).item()
            y_aug[batch_idx, start : start + width, :] = 0.0

    return x, y_aug


def random_sample_split(dataset: AntennaDataset, cfg: Config) -> tuple[Subset, Subset, Subset]:
    total = len(dataset)
    if total < 3:
        raise ValueError("R5.5B-s requires at least 3 samples to create train/val/test splits.")

    train_len = int(total * cfg.train_ratio)
    val_len = int(total * cfg.val_ratio)
    train_len = max(1, train_len)
    val_len = max(1, val_len)
    test_len = total - train_len - val_len

    if test_len <= 0:
        test_len = 1
        if train_len >= val_len and train_len > 1:
            train_len -= 1
        elif val_len > 1:
            val_len -= 1
        else:
            train_len = max(1, train_len - 1)

    while train_len + val_len + test_len > total:
        if train_len >= val_len and train_len > 1:
            train_len -= 1
        elif val_len > 1:
            val_len -= 1
        else:
            test_len -= 1

    lengths = [train_len, val_len, test_len]
    generator = torch.Generator().manual_seed(cfg.random_seed)
    train_ds, val_ds, test_ds = random_split(dataset, lengths, generator=generator)
    return train_ds, val_ds, test_ds


def build_dataloaders(
    dataset: AntennaDataset,
    cfg: Config,
    device: str,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = random_sample_split(dataset, cfg)
    pin_memory = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def sync_config_from_dataset(cfg: Config, dataset: AntennaDataset) -> None:
    cfg.seq_len = dataset.seq_len
    cfg.output_mode = dataset.output_mode


def save_prediction_graphs_for_layout(
    model: Geometry2SParam,
    loader: DataLoader,
    dataset: AntennaDataset,
    cfg: Config,
    device: str,
    split: str,
) -> Path:
    freq_axis_hz = frequency_axis(cfg, device)
    if dataset.layout.is_complex:
        return save_complex_prediction_graphs(
            model=model,
            loader=loader,
            freq_axis_hz=freq_axis_hz,
            gamma_db_fn=gamma_db,
            output_dir=cfg.prediction_graphical_dir / split,
            split=split,
            plot_count=cfg.prediction_plot_count,
            device=device,
        )
    return save_real_channel_prediction_graphs(
        model=model,
        loader=loader,
        freq_axis_hz=freq_axis_hz,
        output_dir=cfg.prediction_graphical_dir / split,
        split=split,
        plot_count=cfg.prediction_plot_count,
        device=device,
        channel_idx=0,
        ylabel="S11 (dB)",
    )


def inspect_dataset(dataset: AntennaDataset) -> None:
    x0, y0 = dataset[0]
    print(f"Dataset: {dataset.csv_path}")
    print(f"Samples: {len(dataset)}")
    print(f"Detected layout: {dataset.layout.name}")
    print(f"Input sample shape: {tuple(x0.shape)}")
    print(f"Target sample shape: {tuple(y0.shape)}")
    print(f"First 10 geometry bits: {x0[:10].tolist()}")
    print(f"First 3 target rows: {y0[:3].tolist()}")


def build_model(cfg: Config, device: str) -> Geometry2SParam:
    adjacency = build_grid_adjacency().to(device)
    return Geometry2SParam(cfg, adjacency).to(device)


def inspect_model(cfg: Config, device: str) -> None:
    model = build_model(cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy_x = torch.randint(0, 2, (2, cfg.input_dim), dtype=torch.float32, device=device)
    with torch.no_grad():
        outputs = model(dummy_x, frequency_axis(cfg, device))
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
    model: Geometry2SParam,
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
    model: Geometry2SParam,
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
    model: Geometry2SParam,
    loader: DataLoader,
    cfg: Config,
    device: str,
) -> dict[str, float]:
    model.eval()
    freq_axis_hz = frequency_axis(cfg, device)
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb, freq_axis_hz)
            loss = complex_mse(outputs["gamma"], yb)
            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            count += batch_size

    metric = total_loss / max(1, count)
    return {"total": metric, "complex_mse": metric}


def train_model(cfg: Config) -> tuple[dict[str, float], list[dict[str, float]]]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg)

    device = get_device(cfg.device)
    dataset = AntennaDataset(cfg.csv_path, input_dim=cfg.input_dim, output_mode=cfg.output_mode)
    sync_config_from_dataset(cfg, dataset)
    train_loader, val_loader, test_loader = build_dataloaders(dataset, cfg, device)
    model = build_model(cfg, device)
    optimizer = create_optimizer(cfg, model.parameters())
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
                "val_total",
                "val_complex_mse",
                "lr",
            ]
        )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_total = 0.0
        count = 0

        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            if cfg.use_augmentation:
                xb, yb = augment_batch(xb, yb, cfg)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(xb, freq_axis_hz)
            loss = complex_mse(outputs["gamma"], yb)
            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            train_total += loss.item() * batch_size
            count += batch_size

            if cfg.log_every_batches > 0 and batch_idx % cfg.log_every_batches == 0:
                running = train_total / max(1, count)
                print(
                    "Epoch "
                    f"{epoch:03d} | "
                    f"batch {batch_idx:04d}/{len(train_loader):04d} | "
                    f"loss {running:.6f} | "
                    f"lr {optimizer.param_groups[0]['lr']:.6e}"
                )

        train_metric = train_total / max(1, count)
        val_metrics = evaluate_model(model, val_loader, cfg, device)
        row = {
            "epoch": epoch,
            "train_total": train_metric,
            "train_complex_mse": train_metric,
            "val_total": val_metrics["total"],
            "val_complex_mse": val_metrics["complex_mse"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history_rows.append(row)

        with cfg.history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    row["epoch"],
                    row["train_total"],
                    row["train_complex_mse"],
                    row["val_total"],
                    row["val_complex_mse"],
                    row["lr"],
                ]
            )

        print(
            "Epoch "
            f"{epoch:03d} | "
            f"train {row['train_total']:.6f} | "
            f"val {row['val_total']:.6f} | "
            f"cmse {row['val_complex_mse']:.6f}"
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
    test_metrics = evaluate_model(best_model, test_loader, cfg, device)
    graphical_dir = save_prediction_graphs_for_layout(
        model=best_model,
        loader=test_loader,
        dataset=dataset,
        cfg=cfg,
        device=device,
        split="test",
    )
    summary = {
        "model_family": "r5_notebook_adapter",
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "test_total": test_metrics["total"],
        "test_complex_mse": test_metrics["complex_mse"],
        "layout": dataset.layout.name,
        "seq_len": dataset.seq_len,
        "prediction_graphical_dir": str(graphical_dir),
        "prediction_graph_count": min(cfg.prediction_plot_count, len(test_loader.dataset)),
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
    metrics = evaluate_model(model, loader, cfg, device)
    graphical_dir = save_prediction_graphs_for_layout(
        model=model,
        loader=loader,
        dataset=dataset,
        cfg=cfg,
        device=device,
        split=split,
    )
    metrics["split"] = split
    metrics["checkpoint"] = str(cfg.checkpoint_path)
    metrics["graphical_dir"] = str(graphical_dir)
    metrics["prediction_graph_count"] = min(cfg.prediction_plot_count, len(loader.dataset))
    return metrics


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="Train the notebook-style R5.5B-s GraphConv + FEDformer model.",
    )
    parser.add_argument("--csv-path", type=Path, default=cfg.csv_path)
    parser.add_argument(
        "--output-mode",
        choices=["complex_61", "mag_only", "auto"],
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
    parser.add_argument(
        "--graph-k",
        type=int,
        default=cfg.graph_k,
        help="Legacy compatibility flag. Notebook R5.5B-s uses fixed grid adjacency.",
    )
    parser.add_argument("--log-every-batches", type=int, default=cfg.log_every_batches)
    parser.add_argument("--eval-split", choices=["val", "test"])
    parser.add_argument("--prediction-plot-count", type=int, default=cfg.prediction_plot_count)
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
    cfg.eval_split = args.eval_split or ""
    cfg.prediction_plot_count = max(0, args.prediction_plot_count)
    return cfg


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.random_seed)

    if cfg.eval_split:
        metrics = run_saved_evaluation(cfg, cfg.eval_split)
        print(f"Evaluation split: {cfg.eval_split}")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        return

    print("R5.5B-s configuration:")
    for key, value in asdict(cfg).items():
        print(f"  {key}: {value}")

    dataset = AntennaDataset(cfg.csv_path, input_dim=cfg.input_dim, output_mode=cfg.output_mode)
    sync_config_from_dataset(cfg, dataset)

    inspect_dataset(dataset)
    device = get_device(cfg.device)
    print(f"Using device: {device}")
    inspect_model(cfg, device)

    summary, history_rows = train_model(cfg)
    plot_loss_curve(
        history_rows,
        cfg.loss_plot_path,
        title="R5.5B-s Notebook Training Curve",
    )
    print(f"Loss curve saved to: {cfg.loss_plot_path}")
    print(f"Prediction graphs saved to: {summary['prediction_graphical_dir']}")
    print("Training summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
