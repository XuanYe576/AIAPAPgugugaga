"""Patch Antenna AI R5.

Clean rewrite of `old/model_training/notebooks/patch_antenna_ai_colab_R5.ipynb`
for the current repository layout.

This script trains a GNN encoder plus FEDformer-style decoder that maps a
10x10 binary patch geometry to 61 S11 samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_csv_path() -> Path:
    candidates = [
        REPO_ROOT / "Data" / "Full_1000Data.csv",
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
    input_dim: int = 100
    seq_len: int = 61
    output_mode: str = "mag_only"
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50
    val_split: float = 0.2
    dmodel: int = 128
    nhead: int = 8
    ffn_hidden: int = 256
    num_transformer_layers: int = 2
    num_spectral_blocks: int = 2
    top_k_freq: int = 16
    freq_hz_start: float = 1e9
    freq_hz_stop: float = 6e9
    device: str = "auto"
    transformer_dropout: float = 0.3
    weight_decay: float = 1e-5
    aug_noise_std: float = 0.01
    aug_freq_mask_prob: float = 0.5
    aug_freq_mask_width: int = 8
    use_augmentation: bool = True
    random_seed: int = 42
    patience: int = 10
    output_dir: Path = field(default_factory=_default_output_dir)
    weights_filename: str = "gnn_fedformer_r5_best.pt"
    history_filename: str = "history.csv"
    loss_plot_filename: str = "loss_curve.png"
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


class AntennaDataset(Dataset):
    """Dataset for flattened 10x10 geometry plus S11 targets."""

    def __init__(
        self,
        csv_path: Path,
        input_dim: int = 100,
        seq_len: int = 61,
        output_mode: str = "complex_61",
    ) -> None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path, header=None)
        values = df.values.astype(np.float32)

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.output_mode = output_mode

        if output_mode == "complex_61":
            expected = input_dim + 2 * seq_len
            if values.shape[1] < expected:
                raise ValueError(
                    f"CSV has {values.shape[1]} columns; expected at least {expected} "
                    f"(100 geometry + 61 real + 61 imag)."
                )
            x = values[:, :input_dim]
            y_real = values[:, input_dim : input_dim + seq_len]
            y_imag = values[:, input_dim + seq_len : input_dim + 2 * seq_len]
        elif output_mode == "mag_only":
            expected = input_dim + seq_len
            if values.shape[1] < expected:
                raise ValueError(
                    f"CSV has {values.shape[1]} columns; expected at least {expected} "
                    f"(100 geometry + 61 magnitudes)."
                )
            x = values[:, :input_dim]
            y_real = values[:, input_dim : input_dim + seq_len]
            y_imag = np.zeros_like(y_real)
        else:
            raise ValueError(f"Unsupported output_mode: {output_mode}")

        self.X = torch.from_numpy(x)
        self.Y = torch.stack(
            [torch.from_numpy(y_real), torch.from_numpy(y_imag)],
            dim=-1,
        )

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


def build_grid_adjacency(h: int = 10, w: int = 10) -> torch.Tensor:
    n_nodes = h * w
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    def node_index(row: int, col: int) -> int:
        return row * w + col

    for row in range(h):
        for col in range(w):
            idx = node_index(row, col)
            if row > 0:
                adjacency[idx, node_index(row - 1, col)] = 1
            if row < h - 1:
                adjacency[idx, node_index(row + 1, col)] = 1
            if col > 0:
                adjacency[idx, node_index(row, col - 1)] = 1
            if col < w - 1:
                adjacency[idx, node_index(row, col + 1)] = 1

    adjacency += np.eye(n_nodes, dtype=np.float32)
    degree = np.sum(adjacency, axis=1)
    degree_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))
    normalized = degree_inv_sqrt @ adjacency @ degree_inv_sqrt
    return torch.from_numpy(normalized)


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.adjacency = adjacency
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ax = torch.einsum("ij,bjf->bif", self.adjacency, x)
        return F.relu(self.linear(ax))


class GridGraphEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        node_feat_dim: int = 1,
        hidden_dims: tuple[int, int] = (32, 64),
        adjacency: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if adjacency is None:
            raise ValueError("adjacency must be provided")
        self.gc1 = GraphConv(node_feat_dim, hidden_dims[0], adjacency)
        self.gc2 = GraphConv(hidden_dims[0], hidden_dims[1], adjacency)
        self.readout = nn.Linear(hidden_dims[1], d_model)

    def forward(self, geom_bits: torch.Tensor) -> torch.Tensor:
        batch = geom_bits.size(0)
        x = geom_bits.view(batch, 100, 1)
        hidden = self.gc1(x)
        hidden = self.gc2(hidden)
        pooled = hidden.mean(dim=1)
        return self.readout(pooled)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class SpectralBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        top_k: int = 16,
        ffn_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.top_k = min(top_k, seq_len // 2 + 1)
        self.w_real = nn.Parameter(torch.randn(d_model, self.top_k) * 0.02)
        self.w_imag = nn.Parameter(torch.randn(d_model, self.top_k) * 0.02)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch, seq_len, _ = x.shape

        x = self.ln1(x)
        x_ch_first = x.transpose(1, 2)
        spectrum = torch.fft.rfft(x_ch_first, dim=-1)

        idx = torch.arange(self.top_k, device=spectrum.device)
        spectrum_top_k = spectrum[..., idx]
        real = spectrum_top_k.real
        imag = spectrum_top_k.imag

        w_real = self.w_real.unsqueeze(0)
        w_imag = self.w_imag.unsqueeze(0)
        mixed_real = real * w_real - imag * w_imag
        mixed_imag = real * w_imag + imag * w_real

        updated = torch.zeros_like(spectrum)
        updated[..., idx] = torch.complex(mixed_real, mixed_imag)
        restored = torch.fft.irfft(updated, n=seq_len, dim=-1).transpose(1, 2)

        x = residual + restored
        return x + self.ff(self.ln2(x))


class FEDformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        nhead: int = 8,
        ffn_hidden: int = 256,
        num_transformer_layers: int = 2,
        num_spectral_blocks: int = 2,
        top_k: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos = PositionalEncoding(d_model, max_len=seq_len)
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
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )
        self.head = nn.Linear(d_model, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        z = self.pos(tokens)
        for block in self.spectral:
            z = block(z)
        z = self.transformer(z)
        return self.head(z)


class Geometry2SParam(nn.Module):
    def __init__(
        self,
        adjacency: torch.Tensor,
        seq_len: int = 61,
        dmodel: int = 128,
        nhead: int = 8,
        ffn_hidden: int = 256,
        num_transformer_layers: int = 2,
        num_spectral_blocks: int = 2,
        top_k: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = GridGraphEncoder(
            d_model=dmodel,
            node_feat_dim=1,
            hidden_dims=(32, 64),
            adjacency=adjacency,
        )
        self.to_tokens = nn.Linear(dmodel, seq_len * dmodel)
        self.decoder = FEDformerDecoder(
            d_model=dmodel,
            seq_len=seq_len,
            nhead=nhead,
            ffn_hidden=ffn_hidden,
            num_transformer_layers=num_transformer_layers,
            num_spectral_blocks=num_spectral_blocks,
            top_k=top_k,
            dropout=dropout,
        )
        self.seq_len = seq_len
        self.dmodel = dmodel

    def forward(self, geom_bits: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(geom_bits)
        tokens = self.to_tokens(embedding).view(-1, self.seq_len, self.dmodel)
        return self.decoder(tokens)


def complex_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def augment_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Augment the target spectrum during training."""
    y = y.clone()
    batch, seq_len, _ = y.shape

    if cfg.aug_noise_std > 0.0:
        noise = torch.randn(batch, seq_len, device=y.device) * cfg.aug_noise_std
        y = y + noise.unsqueeze(-1)

    if cfg.aug_freq_mask_prob > 0.0 and cfg.aug_freq_mask_width > 0:
        max_width = min(cfg.aug_freq_mask_width, seq_len)
        for b in range(batch):
            if torch.rand(1, device=y.device).item() < cfg.aug_freq_mask_prob:
                width = torch.randint(1, max_width + 1, (1,), device=y.device).item()
                start = torch.randint(0, seq_len - width + 1, (1,), device=y.device).item()
                y[b, start : start + width, :] = 0.0

    return x, y


def build_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, int, int]:
    dataset = AntennaDataset(
        cfg.csv_path,
        input_dim=cfg.input_dim,
        seq_len=cfg.seq_len,
        output_mode=cfg.output_mode,
    )
    total = len(dataset)
    n_val = int(cfg.val_split * total)
    n_train = total - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.random_seed),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader, n_train, n_val


def inspect_dataset(cfg: Config) -> None:
    dataset = AntennaDataset(
        cfg.csv_path,
        input_dim=cfg.input_dim,
        seq_len=cfg.seq_len,
        output_mode=cfg.output_mode,
    )
    x0, y0 = dataset[0]
    print(f"Dataset: {cfg.csv_path}")
    print(f"Samples: {len(dataset)}")
    print(f"Input sample shape: {tuple(x0.shape)}")
    print(f"Target sample shape: {tuple(y0.shape)}")
    print(f"First 10 input bits: {x0[:10].tolist()}")
    print(f"First 5 target rows: {y0[:5].tolist()}")


def build_model(cfg: Config, device: str) -> Geometry2SParam:
    adjacency = build_grid_adjacency(10, 10).to(device)
    return Geometry2SParam(
        adjacency=adjacency,
        seq_len=cfg.seq_len,
        dmodel=cfg.dmodel,
        nhead=cfg.nhead,
        ffn_hidden=cfg.ffn_hidden,
        num_transformer_layers=cfg.num_transformer_layers,
        num_spectral_blocks=cfg.num_spectral_blocks,
        top_k=cfg.top_k_freq,
        dropout=cfg.transformer_dropout,
    ).to(device)


def inspect_model(cfg: Config, device: str) -> None:
    model = build_model(cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy_x = torch.randint(0, 2, (2, cfg.input_dim), dtype=torch.float32).to(device)
    with torch.no_grad():
        dummy_y = model(dummy_x)
    print(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Dummy input shape: {tuple(dummy_x.shape)}")
    print(f"Dummy output shape: {tuple(dummy_y.shape)}")


def build_full_augmented_dataset(cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = AntennaDataset(
        cfg.csv_path,
        input_dim=cfg.input_dim,
        seq_len=cfg.seq_len,
        output_mode=cfg.output_mode,
    )
    total = len(dataset)
    n_val = int(cfg.val_split * total)
    n_train = total - n_val
    train_ds, _ = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.random_seed),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)

    x_batches: list[torch.Tensor] = []
    y_batches: list[torch.Tensor] = []
    for xb, yb in train_loader:
        xb_aug, yb_aug = augment_batch(xb, yb, cfg)
        x_batches.append(xb_aug)
        y_batches.append(yb_aug)
    return torch.cat(x_batches, dim=0), torch.cat(y_batches, dim=0)


def save_config(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = cfg.output_dir / cfg.config_filename
    payload: dict[str, object] = {}
    for key, value in asdict(cfg).items():
        if isinstance(value, Path):
            value = str(value)
        payload[key] = value
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def train_model(cfg: Config) -> tuple[Path, float, list[int], list[float], list[float]]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg)

    device = get_device(cfg.device)
    train_loader, val_loader, n_train, n_val = build_dataloaders(cfg)
    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    hist_epoch: list[int] = []
    hist_train: list[float] = []
    hist_val: list[float] = []
    best_val = float("inf")
    patience_left = cfg.patience

    with cfg.history_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if cfg.use_augmentation:
                xb, yb = augment_batch(xb, yb, cfg)

            optimizer.zero_grad()
            yhat = model(xb)
            loss = complex_mse(yhat, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= max(1, n_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                yhat = model(xb)
                loss = complex_mse(yhat, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= max(1, n_val)

        hist_epoch.append(epoch)
        hist_train.append(train_loss)
        hist_val.append(val_loss)

        with cfg.history_path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([epoch, train_loss, val_loss])

        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            torch.save(model.state_dict(), cfg.weights_path)
            print(f"  saved: {cfg.weights_path}")
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    return cfg.weights_path, best_val, hist_epoch, hist_train, hist_val


def plot_loss_curve(
    epochs: list[int],
    train_loss: list[float],
    val_loss: list[float],
    output_path: Path,
) -> None:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Complex MSE)")
    plt.title("Training / Validation Loss vs. Epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_args() -> Config:
    default_cfg = Config()
    parser = argparse.ArgumentParser(description="Train Patch Antenna AI R5 model.")
    parser.add_argument("--csv-path", type=Path, default=default_cfg.csv_path)
    parser.add_argument(
        "--output-mode",
        choices=["mag_only", "complex_61"],
        default=default_cfg.output_mode,
    )
    parser.add_argument("--epochs", type=int, default=default_cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=default_cfg.batch_size)
    parser.add_argument("--lr", type=float, default=default_cfg.lr)
    parser.add_argument("--device", type=str, default=default_cfg.device)
    parser.add_argument("--output-dir", type=Path, default=default_cfg.output_dir)
    parser.add_argument("--seed", type=int, default=default_cfg.random_seed)
    parser.add_argument("--no-augmentation", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    cfg.csv_path = args.csv_path
    cfg.output_mode = args.output_mode
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.device = args.device
    cfg.output_dir = args.output_dir
    cfg.random_seed = args.seed
    cfg.use_augmentation = not args.no_augmentation
    return cfg


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.random_seed)

    print("Configuration:")
    for key, value in asdict(cfg).items():
        print(f"  {key}: {value}")

    device = get_device(cfg.device)
    print(f"Using device: {device}")
    inspect_dataset(cfg)
    inspect_model(cfg, device)

    if cfg.use_augmentation:
        x_aug, y_aug = build_full_augmented_dataset(cfg)
        print(f"Augmented preview X shape: {tuple(x_aug.shape)}")
        print(f"Augmented preview Y shape: {tuple(y_aug.shape)}")

    weights_path, best_val, epochs, train_loss, val_loss = train_model(cfg)
    print(f"Training finished. Best validation loss: {best_val:.6f}")
    print(f"Best model path: {weights_path}")

    plot_loss_curve(epochs, train_loss, val_loss, cfg.loss_plot_path)
    print(f"Loss curve saved to: {cfg.loss_plot_path}")


if __name__ == "__main__":
    main()
