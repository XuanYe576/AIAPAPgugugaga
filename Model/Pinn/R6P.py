"""
R6P: pole-residue physics-guided surrogate built on top of the R5 PINN pipeline.

Core idea:
- Input: 10x10 geometry + optional material map channel.
- Shared CNN encoder.
- Two decoding branches:
  1) Rational S11 branch: predicts poles/residues/direct/delay.
  2) Dedicated notch branch: predicts Gaussian notch terms.
- Optional auxiliary physics branch:
  predicts port propagation constant proxy and coarse field map, regularized by
  transmission-line/Helmholtz-style residuals.
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

from metrics.prediction_graphs import plot_loss_curve
from metrics.plotting import load_pyplot
from utils.adamw import create_optimizer as create_shared_optimizer
from utils.amp import autocast_context, build_grad_scaler
from utils.dataP import RAW_DATA_ROOT, preprocess_uploaded_dataset

GRID_HEIGHT = 10
GRID_WIDTH = 10
NUM_CELLS = GRID_HEIGHT * GRID_WIDTH
EPS = 1.0e-9


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
    return REPO_ROOT / "results" / "R6P_pole_residue"


@dataclass
class Config:
    command: str = "train"
    geometry_catalog_path: Path = field(default_factory=_default_geometry_catalog_path)
    curves_root: Path = RAW_DATA_ROOT
    processed_csv_path: Path = field(default_factory=_default_processed_csv)
    processed_meta_path: Path = field(default_factory=_default_processed_meta)
    results_dir: Path = field(default_factory=_default_results_dir)
    input_dim: int = NUM_CELLS
    batch_size: int = 128
    lr: float = 1.0e-3
    optimizer_name: str = "adamw"
    weight_decay: float = 1.0e-4
    epochs: int = 120
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    encoder_dim: int = 256
    dropout: float = 0.10
    num_poles: int = 8
    num_notches: int = 2
    use_aux_head: bool = True
    use_material_channel: bool = True
    er_default: float = 4.4
    er_min: float = 1.0
    er_max: float = 12.0
    freq_hz_min: float = 1.0e9
    freq_hz_max: float = 6.0e9
    loss_data: float = 1.0
    loss_passive: float = 0.10
    loss_pole_prior: float = 0.20
    loss_beta: float = 0.10
    loss_field: float = 0.02
    loss_notch: float = 0.05
    phys_mode_count: int = 2
    pole_damping_ratio: float = 0.03
    notch_sigma_db: float = 4.0
    patience: int = 18
    gradient_clip: float = 1.0
    log_every_batches: int = 10
    use_amp: bool = True
    random_seed: int = 42
    max_antennas: int = 0
    device: str = "auto"
    eval_split: str = ""
    prediction_plot_count: int = 12
    overwrite_processed: bool = False
    weights_filename: str = "r6p_best.pt"
    checkpoint_filename: str = "r6p_best.ckpt"
    history_filename: str = "history.csv"
    summary_filename: str = "summary.json"
    config_filename: str = "config.json"
    loss_plot_filename: str = "loss_curve.png"

    @property
    def weights_path(self) -> Path:
        return self.results_dir / self.weights_filename

    @property
    def history_path(self) -> Path:
        return self.results_dir / self.history_filename

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

    @property
    def loss_plot_path(self) -> Path:
        return self.results_dir / self.loss_plot_filename


@dataclass
class LossBreakdown:
    total: torch.Tensor
    data: torch.Tensor
    passive: torch.Tensor
    pole_prior: torch.Tensor
    beta: torch.Tensor
    field: torch.Tensor
    notch: torch.Tensor


@dataclass(frozen=True)
class EvalOutputs:
    gamma_complex: torch.Tensor
    gamma_db: torch.Tensor
    poles: torch.Tensor
    residues: torch.Tensor
    beta_pred: torch.Tensor
    field_map: torch.Tensor
    notch_params: torch.Tensor


class ProcessedCurveDataset:
    def __init__(self, csv_path: Path, meta_path: Path, sigma_db: float, er_default: float) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Processed CSV not found: {csv_path}")
        values = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[1] < NUM_CELLS + 1:
            raise ValueError(
                f"Processed CSV must have at least {NUM_CELLS + 1} columns, found {values.shape[1]}."
            )

        inferred_seq_len = int(values.shape[1] - NUM_CELLS)
        inferred_freq_axis = np.linspace(1.0, 6.0, inferred_seq_len, dtype=np.float32).tolist()
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as handle:
                meta = json.load(handle)
        else:
            meta = {
                "seq_len": inferred_seq_len,
                "freq_axis_ghz": inferred_freq_axis,
                "matched_antenna_ids": list(range(1, values.shape[0] + 1)),
            }

        self.csv_path = csv_path
        self.meta_path = meta_path
        self.meta = meta
        self.seq_len = int(meta.get("seq_len", inferred_seq_len))
        if values.shape[1] != NUM_CELLS + self.seq_len:
            raise ValueError(
                f"Processed CSV column count mismatch. Expected {NUM_CELLS + self.seq_len}, found {values.shape[1]}."
            )

        self.geometry = torch.from_numpy((values[:, :NUM_CELLS] > 0.5).astype(np.float32))
        self.curves_db = torch.from_numpy(values[:, NUM_CELLS:])
        freq_axis_ghz = meta.get("freq_axis_ghz", inferred_freq_axis)
        if len(freq_axis_ghz) != self.seq_len:
            freq_axis_ghz = inferred_freq_axis
        self.freq_axis_ghz = torch.tensor(freq_axis_ghz, dtype=torch.float32)
        self.freq_axis_hz = self.freq_axis_ghz * 1.0e9
        self.antenna_ids = meta.get(
            "matched_antenna_ids",
            list(range(1, self.geometry.size(0) + 1)),
        )
        self.sample_weights = resonance_weights_db(self.curves_db, sigma_db=sigma_db)

        self.material_map = torch.full(
            (self.geometry.size(0), NUM_CELLS),
            float(er_default),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return self.geometry.size(0)


class AntennaCurveDataset(Dataset):
    def __init__(self, base: ProcessedCurveDataset, indices: list[int]) -> None:
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        base_idx = self.indices[idx]
        return (
            self.base.geometry[base_idx],
            self.base.material_map[base_idx],
            self.base.curves_db[base_idx],
            base_idx,
        )


def resonance_weights_db(curves_db: torch.Tensor, sigma_db: float) -> torch.Tensor:
    minima = curves_db.min(dim=1, keepdim=True).values
    return 1.0 + 2.0 * torch.exp(-((curves_db - minima) / sigma_db) ** 2)


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
            "Preprocess can run without torch, but train/inspect/eval cannot."
        )


def create_optimizer(cfg: Config, parameters: object) -> object:
    require_torch("optimizer creation")
    return create_shared_optimizer(
        cfg.optimizer_name,
        parameters,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )


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
    best_val_total: float,
) -> None:
    require_torch("checkpoint save")
    torch.save(
        {
            "epoch": epoch,
            "best_val_total": best_val_total,
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


def theoretical_resonant_freq_hz(L_m: torch.Tensor, eps_eff: torch.Tensor) -> torch.Tensor:
    c_m_per_s = 3.0e8
    h_m = 1.57e-3
    L_m = L_m.to(dtype=torch.float32).clamp_min(1.0e-6)
    eps_eff = eps_eff.to(dtype=torch.float32).clamp_min(1.0)
    u = 1.0 + 12.0 * h_m / L_m
    ee = (eps_eff + 1.0) / 2.0 + (eps_eff - 1.0) / (2.0 * torch.sqrt(u))
    return c_m_per_s / (2.0 * L_m * torch.sqrt(ee))


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
) -> tuple[DataLoader, DataLoader, DataLoader, list[int], list[int], list[int]]:
    train_idx, val_idx, test_idx = family_aware_split_indices(base, cfg)
    pin_memory = device == "cuda"
    train_loader = DataLoader(
        AntennaCurveDataset(base, train_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        AntennaCurveDataset(base, val_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        AntennaCurveDataset(base, test_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx


class SharedConvEncoder(nn.Module):
    def __init__(self, in_channels: int, encoder_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class R6PPoleResidue(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        in_channels = 2 if cfg.use_material_channel else 1
        self.cfg = cfg
        self.encoder = SharedConvEncoder(in_channels=in_channels, encoder_dim=cfg.encoder_dim, dropout=cfg.dropout)

        hidden = cfg.encoder_dim
        self.pole_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.pole_out = nn.Linear(hidden, cfg.num_poles * 4 + 3)

        self.notch_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden // 2, cfg.num_notches * 3),
        )

        self.aux_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden // 2, 1 + 25),
        )

    def _build_input(self, geom_bits: torch.Tensor, material_map: torch.Tensor) -> torch.Tensor:
        geom = geom_bits.view(-1, 1, GRID_HEIGHT, GRID_WIDTH)
        if self.cfg.use_material_channel:
            er = material_map.view(-1, 1, GRID_HEIGHT, GRID_WIDTH)
            er = (er - self.cfg.er_min) / max(self.cfg.er_max - self.cfg.er_min, 1.0e-6)
            er = er.clamp(0.0, 1.0)
            return torch.cat([geom, er], dim=1)
        return geom

    def _decode_rational(
        self,
        packed: torch.Tensor,
        freq_hz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = packed.size(0)
        n = self.cfg.num_poles
        out = self.pole_out(packed)

        idx = 0
        raw_alpha = out[:, idx : idx + n]
        idx += n
        raw_freq = out[:, idx : idx + n]
        idx += n
        raw_r_re = out[:, idx : idx + n]
        idx += n
        raw_r_im = out[:, idx : idx + n]
        idx += n
        direct_re = out[:, idx : idx + 1]
        idx += 1
        direct_im = out[:, idx : idx + 1]
        idx += 1
        raw_tau = out[:, idx : idx + 1]

        alpha_hz = 1.0e7 + F.softplus(raw_alpha) * 8.0e8
        pole_freq_hz = self.cfg.freq_hz_min + (self.cfg.freq_hz_max - self.cfg.freq_hz_min) * torch.sigmoid(raw_freq)
        pole_real = -2.0 * math.pi * alpha_hz
        pole_imag = 2.0 * math.pi * pole_freq_hz

        residue_scale = 0.6
        residue_re = residue_scale * torch.tanh(raw_r_re)
        residue_im = residue_scale * torch.tanh(raw_r_im)

        poles = torch.complex(pole_real, pole_imag)
        residues = torch.complex(residue_re, residue_im)
        direct = torch.complex(torch.tanh(direct_re), torch.tanh(direct_im)).squeeze(-1)
        tau = 2.0e-10 * torch.tanh(raw_tau).squeeze(-1)

        omega = 2.0 * math.pi * freq_hz.view(1, 1, -1)
        jw = torch.complex(torch.zeros_like(omega), omega)
        poles_e = poles.unsqueeze(-1)
        residues_e = residues.unsqueeze(-1)
        conj_p = torch.conj(poles_e)
        conj_r = torch.conj(residues_e)

        rational_terms = residues_e / (jw - poles_e) + conj_r / (jw - conj_p)
        rational_sum = rational_terms.sum(dim=1)
        jw_tau = torch.complex(torch.zeros_like(omega[:, 0, :]), omega[:, 0, :] * tau.unsqueeze(-1))
        gamma = direct.unsqueeze(-1) + jw_tau + rational_sum
        return gamma, poles, residues

    def _decode_notch_gain(
        self,
        features: torch.Tensor,
        freq_hz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = features.size(0)
        n = self.cfg.num_notches
        raw = self.notch_head(features).view(bsz, n, 3)
        f0 = self.cfg.freq_hz_min + (self.cfg.freq_hz_max - self.cfg.freq_hz_min) * torch.sigmoid(raw[:, :, 0])
        bw = 5.0e7 + F.softplus(raw[:, :, 1]) * 5.0e8
        depth = 0.95 * torch.sigmoid(raw[:, :, 2])

        f = freq_hz.view(1, 1, -1)
        g = torch.exp(-0.5 * ((f - f0.unsqueeze(-1)) / bw.unsqueeze(-1)) ** 2)
        notch = 1.0 - depth.unsqueeze(-1) * g
        notch_gain = torch.prod(torch.clamp(notch, min=0.05), dim=1)
        notch_params = torch.stack([f0, bw, depth], dim=-1)
        return notch_gain, notch_params

    def _decode_aux(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.aux_head(features)
        beta = 1.0 + F.softplus(raw[:, 0])
        field = raw[:, 1:].view(-1, 5, 5)
        return beta, field

    def forward(
        self,
        geom_bits: torch.Tensor,
        material_map: torch.Tensor,
        freq_hz: torch.Tensor,
    ) -> EvalOutputs:
        x = self._build_input(geom_bits, material_map)
        embedding = self.encoder(x)
        packed = self.pole_head(embedding)
        gamma_rational, poles, residues = self._decode_rational(packed, freq_hz)
        notch_gain, notch_params = self._decode_notch_gain(embedding, freq_hz)
        gamma = gamma_rational * torch.complex(notch_gain, torch.zeros_like(notch_gain))
        gamma_mag = torch.abs(gamma).clamp_min(EPS)
        gamma_db = 20.0 * torch.log10(gamma_mag)

        if self.cfg.use_aux_head:
            beta_pred, field_map = self._decode_aux(embedding)
        else:
            beta_pred = gamma_db.new_zeros((gamma_db.size(0),))
            field_map = gamma_db.new_zeros((gamma_db.size(0), 5, 5))

        return EvalOutputs(
            gamma_complex=gamma,
            gamma_db=gamma_db,
            poles=poles,
            residues=residues,
            beta_pred=beta_pred,
            field_map=field_map,
            notch_params=notch_params,
        )


def _field_helmholtz_loss(field_map: torch.Tensor, beta_pred: torch.Tensor) -> torch.Tensor:
    if field_map.numel() == 0:
        return field_map.new_zeros(())
    kernel = field_map.new_tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    ).view(1, 1, 3, 3)
    field = field_map.unsqueeze(1)
    lap = F.conv2d(field, kernel, padding=1).squeeze(1)
    beta_scale = (beta_pred / 200.0).view(-1, 1, 1)
    residual = lap + beta_scale * field_map
    return residual.pow(2).mean()


def _notch_target_stats(
    curves_db: torch.Tensor,
    freq_hz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    notch_idx = curves_db.argmin(dim=1)
    notch_freq = freq_hz[notch_idx]
    notch_depth = -curves_db.min(dim=1).values
    return notch_freq, notch_depth


def compute_losses(
    outputs: EvalOutputs,
    geom_bits: torch.Tensor,
    material_map: torch.Tensor,
    target_db: torch.Tensor,
    sample_weights: torch.Tensor,
    freq_hz: torch.Tensor,
    cfg: Config,
) -> LossBreakdown:
    data = sample_weights * F.smooth_l1_loss(outputs.gamma_db, target_db, reduction="none")
    data_loss = data.mean()

    gamma_abs = torch.abs(outputs.gamma_complex)
    passive_loss = F.relu(gamma_abs - 1.0).pow(2).mean()

    eps_eff = material_map.mean(dim=1).clamp(min=cfg.er_min, max=cfg.er_max)
    L_m = extract_patch_length(geom_bits)
    fr_hz = theoretical_resonant_freq_hz(L_m, eps_eff)

    mode_count = max(1, min(cfg.phys_mode_count, cfg.num_poles))
    pole_imag_hz = outputs.poles.imag / (2.0 * math.pi)
    pole_real_hz = -outputs.poles.real / (2.0 * math.pi)
    sorted_imag, _ = torch.sort(pole_imag_hz, dim=1)
    sorted_real, _ = torch.sort(pole_real_hz, dim=1)
    mode_ids = torch.arange(1, mode_count + 1, device=geom_bits.device, dtype=torch.float32).view(1, -1)
    target_modes_hz = fr_hz.unsqueeze(-1) * mode_ids
    target_damping_hz = cfg.pole_damping_ratio * target_modes_hz
    pole_prior_loss = F.mse_loss(sorted_imag[:, :mode_count], target_modes_hz) + F.mse_loss(
        sorted_real[:, :mode_count],
        target_damping_hz,
    )

    if cfg.use_aux_head:
        c0 = 3.0e8
        beta_target = 2.0 * math.pi * fr_hz * torch.sqrt(eps_eff) / c0
        beta_loss = F.mse_loss(outputs.beta_pred, beta_target)
        field_loss = _field_helmholtz_loss(outputs.field_map, outputs.beta_pred)
    else:
        beta_loss = data_loss.new_zeros(())
        field_loss = data_loss.new_zeros(())

    notch_target_f, notch_target_d = _notch_target_stats(target_db, freq_hz)
    notch_pred_f = outputs.notch_params[:, 0, 0]
    notch_pred_d = outputs.notch_params[:, 0, 2] * 30.0
    freq_span = max(float((freq_hz[-1] - freq_hz[0]).item()), 1.0)
    notch_loss = torch.abs(notch_pred_f - notch_target_f).mean() / freq_span + 0.05 * torch.abs(
        notch_pred_d - notch_target_d
    ).mean()

    total = (
        cfg.loss_data * data_loss
        + cfg.loss_passive * passive_loss
        + cfg.loss_pole_prior * pole_prior_loss
        + cfg.loss_beta * beta_loss
        + cfg.loss_field * field_loss
        + cfg.loss_notch * notch_loss
    )
    return LossBreakdown(
        total=total,
        data=data_loss,
        passive=passive_loss,
        pole_prior=pole_prior_loss,
        beta=beta_loss,
        field=field_loss,
        notch=notch_loss,
    )


def build_model(cfg: Config, device: str) -> R6PPoleResidue:
    require_torch("model build")
    return R6PPoleResidue(cfg).to(device)


def evaluate_model(
    model: R6PPoleResidue,
    loader: DataLoader,
    cfg: Config,
    device: str,
    freq_hz: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    totals = {
        "total": 0.0,
        "data": 0.0,
        "passive": 0.0,
        "pole_prior": 0.0,
        "beta": 0.0,
        "field": 0.0,
        "notch": 0.0,
        "mae_db": 0.0,
    }
    count = 0
    with torch.no_grad():
        for geom_bits, material_map, target_db, base_idx in loader:
            del base_idx
            geom_bits = geom_bits.to(device)
            material_map = material_map.to(device)
            target_db = target_db.to(device)
            weights = resonance_weights_db(target_db, sigma_db=cfg.notch_sigma_db)
            outputs = model(geom_bits, material_map, freq_hz)
            losses = compute_losses(outputs, geom_bits, material_map, target_db, weights, freq_hz, cfg)
            batch = geom_bits.size(0)
            totals["total"] += losses.total.item() * batch
            totals["data"] += losses.data.item() * batch
            totals["passive"] += losses.passive.item() * batch
            totals["pole_prior"] += losses.pole_prior.item() * batch
            totals["beta"] += losses.beta.item() * batch
            totals["field"] += losses.field.item() * batch
            totals["notch"] += losses.notch.item() * batch
            totals["mae_db"] += torch.abs(outputs.gamma_db - target_db).mean().item() * batch
            count += batch
    return {key: value / max(1, count) for key, value in totals.items()}


def save_prediction_graphs(
    model: R6PPoleResidue,
    base: ProcessedCurveDataset,
    indices: list[int],
    output_dir: Path,
    split: str,
    plot_count: int,
    device: str,
    freq_hz: torch.Tensor,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if plot_count <= 0:
        return output_dir

    plt = load_pyplot()
    x_values = (freq_hz.detach().cpu().numpy() / 1.0e9).tolist()
    model.eval()
    saved = 0
    with torch.no_grad():
        for idx in indices:
            if saved >= plot_count:
                break
            geom = base.geometry[idx].unsqueeze(0).to(device)
            mat = base.material_map[idx].unsqueeze(0).to(device)
            target = base.curves_db[idx].detach().cpu().numpy()
            pred = model(geom, mat, freq_hz).gamma_db.squeeze(0).detach().cpu().numpy()
            plt.figure(figsize=(8, 4.5))
            plt.plot(x_values, target, label="Target", linewidth=2.0)
            plt.plot(x_values, pred, label="Prediction", linewidth=2.0, linestyle="--")
            plt.xlabel("Frequency (GHz)")
            plt.ylabel("S11 (dB)")
            plt.title(f"{split.upper()} Antenna {base.antenna_ids[idx]}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            out_path = output_dir / f"{split}_antenna_{int(base.antenna_ids[idx]):05d}.png"
            plt.savefig(out_path, dpi=180)
            plt.close()
            saved += 1
    return output_dir


def train_model(cfg: Config) -> dict[str, float]:
    require_torch("train")
    if (
        cfg.overwrite_processed
        or not cfg.processed_csv_path.exists()
        or not cfg.processed_meta_path.exists()
    ):
        preprocess_uploaded_dataset(
            geometry_catalog_path=cfg.geometry_catalog_path,
            curves_root=cfg.curves_root,
            processed_csv_path=cfg.processed_csv_path,
            processed_meta_path=cfg.processed_meta_path,
            overwrite_processed=cfg.overwrite_processed,
            max_antennas=cfg.max_antennas,
            grid_height=GRID_HEIGHT,
            grid_width=GRID_WIDTH,
        )

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg)

    device = get_device(cfg.device)
    base = ProcessedCurveDataset(
        cfg.processed_csv_path,
        cfg.processed_meta_path,
        sigma_db=cfg.notch_sigma_db,
        er_default=cfg.er_default,
    )
    freq_hz = base.freq_axis_hz.to(device)
    train_loader, val_loader, test_loader, train_idx, val_idx, test_idx = build_dataloaders(base, cfg, device)
    model = build_model(cfg, device)
    optimizer = create_optimizer(cfg, model.parameters())
    scaler = build_grad_scaler(device, cfg.use_amp)

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
                "train_data",
                "train_passive",
                "train_pole_prior",
                "train_beta",
                "train_field",
                "train_notch",
                "val_total",
                "val_data",
                "val_passive",
                "val_pole_prior",
                "val_beta",
                "val_field",
                "val_notch",
                "val_mae_db",
            ]
        )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_totals = {
            "total": 0.0,
            "data": 0.0,
            "passive": 0.0,
            "pole_prior": 0.0,
            "beta": 0.0,
            "field": 0.0,
            "notch": 0.0,
        }
        count = 0

        for batch_idx, (geom_bits, material_map, target_db, base_idx) in enumerate(train_loader, start=1):
            del base_idx
            geom_bits = geom_bits.to(device)
            material_map = material_map.to(device)
            target_db = target_db.to(device)
            sample_weights = resonance_weights_db(target_db, sigma_db=cfg.notch_sigma_db)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, cfg.use_amp):
                outputs = model(geom_bits, material_map, freq_hz)
                losses = compute_losses(
                    outputs=outputs,
                    geom_bits=geom_bits,
                    material_map=material_map,
                    target_db=target_db,
                    sample_weights=sample_weights,
                    freq_hz=freq_hz,
                    cfg=cfg,
                )

            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            batch = geom_bits.size(0)
            train_totals["total"] += losses.total.item() * batch
            train_totals["data"] += losses.data.item() * batch
            train_totals["passive"] += losses.passive.item() * batch
            train_totals["pole_prior"] += losses.pole_prior.item() * batch
            train_totals["beta"] += losses.beta.item() * batch
            train_totals["field"] += losses.field.item() * batch
            train_totals["notch"] += losses.notch.item() * batch
            count += batch

            if cfg.log_every_batches > 0 and batch_idx % cfg.log_every_batches == 0:
                running = {key: value / max(1, count) for key, value in train_totals.items()}
                print(
                    "Epoch "
                    f"{epoch:03d} | "
                    f"batch {batch_idx:04d}/{len(train_loader):04d} | "
                    f"loss {running['total']:.6f} | "
                    f"data {running['data']:.6f} | "
                    f"pole {running['pole_prior']:.6f} | "
                    f"notch {running['notch']:.6f}"
                )

        train_metrics = {key: value / max(1, count) for key, value in train_totals.items()}
        val_metrics = evaluate_model(model, val_loader, cfg, device, freq_hz)
        row = {
            "epoch": epoch,
            "train_total": train_metrics["total"],
            "train_data": train_metrics["data"],
            "train_passive": train_metrics["passive"],
            "train_pole_prior": train_metrics["pole_prior"],
            "train_beta": train_metrics["beta"],
            "train_field": train_metrics["field"],
            "train_notch": train_metrics["notch"],
            "val_total": val_metrics["total"],
            "val_data": val_metrics["data"],
            "val_passive": val_metrics["passive"],
            "val_pole_prior": val_metrics["pole_prior"],
            "val_beta": val_metrics["beta"],
            "val_field": val_metrics["field"],
            "val_notch": val_metrics["notch"],
            "val_mae_db": val_metrics["mae_db"],
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
                    row["train_pole_prior"],
                    row["train_beta"],
                    row["train_field"],
                    row["train_notch"],
                    row["val_total"],
                    row["val_data"],
                    row["val_passive"],
                    row["val_pole_prior"],
                    row["val_beta"],
                    row["val_field"],
                    row["val_notch"],
                    row["val_mae_db"],
                ]
            )

        print(
            "Epoch "
            f"{epoch:03d} | "
            f"train {row['train_total']:.6f} | "
            f"val {row['val_total']:.6f} | "
            f"mae_db {row['val_mae_db']:.6f}"
        )

        if row["val_total"] < best_val - 1.0e-6:
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
    test_metrics = evaluate_model(best_model, test_loader, cfg, device, freq_hz)
    graphical_dir = save_prediction_graphs(
        model=best_model,
        base=base,
        indices=test_idx,
        output_dir=cfg.prediction_graphical_dir / "test",
        split="test",
        plot_count=cfg.prediction_plot_count,
        device=device,
        freq_hz=freq_hz,
    )
    plot_loss_curve(history_rows, cfg.loss_plot_path, title="R6P Pole-Residue Training Curve")
    summary = {
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "test_total": test_metrics["total"],
        "test_data": test_metrics["data"],
        "test_passive": test_metrics["passive"],
        "test_pole_prior": test_metrics["pole_prior"],
        "test_beta": test_metrics["beta"],
        "test_field": test_metrics["field"],
        "test_notch": test_metrics["notch"],
        "test_mae_db": test_metrics["mae_db"],
        "num_antennas": len(base),
        "seq_len": base.seq_len,
        "prediction_graphical_dir": str(graphical_dir),
        "prediction_graph_count": min(cfg.prediction_plot_count, len(test_idx)),
        "loss_plot_path": str(cfg.loss_plot_path),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
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
        preprocess_uploaded_dataset(
            geometry_catalog_path=cfg.geometry_catalog_path,
            curves_root=cfg.curves_root,
            processed_csv_path=cfg.processed_csv_path,
            processed_meta_path=cfg.processed_meta_path,
            overwrite_processed=cfg.overwrite_processed,
            max_antennas=cfg.max_antennas,
            grid_height=GRID_HEIGHT,
            grid_width=GRID_WIDTH,
        )
    device = get_device(cfg.device)
    base = ProcessedCurveDataset(
        cfg.processed_csv_path,
        cfg.processed_meta_path,
        sigma_db=cfg.notch_sigma_db,
        er_default=cfg.er_default,
    )
    freq_hz = base.freq_axis_hz.to(device)
    _train_loader, val_loader, test_loader, _train_idx, val_idx, test_idx = build_dataloaders(base, cfg, device)
    loader = val_loader if split == "val" else test_loader
    indices = val_idx if split == "val" else test_idx
    model = build_model(cfg, device)
    load_model_weights(model, cfg.checkpoint_path, device)
    metrics = evaluate_model(model, loader, cfg, device, freq_hz)
    graphical_dir = save_prediction_graphs(
        model=model,
        base=base,
        indices=indices,
        output_dir=cfg.prediction_graphical_dir / split,
        split=split,
        plot_count=cfg.prediction_plot_count,
        device=device,
        freq_hz=freq_hz,
    )
    metrics["split"] = split
    metrics["checkpoint"] = str(cfg.checkpoint_path)
    metrics["graphical_dir"] = str(graphical_dir)
    metrics["prediction_graph_count"] = min(cfg.prediction_plot_count, len(indices))
    return metrics


def inspect_processed_dataset(cfg: Config) -> None:
    require_torch("inspect")
    base = ProcessedCurveDataset(
        cfg.processed_csv_path,
        cfg.processed_meta_path,
        sigma_db=cfg.notch_sigma_db,
        er_default=cfg.er_default,
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
        description="R6P pole-residue + notch + auxiliary physics model."
    )
    parser.add_argument("command", choices=["preprocess", "inspect", "train"], nargs="?", default=cfg.command)
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
    parser.add_argument("--num-poles", type=int, default=cfg.num_poles)
    parser.add_argument("--num-notches", type=int, default=cfg.num_notches)
    parser.add_argument("--loss-pole-prior", type=float, default=cfg.loss_pole_prior)
    parser.add_argument("--loss-beta", type=float, default=cfg.loss_beta)
    parser.add_argument("--loss-field", type=float, default=cfg.loss_field)
    parser.add_argument("--loss-notch", type=float, default=cfg.loss_notch)
    parser.add_argument("--disable-aux-head", action="store_true")
    parser.add_argument("--disable-material-channel", action="store_true")
    parser.add_argument("--er-default", type=float, default=cfg.er_default)
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
    cfg.num_poles = max(2, args.num_poles)
    cfg.num_notches = max(1, args.num_notches)
    cfg.loss_pole_prior = max(0.0, args.loss_pole_prior)
    cfg.loss_beta = max(0.0, args.loss_beta)
    cfg.loss_field = max(0.0, args.loss_field)
    cfg.loss_notch = max(0.0, args.loss_notch)
    cfg.use_aux_head = not args.disable_aux_head
    cfg.use_material_channel = not args.disable_material_channel
    cfg.er_default = float(np.clip(args.er_default, cfg.er_min, cfg.er_max))
    cfg.overwrite_processed = args.overwrite_processed
    return cfg


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.random_seed)

    if cfg.command == "preprocess":
        summary = preprocess_uploaded_dataset(
            geometry_catalog_path=cfg.geometry_catalog_path,
            curves_root=cfg.curves_root,
            processed_csv_path=cfg.processed_csv_path,
            processed_meta_path=cfg.processed_meta_path,
            overwrite_processed=cfg.overwrite_processed,
            max_antennas=cfg.max_antennas,
            grid_height=GRID_HEIGHT,
            grid_width=GRID_WIDTH,
        )
        compact = {
            "geometry_catalog_path": summary["geometry_catalog_path"],
            "curves_root": summary["curves_root"],
            "processed_csv_path": summary["processed_csv_path"],
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
            preprocess_uploaded_dataset(
                geometry_catalog_path=cfg.geometry_catalog_path,
                curves_root=cfg.curves_root,
                processed_csv_path=cfg.processed_csv_path,
                processed_meta_path=cfg.processed_meta_path,
                overwrite_processed=cfg.overwrite_processed,
                max_antennas=cfg.max_antennas,
                grid_height=GRID_HEIGHT,
                grid_width=GRID_WIDTH,
            )
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
    print(f"Loss curve saved to: {summary['loss_plot_path']}")
    print("Training summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
