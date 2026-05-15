"""
R6O: multi-task notch-oriented model (no physics-induced losses).

Architecture:
- Shared encoder from geometry (and optional material channel).
- Branch 1: fixed high-resolution S11(dB) reconstruction.
- Branch 2: resonance-frequency regression.

Loss:
- Curve MAE loss. If the dataset target has fewer points than the model output,
  the model output is linearly resampled to the target length for loss only.
- Feature loss.
- Total = curve_loss + feature_weight * feature_loss.
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
from utils.interpolation import maybe_resample_curve_matrix

GRID_HEIGHT = 10
GRID_WIDTH = 10
NUM_CELLS = GRID_HEIGHT * GRID_WIDTH


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


def _default_train_index_path() -> Path:
    return _default_processed_dir() / "train.txt"


def _default_val_index_path() -> Path:
    return _default_processed_dir() / "val.txt"


def _default_labels_csv_path() -> Path:
    return _default_processed_dir() / "resonance_labels.csv"


def _default_results_dir() -> Path:
    return REPO_ROOT / "results" / "R6O_multitask"


@dataclass
class Config:
    command: str = "train"
    geometry_catalog_path: Path = field(default_factory=_default_geometry_catalog_path)
    curves_root: Path = RAW_DATA_ROOT
    processed_csv_path: Path = field(default_factory=_default_processed_csv)
    processed_meta_path: Path = field(default_factory=_default_processed_meta)
    train_index_path: Path = field(default_factory=_default_train_index_path)
    val_index_path: Path = field(default_factory=_default_val_index_path)
    labels_csv_path: Path = field(default_factory=_default_labels_csv_path)
    results_dir: Path = field(default_factory=_default_results_dir)
    batch_size: int = 128
    lr: float = 1.0e-3
    optimizer_name: str = "adamw"
    weight_decay: float = 1.0e-4
    epochs: int = 120
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    encoder_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.10
    use_material_channel: bool = False
    use_feed_channel: bool = False
    use_index_files: bool = True
    use_notch_label_file: bool = True
    label_min_confidence: float = 0.0
    use_label_fallbacks: bool = True
    boundary_antenna_id: int = 30002
    boundary_loss_weight: float = 0.05
    er_default: float = 4.4
    er_min: float = 1.0
    er_max: float = 12.0
    loss_curve_weight: float = 1.0
    loss_feature_weight: float = 10.0
    loss_focus_alpha: float = 2.0
    loss_focus_bw_norm: float = 0.03
    feature_finetune_epochs: int = 0
    feature_finetune_min_confidence: float = 0.35
    feature_finetune_lr: float = 1.0e-4
    patience: int = 18
    gradient_clip: float = 1.0
    log_every_batches: int = 10
    use_amp: bool = True
    random_seed: int = 42
    max_antennas: int = 0
    device: str = "auto"
    eval_split: str = ""
    prediction_plot_count: int = 12
    export_curve_points: int = 501
    target_curve_points: int = 0
    output_curve_points: int = 501
    # Deprecated compatibility args. R6O now uses one 501-point output head.
    curve_head_points: str = ""
    active_curve_points: int = 0
    export_head_points: int = 0
    init_checkpoint_path: Path | None = None
    strict_init: bool = False
    overwrite_processed: bool = False
    weights_filename: str = "r6o_best.pt"
    checkpoint_filename: str = "r6o_best.ckpt"
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
    curve_mae: torch.Tensor
    feature_loss: torch.Tensor
    feature_mae_norm: torch.Tensor


@dataclass(frozen=True)
class EvalOutputs:
    curve_db: torch.Tensor
    features: torch.Tensor  # [f_resonance_ghz_norm]


def _interpolate_curve_1d(curve: torch.Tensor, target_points: int) -> torch.Tensor:
    if target_points <= 0 or curve.numel() == target_points:
        return curve
    if curve.ndim != 1:
        raise ValueError("Expected a 1D curve tensor for interpolation.")
    src = curve.view(1, 1, -1)
    out = F.interpolate(src, size=target_points, mode="linear", align_corners=True)
    return out.view(-1)


def _resample_curve_batch(curves: torch.Tensor, target_points: int) -> torch.Tensor:
    if target_points <= 0 or curves.shape[-1] == target_points:
        return curves
    if curves.ndim != 2:
        raise ValueError("Expected a [batch, points] curve tensor for interpolation.")
    return F.interpolate(
        curves.unsqueeze(1),
        size=target_points,
        mode="linear",
        align_corners=True,
    ).squeeze(1)


def _nonpositive_db(raw_db: torch.Tensor) -> torch.Tensor:
    # Smooth min(raw_db, 0). This preserves negative dB values while preventing
    # non-physical positive S11(dB) predictions.
    return -F.softplus(-raw_db)


def _build_export_freq_axis(freq_axis_ghz: torch.Tensor, target_points: int) -> torch.Tensor:
    if target_points <= 0 or freq_axis_ghz.numel() == target_points:
        return freq_axis_ghz
    start = float(freq_axis_ghz[0].item())
    end = float(freq_axis_ghz[-1].item())
    return torch.linspace(start, end, target_points, dtype=freq_axis_ghz.dtype, device=freq_axis_ghz.device)


class ProcessedCurveDataset:
    def __init__(
        self,
        csv_path: Path,
        meta_path: Path,
        er_default: float,
        target_curve_points: int = 0,
    ) -> None:
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

        # Drop rows whose curve targets contain NaN/Inf — they poison MAE
        # loss and propagate NaN through gradients, corrupting weights for
        # all subsequent batches.
        curves_np = values[:, NUM_CELLS:]
        bad_mask = ~np.isfinite(curves_np).all(axis=1)
        num_bad = int(bad_mask.sum())
        if num_bad > 0:
            print(
                f"ProcessedCurveDataset: dropping {num_bad} of {curves_np.shape[0]} "
                f"rows with NaN/Inf curve values from {csv_path.name}."
            )
            keep_mask = ~bad_mask
            values = values[keep_mask]
            curves_np = curves_np[keep_mask]
        else:
            keep_mask = np.ones(curves_np.shape[0], dtype=bool)

        freq_axis_ghz = meta.get("freq_axis_ghz", inferred_freq_axis)
        if len(freq_axis_ghz) != self.seq_len:
            freq_axis_ghz = inferred_freq_axis
        curves_np, freq_axis_ghz = maybe_resample_curve_matrix(
            curves_np,
            np.asarray(freq_axis_ghz, dtype=np.float32),
            target_curve_points,
        )
        self.seq_len = int(curves_np.shape[1])

        self.geometry = torch.from_numpy((values[:, :NUM_CELLS] > 0.5).astype(np.float32))
        self.curves_db = torch.from_numpy(curves_np)
        self.freq_axis_ghz = torch.tensor(freq_axis_ghz, dtype=torch.float32)
        raw_ids = meta.get(
            "matched_antenna_ids",
            list(range(1, keep_mask.shape[0] + 1)),
        )
        if len(raw_ids) == keep_mask.shape[0]:
            self.antenna_ids = [aid for aid, keep in zip(raw_ids, keep_mask.tolist()) if keep]
        else:
            self.antenna_ids = list(range(1, self.geometry.size(0) + 1))
        self.material_map = torch.full(
            (self.geometry.size(0), NUM_CELLS),
            float(er_default),
            dtype=torch.float32,
        )
        # Binary feed-structure mask: 1 at cells listed in meta feed_cells, 0 elsewhere.
        # Broadcast across all samples as a fixed indicator channel.
        feed_cells = meta.get("feed_cells", [])
        feed_vec = torch.zeros(NUM_CELLS, dtype=torch.float32)
        for c in feed_cells:
            if 0 <= c < NUM_CELLS:
                feed_vec[c] = 1.0
        self.feed_mask = feed_vec.unsqueeze(0).expand(self.geometry.size(0), -1)

    def __len__(self) -> int:
        return self.geometry.size(0)


class AntennaCurveDataset(Dataset):
    def __init__(self, base: ProcessedCurveDataset, indices: list[int]) -> None:
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        base_idx = self.indices[idx]
        return (
            self.base.geometry[base_idx],
            self.base.material_map[base_idx],
            self.base.feed_mask[base_idx],
            self.base.curves_db[base_idx],
            base_idx,
        )


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


def load_compatible_model_weights(
    model: nn.Module,
    checkpoint_path: Path,
    device: str,
    *,
    strict: bool = False,
) -> dict[str, object]:
    require_torch("checkpoint init")
    payload = torch.load(checkpoint_path, map_location=device)
    source_state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    if strict:
        model.load_state_dict(source_state)
        return payload if isinstance(payload, dict) else {}

    target_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in source_state.items():
        target_key = key
        if key.startswith("curve_heads."):
            # Backward compatibility for the removed multi-head variant:
            # only import the head that matches the fixed output resolution.
            parts = key.split(".", 2)
            if len(parts) == 3 and parts[1] == str(getattr(model, "output_curve_points", "")):
                target_key = f"curve_head.{parts[2]}"
            else:
                skipped.append(key)
                continue
        elif key.startswith("curve_head."):
            # Backward compatibility for older single-head checkpoints.
            suffix = key.removeprefix("curve_head.")
            candidates = [candidate for candidate in target_state if candidate.endswith(f".{suffix}")]
            target_key = candidates[0] if len(candidates) == 1 else key
        if target_key in target_state and target_state[target_key].shape == value.shape:
            compatible[target_key] = value
        else:
            skipped.append(key)

    merged = dict(target_state)
    merged.update(compatible)
    model.load_state_dict(merged)
    print(
        f"Initialized from {checkpoint_path}: "
        f"loaded {len(compatible)} tensors, skipped {len(skipped)} tensors."
    )
    return payload if isinstance(payload, dict) else {}


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


def _load_index_file(path: Path, antenna_id_to_row: dict[int, int]) -> list[int]:
    if not path.exists():
        return []
    indices: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            if line_idx == 0 and ("antenna_id" in line.lower()):
                continue
            token = line.split("\t")[0].split(",")[0].strip()
            if not token:
                continue
            try:
                antenna_id = int(token)
            except ValueError:
                continue
            if antenna_id in antenna_id_to_row:
                indices.append(antenna_id_to_row[antenna_id])
    # keep order and unique
    seen = set()
    unique_indices: list[int] = []
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        unique_indices.append(idx)
    return unique_indices


def _split_from_index_files(
    base: ProcessedCurveDataset,
    cfg: Config,
) -> tuple[list[int], list[int], list[int]]:
    antenna_id_to_row = {int(aid): row for row, aid in enumerate(base.antenna_ids)}
    train_idx = _load_index_file(cfg.train_index_path, antenna_id_to_row)
    val_idx = _load_index_file(cfg.val_index_path, antenna_id_to_row)
    if not train_idx or not val_idx:
        return [], [], []
    all_idx = set(range(len(base)))
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_idx = sorted(all_idx - train_set - val_set)
    return train_idx, val_idx, test_idx


def build_label_feature_targets(
    base: ProcessedCurveDataset,
    labels_csv_path: Path,
    min_confidence: float = 0.0,
    use_fallbacks: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (targets, weights) where weights = confidence in [0,1]; 0 means excluded."""
    freq_axis = base.freq_axis_ghz
    freq_span = max(float((freq_axis[-1] - freq_axis[0]).item()), 1.0e-6)
    targets = torch.zeros((len(base), 1), dtype=torch.float32)
    weights = torch.zeros((len(base),), dtype=torch.float32)

    antenna_id_to_row = {int(aid): row for row, aid in enumerate(base.antenna_ids)}
    if labels_csv_path.exists():
        with labels_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    antenna_id = int(row["antenna_id"])
                    notch_freq_ghz = float(row["resonant_freq_ghz"])
                    confidence = float(row.get("confidence", "1.0") or 1.0)
                except (KeyError, ValueError):
                    continue
                if not (math.isfinite(notch_freq_ghz) and math.isfinite(confidence)):
                    continue
                valid_token = str(row.get("is_valid_local_min", "1")).strip().lower()
                is_valid_local = valid_token in {"1", "true", "yes", "y"}
                method = str(row.get("method", ""))
                if not use_fallbacks and (not is_valid_local or method == "global_min_fallback"):
                    continue
                if confidence < min_confidence:
                    continue
                if antenna_id not in antenna_id_to_row:
                    continue
                row_idx = antenna_id_to_row[antenna_id]
                targets[row_idx, 0] = float(
                    np.clip((notch_freq_ghz - float(freq_axis[0].item())) / freq_span, 0.0, 1.0)
                )
                weights[row_idx] = float(np.clip(confidence, 0.0, 1.0))
    return targets, weights


def build_dataloaders(
    base: ProcessedCurveDataset,
    cfg: Config,
    device: str,
) -> tuple[DataLoader, DataLoader, DataLoader, list[int], list[int], list[int]]:
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    if cfg.use_index_files:
        train_idx, val_idx, test_idx = _split_from_index_files(base, cfg)
    if not train_idx or not val_idx:
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


def resolve_output_curve_points(cfg: Config, native_seq_len: int) -> int:
    if cfg.output_curve_points > 0:
        return int(cfg.output_curve_points)
    if cfg.export_curve_points > 0:
        return int(cfg.export_curve_points)
    return int(native_seq_len)


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


class R6OMultiTask(nn.Module):
    def __init__(self, cfg: Config, seq_len: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.seq_len = seq_len
        self.output_curve_points = resolve_output_curve_points(cfg, seq_len)
        in_channels = 1 + int(cfg.use_material_channel) + int(cfg.use_feed_channel)
        hidden = cfg.hidden_dim

        self.encoder = SharedConvEncoder(
            in_channels=in_channels,
            encoder_dim=cfg.encoder_dim,
            dropout=cfg.dropout,
        )
        self.curve_head = self._make_curve_head(hidden, self.output_curve_points)
        self.feature_head = nn.Sequential(
            nn.Linear(cfg.encoder_dim, hidden // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden // 2, 1),
        )

    def _make_curve_head(self, hidden: int, output_points: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.cfg.encoder_dim, hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(hidden, output_points),
        )

    def set_active_curve_points(self, output_points: int) -> None:
        # Kept only so older launch commands do not crash. The model has one
        # fixed output head; loss-time resampling handles 61-point targets.
        if int(output_points) != self.output_curve_points:
            print(
                "R6O ignores active curve head changes; "
                f"using fixed {self.output_curve_points}-point output."
            )

    def _build_input(
        self,
        geom_bits: torch.Tensor,
        material_map: torch.Tensor,
        feed_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        geom = geom_bits.view(-1, 1, GRID_HEIGHT, GRID_WIDTH)
        channels = [geom]
        if self.cfg.use_material_channel:
            er = material_map.view(-1, 1, GRID_HEIGHT, GRID_WIDTH)
            er = (er - self.cfg.er_min) / max(self.cfg.er_max - self.cfg.er_min, 1.0e-6)
            channels.append(er.clamp(0.0, 1.0))
        if self.cfg.use_feed_channel:
            if feed_mask is not None:
                fm = feed_mask.view(-1, 1, GRID_HEIGHT, GRID_WIDTH).float()
            else:
                fm = torch.zeros_like(geom)
            channels.append(fm)
        return torch.cat(channels, dim=1) if len(channels) > 1 else channels[0]

    def forward(
        self,
        geom_bits: torch.Tensor,
        material_map: torch.Tensor,
        curve_points: int | None = None,
        feed_mask: torch.Tensor | None = None,
    ) -> EvalOutputs:
        del curve_points
        x = self._build_input(geom_bits, material_map, feed_mask)
        emb = self.encoder(x)
        curve_db = _nonpositive_db(self.curve_head(emb))
        feature_raw = torch.sigmoid(self.feature_head(emb))
        return EvalOutputs(curve_db=curve_db, features=feature_raw)


def _adaptive_resonance_loss(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    alpha: float,
    bw_norm: float,
) -> "torch.Tensor":
    """Gaussian-amplified MSE: higher penalty when close-but-not-exact.

    focus(err) = 1 + alpha * exp(-err^2 / (2 * bw_norm^2))
    loss = focus * err^2

    Far from target: focus~1 (standard MSE, pulls prediction in).
    Within ~1 bandwidth of target: focus up to (1+alpha), amplified to
    force precision — because a small frequency miss still fails the dip.
    """
    err = pred - target
    focus = 1.0 + alpha * torch.exp(-err.pow(2) / (2.0 * max(bw_norm, 1e-6) ** 2))
    return focus * err.pow(2)


def compute_losses(
    outputs: EvalOutputs,
    target_db: torch.Tensor,
    cfg: Config,
    feature_targets: torch.Tensor | None = None,
    feature_weights: torch.Tensor | None = None,
) -> LossBreakdown:
    curve_pred_for_loss = _resample_curve_batch(outputs.curve_db, target_db.shape[-1])
    curve_mae = torch.abs(curve_pred_for_loss - target_db).mean()
    if feature_targets is None:
        feature_loss = curve_mae.new_zeros(())
        feature_mae_norm = curve_mae.new_zeros(())
    else:
        if feature_weights is None or feature_weights.sum() < 1e-8:
            feature_loss = curve_mae.new_zeros(())
            feature_mae_norm = curve_mae.new_zeros(())
        else:
            w = feature_weights.unsqueeze(-1)  # [B, 1]
            w_sum = w.sum().clamp(min=1e-8)
            per_sample = _adaptive_resonance_loss(
                outputs.features, feature_targets,
                alpha=cfg.loss_focus_alpha,
                bw_norm=cfg.loss_focus_bw_norm,
            )
            feature_loss = (w * per_sample).sum() / w_sum
            feature_mae_norm = (feature_weights * (outputs.features - feature_targets).abs().squeeze(-1)).sum() / w_sum

    total = cfg.loss_curve_weight * curve_mae + cfg.loss_feature_weight * feature_loss
    return LossBreakdown(
        total=total,
        curve_mae=curve_mae,
        feature_loss=feature_loss,
        feature_mae_norm=feature_mae_norm,
    )


def build_model(cfg: Config, device: str, seq_len: int) -> R6OMultiTask:
    require_torch("model build")
    return R6OMultiTask(cfg, seq_len=seq_len).to(device)


def evaluate_model(
    model: R6OMultiTask,
    loader: DataLoader,
    cfg: Config,
    device: str,
    all_feature_targets: torch.Tensor | None,
    all_feature_weights: torch.Tensor | None,
) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "curve_mae": 0.0, "feature_loss": 0.0, "feature_mae_norm": 0.0}
    pred_max_db = float("-inf")
    pred_min_db = float("inf")
    count = 0
    with torch.no_grad():
        for geom_bits, material_map, feed_mask, target_db, base_idx in loader:
            geom_bits = geom_bits.to(device)
            material_map = material_map.to(device)
            feed_mask = feed_mask.to(device)
            target_db = target_db.to(device)
            if all_feature_targets is not None:
                batch_feature_targets = all_feature_targets[base_idx].to(device)
                batch_feature_weights = (
                    all_feature_weights[base_idx].to(device)
                    if all_feature_weights is not None
                    else None
                )
            else:
                batch_feature_targets = None
                batch_feature_weights = None
            outputs = model(geom_bits, material_map, feed_mask=feed_mask)
            losses = compute_losses(
                outputs,
                target_db,
                cfg,
                feature_targets=batch_feature_targets,
                feature_weights=batch_feature_weights,
            )
            batch = geom_bits.size(0)
            totals["total"] += losses.total.item() * batch
            totals["curve_mae"] += losses.curve_mae.item() * batch
            totals["feature_loss"] += losses.feature_loss.item() * batch
            totals["feature_mae_norm"] += losses.feature_mae_norm.item() * batch
            pred_max_db = max(pred_max_db, float(outputs.curve_db.max().item()))
            pred_min_db = min(pred_min_db, float(outputs.curve_db.min().item()))
            count += batch
    metrics = {key: value / max(1, count) for key, value in totals.items()}
    metrics["pred_max_db"] = pred_max_db if count else 0.0
    metrics["pred_min_db"] = pred_min_db if count else 0.0
    return metrics


def boundary_indices(base: ProcessedCurveDataset, antenna_id: int) -> list[int]:
    if antenna_id <= 0:
        return []
    return [idx for idx, aid in enumerate(base.antenna_ids) if int(aid) == int(antenna_id)]


def boundary_anchor_loss(
    model: R6OMultiTask,
    base: ProcessedCurveDataset,
    indices: list[int],
    cfg: Config,
    device: str,
    all_feature_targets: torch.Tensor | None,
    all_feature_weights: torch.Tensor | None,
) -> torch.Tensor:
    if not indices or cfg.boundary_loss_weight <= 0.0:
        return next(model.parameters()).new_zeros(())
    idx = indices[0]
    geom = base.geometry[idx].unsqueeze(0).to(device)
    mat = base.material_map[idx].unsqueeze(0).to(device)
    fm = base.feed_mask[idx].unsqueeze(0).to(device)
    target = base.curves_db[idx].unsqueeze(0).to(device)
    if all_feature_targets is not None and all_feature_weights is not None:
        feature_targets = all_feature_targets[idx : idx + 1].to(device)
        feature_weights = all_feature_weights[idx : idx + 1].to(device)
    else:
        feature_targets = None
        feature_weights = None
    outputs = model(geom, mat, feed_mask=fm)
    losses = compute_losses(
        outputs,
        target,
        cfg,
        feature_targets=feature_targets,
        feature_weights=feature_weights,
    )
    return losses.total


def evaluate_boundary_condition(
    model: R6OMultiTask,
    base: ProcessedCurveDataset,
    indices: list[int],
    cfg: Config,
    device: str,
    all_feature_targets: torch.Tensor | None,
    all_feature_weights: torch.Tensor | None,
) -> dict[str, float]:
    if not indices:
        return {}
    idx = indices[0]
    freq_axis = base.freq_axis_ghz
    freq_start = float(freq_axis[0].item())
    freq_span = max(float((freq_axis[-1] - freq_axis[0]).item()), 1.0e-6)
    model.eval()
    with torch.no_grad():
        geom = base.geometry[idx].unsqueeze(0).to(device)
        mat = base.material_map[idx].unsqueeze(0).to(device)
        fm = base.feed_mask[idx].unsqueeze(0).to(device)
        target = base.curves_db[idx].unsqueeze(0).to(device)
        outputs = model(geom, mat, feed_mask=fm)
        curve_mae = torch.abs(outputs.curve_db - target).mean().item()
        pred_max_db = outputs.curve_db.max().item()
        pred_min_db = outputs.curve_db.min().item()
        target_min_db = target.min().item()
        pred_feature_norm = float(outputs.features[0, 0].clamp(0.0, 1.0).item())
        pred_res_ghz = freq_start + pred_feature_norm * freq_span
        if all_feature_targets is not None and all_feature_weights is not None and bool(all_feature_weights[idx].item()):
            target_feature_norm = float(all_feature_targets[idx, 0].item())
        else:
            target_feature_norm = float(target.argmin(dim=1).item()) / max(target.shape[-1] - 1, 1)
        target_res_ghz = freq_start + target_feature_norm * freq_span
    return {
        "boundary_antenna_id": float(base.antenna_ids[idx]),
        "boundary_curve_mae": float(curve_mae),
        "boundary_pred_max_db": float(pred_max_db),
        "boundary_pred_min_db": float(pred_min_db),
        "boundary_target_min_db": float(target_min_db),
        "boundary_pred_resonance_ghz": float(pred_res_ghz),
        "boundary_target_resonance_ghz": float(target_res_ghz),
        "boundary_feature_mae_ghz": float(abs(pred_res_ghz - target_res_ghz)),
    }


def save_prediction_graphs(
    model: R6OMultiTask,
    base: ProcessedCurveDataset,
    indices: list[int],
    output_dir: Path,
    split: str,
    plot_count: int,
    device: str,
    freq_axis_ghz: torch.Tensor,
    export_curve_points: int,
    export_head_points: int = 0,
) -> Path:
    del export_head_points
    output_dir.mkdir(parents=True, exist_ok=True)
    if plot_count <= 0:
        return output_dir

    plt = load_pyplot()
    export_freq_axis = _build_export_freq_axis(freq_axis_ghz, export_curve_points)
    x_values = export_freq_axis.detach().cpu().numpy().tolist()
    model.eval()
    saved = 0
    with torch.no_grad():
        for idx in indices:
            if saved >= plot_count:
                break
            geom = base.geometry[idx].unsqueeze(0).to(device)
            mat = base.material_map[idx].unsqueeze(0).to(device)
            fm = base.feed_mask[idx].unsqueeze(0).to(device)
            target_tensor = _interpolate_curve_1d(base.curves_db[idx].detach(), export_curve_points)
            pred_tensor = _interpolate_curve_1d(
                model(geom, mat, feed_mask=fm).curve_db.squeeze(0).detach().cpu(),
                export_curve_points,
            )
            target = target_tensor.numpy()
            pred = pred_tensor.numpy()
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


def _run_feature_finetune(
    model: "R6OMultiTask",
    base: "ProcessedCurveDataset",
    train_idx: list[int],
    all_feature_targets: "torch.Tensor",
    all_feature_weights: "torch.Tensor",
    cfg: Config,
    device: str,
) -> None:
    """Freeze encoder, train feature head only on high-confidence samples."""
    min_conf = cfg.feature_finetune_min_confidence
    ft_indices = [i for i in train_idx if float(all_feature_weights[i].item()) >= min_conf]
    if not ft_indices:
        print(f"Feature fine-tune: no samples with confidence >= {min_conf:.2f}; skipping.")
        return
    print(
        f"Feature fine-tune: {len(ft_indices)} high-conf samples "
        f"(conf >= {min_conf:.2f}), {cfg.feature_finetune_epochs} epochs, "
        f"lr={cfg.feature_finetune_lr}"
    )
    for param in model.encoder.parameters():
        param.requires_grad = False
    ft_loader = DataLoader(
        AntennaCurveDataset(base, ft_indices),
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )
    ft_optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.feature_finetune_lr,
    )
    freq_span = max(float((base.freq_axis_ghz[-1] - base.freq_axis_ghz[0]).item()), 1.0e-6)
    for ft_epoch in range(1, cfg.feature_finetune_epochs + 1):
        model.train()
        total_loss = 0.0
        total_mae_norm = 0.0
        count = 0
        n_batches = 0
        for geom_bits, material_map, feed_mask, _target_db, base_idx in ft_loader:
            geom_bits = geom_bits.to(device)
            material_map = material_map.to(device)
            feed_mask = feed_mask.to(device)
            w = all_feature_weights[base_idx].to(device)
            tgt = all_feature_targets[base_idx].to(device)
            ft_optimizer.zero_grad(set_to_none=True)
            outputs = model(geom_bits, material_map, feed_mask=feed_mask)
            w_sum = w.unsqueeze(-1).sum().clamp(min=1e-8)
            per_sample = _adaptive_resonance_loss(
                outputs.features, tgt,
                alpha=cfg.loss_focus_alpha,
                bw_norm=cfg.loss_focus_bw_norm,
            )
            loss = (w.unsqueeze(-1) * per_sample).sum() / w_sum
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            ft_optimizer.step()
            batch = geom_bits.size(0)
            total_loss += loss.item() * batch
            mae_norm = (w * (outputs.features.detach() - tgt).abs().squeeze(-1)).sum().item() / float(w_sum.item())
            total_mae_norm += mae_norm
            count += batch
            n_batches += 1
        avg_loss = total_loss / max(1, count)
        avg_mae_norm = total_mae_norm / max(1, n_batches)
        print(
            f"  FT {ft_epoch:03d}/{cfg.feature_finetune_epochs} | "
            f"feat_loss {avg_loss:.6f} | "
            f"feat_mae {avg_mae_norm * freq_span:.4f} GHz"
        )
    for param in model.encoder.parameters():
        param.requires_grad = True
    print("Feature fine-tune complete.")


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
        er_default=cfg.er_default,
        target_curve_points=cfg.target_curve_points,
    )
    all_feature_targets: torch.Tensor | None = None
    all_feature_weights: torch.Tensor | None = None
    if cfg.use_notch_label_file:
        all_feature_targets, all_feature_weights = build_label_feature_targets(
            base,
            cfg.labels_csv_path,
            min_confidence=cfg.label_min_confidence,
            use_fallbacks=cfg.use_label_fallbacks,
        )
    boundary_idx = boundary_indices(base, cfg.boundary_antenna_id)

    freq_axis_ghz = base.freq_axis_ghz.to(device)
    train_loader, val_loader, test_loader, train_idx, val_idx, test_idx = build_dataloaders(base, cfg, device)
    model = build_model(cfg, device, seq_len=base.seq_len)
    if cfg.init_checkpoint_path is not None and cfg.init_checkpoint_path.exists():
        load_compatible_model_weights(
            model,
            cfg.init_checkpoint_path,
            device,
            strict=cfg.strict_init,
        )
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
                "train_curve_mae",
                "train_feature_loss",
                "train_feature_mae_norm",
                "train_boundary_loss",
                "val_total",
                "val_curve_mae",
                "val_feature_loss",
                "val_feature_mae_norm",
            ]
        )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_totals = {
            "total": 0.0,
            "curve_mae": 0.0,
            "feature_loss": 0.0,
            "feature_mae_norm": 0.0,
            "boundary_loss": 0.0,
        }
        count = 0
        for batch_idx, (geom_bits, material_map, feed_mask, target_db, base_idx) in enumerate(train_loader, start=1):
            geom_bits = geom_bits.to(device)
            material_map = material_map.to(device)
            feed_mask = feed_mask.to(device)
            target_db = target_db.to(device)
            if all_feature_targets is not None:
                batch_feature_targets = all_feature_targets[base_idx].to(device)
                batch_feature_weights = (
                    all_feature_weights[base_idx].to(device)
                    if all_feature_weights is not None
                    else None
                )
            else:
                batch_feature_targets = None
                batch_feature_weights = None

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, cfg.use_amp):
                outputs = model(geom_bits, material_map, feed_mask=feed_mask)
                losses = compute_losses(
                    outputs,
                    target_db,
                    cfg,
                    feature_targets=batch_feature_targets,
                    feature_weights=batch_feature_weights,
                )
                anchor = boundary_anchor_loss(
                    model,
                    base,
                    boundary_idx,
                    cfg,
                    device,
                    all_feature_targets,
                    all_feature_weights,
                )
                total_loss = losses.total + cfg.boundary_loss_weight * anchor

            # Guard: never backprop a NaN/Inf loss; otherwise a single corrupt
            # batch permanently poisons every parameter via Adam's running
            # moment estimates.
            if not torch.isfinite(total_loss):
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            batch = geom_bits.size(0)
            train_totals["total"] += total_loss.item() * batch
            train_totals["curve_mae"] += losses.curve_mae.item() * batch
            train_totals["feature_loss"] += losses.feature_loss.item() * batch
            train_totals["feature_mae_norm"] += losses.feature_mae_norm.item() * batch
            train_totals["boundary_loss"] += anchor.item() * batch
            count += batch

            if cfg.log_every_batches > 0 and batch_idx % cfg.log_every_batches == 0:
                running = {key: value / max(1, count) for key, value in train_totals.items()}
                print(
                    "Epoch "
                    f"{epoch:03d} | "
                    f"batch {batch_idx:04d}/{len(train_loader):04d} | "
                    f"loss {running['total']:.6f} | "
                    f"mae {running['curve_mae']:.6f} | "
                    f"feat {running['feature_loss']:.6f} | "
                    f"feat_mae {running['feature_mae_norm']:.4f} | "
                    f"bc {running['boundary_loss']:.6f}"
                )

        train_metrics = {key: value / max(1, count) for key, value in train_totals.items()}
        val_metrics = evaluate_model(
            model,
            val_loader,
            cfg,
            device,
            all_feature_targets=all_feature_targets,
            all_feature_weights=all_feature_weights,
        )
        row = {
            "epoch": epoch,
            "train_total": train_metrics["total"],
            "train_curve_mae": train_metrics["curve_mae"],
            "train_feature_loss": train_metrics["feature_loss"],
            "train_feature_mae_norm": train_metrics["feature_mae_norm"],
            "train_boundary_loss": train_metrics["boundary_loss"],
            "val_total": val_metrics["total"],
            "val_curve_mae": val_metrics["curve_mae"],
            "val_feature_loss": val_metrics["feature_loss"],
            "val_feature_mae_norm": val_metrics["feature_mae_norm"],
        }
        history_rows.append(row)
        with cfg.history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    row["epoch"],
                    row["train_total"],
                    row["train_curve_mae"],
                    row["train_feature_loss"],
                    row["train_feature_mae_norm"],
                    row["train_boundary_loss"],
                    row["val_total"],
                    row["val_curve_mae"],
                    row["val_feature_loss"],
                    row["val_feature_mae_norm"],
                ]
            )

        print(
            "Epoch "
            f"{epoch:03d} | "
            f"train {row['train_total']:.6f} | "
            f"val {row['val_total']:.6f} | "
            f"mae {row['val_curve_mae']:.6f} | "
            f"feat {row['val_feature_loss']:.6f} | "
            f"feat_mae {row['val_feature_mae_norm']:.4f} | "
            f"bc {row['train_boundary_loss']:.6f}"
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

    best_model = build_model(cfg, device, seq_len=base.seq_len)
    load_model_weights(best_model, cfg.checkpoint_path, device)

    if cfg.feature_finetune_epochs > 0 and all_feature_targets is not None and all_feature_weights is not None:
        _run_feature_finetune(
            best_model, base, train_idx,
            all_feature_targets, all_feature_weights,
            cfg, device,
        )
        save_checkpoint(cfg, best_model, optimizer, best_epoch, best_val)

    test_metrics = evaluate_model(
        best_model,
        test_loader,
        cfg,
        device,
        all_feature_targets=all_feature_targets,
        all_feature_weights=all_feature_weights,
    )
    graphical_dir = save_prediction_graphs(
        model=best_model,
        base=base,
        indices=test_idx,
        output_dir=cfg.prediction_graphical_dir / "test",
        split="test",
        plot_count=cfg.prediction_plot_count,
        device=device,
        freq_axis_ghz=freq_axis_ghz,
        export_curve_points=cfg.export_curve_points,
        export_head_points=cfg.export_head_points,
    )
    boundary_graphical_dir = ""
    if boundary_idx:
        boundary_graphical_dir = str(
            save_prediction_graphs(
                model=best_model,
                base=base,
                indices=boundary_idx,
                output_dir=cfg.prediction_graphical_dir / "boundary",
                split="boundary",
                plot_count=len(boundary_idx),
                device=device,
                freq_axis_ghz=freq_axis_ghz,
                export_curve_points=cfg.export_curve_points,
                export_head_points=cfg.export_head_points,
            )
        )
    boundary_metrics = evaluate_boundary_condition(
        best_model,
        base,
        boundary_idx,
        cfg,
        device,
        all_feature_targets,
        all_feature_weights,
    )
    plot_loss_curve(history_rows, cfg.loss_plot_path, title="R6O Multi-Task Training Curve")
    summary = {
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "test_total": test_metrics["total"],
        "test_curve_mae": test_metrics["curve_mae"],
        "test_pred_max_db": test_metrics["pred_max_db"],
        "test_pred_min_db": test_metrics["pred_min_db"],
        "test_feature_loss": test_metrics["feature_loss"],
        "test_feature_mae_norm": test_metrics["feature_mae_norm"],
        "test_feature_mae_ghz": test_metrics["feature_mae_norm"]
        * max(float((base.freq_axis_ghz[-1] - base.freq_axis_ghz[0]).item()), 1.0e-6),
        "num_antennas": len(base),
        "seq_len": base.seq_len,
        "target_curve_points": cfg.target_curve_points,
        "output_curve_points": model.output_curve_points,
        "loss_target_points": base.seq_len,
        "export_curve_points": cfg.export_curve_points,
        "prediction_graphical_dir": str(graphical_dir),
        "boundary_graphical_dir": boundary_graphical_dir,
        "prediction_graph_count": min(cfg.prediction_plot_count, len(test_idx)),
        "loss_plot_path": str(cfg.loss_plot_path),
        "label_min_confidence": cfg.label_min_confidence,
        "use_label_fallbacks": cfg.use_label_fallbacks,
        "boundary_antenna_id": cfg.boundary_antenna_id,
        "boundary_loss_weight": cfg.boundary_loss_weight,
        "boundary_found": bool(boundary_idx),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "train_labeled_samples": int((all_feature_weights[train_idx] > 0).sum().item()) if all_feature_weights is not None else 0,
        "val_labeled_samples": int((all_feature_weights[val_idx] > 0).sum().item()) if all_feature_weights is not None else 0,
        "test_labeled_samples": int((all_feature_weights[test_idx] > 0).sum().item()) if all_feature_weights is not None else 0,
    }
    summary.update(boundary_metrics)
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
        er_default=cfg.er_default,
        target_curve_points=cfg.target_curve_points,
    )
    all_feature_targets: torch.Tensor | None = None
    all_feature_weights: torch.Tensor | None = None
    if cfg.use_notch_label_file:
        all_feature_targets, all_feature_weights = build_label_feature_targets(
            base,
            cfg.labels_csv_path,
            min_confidence=cfg.label_min_confidence,
            use_fallbacks=cfg.use_label_fallbacks,
        )
    freq_axis_ghz = base.freq_axis_ghz.to(device)
    _train_loader, val_loader, test_loader, _train_idx, val_idx, test_idx = build_dataloaders(base, cfg, device)
    loader = val_loader if split == "val" else test_loader
    indices = val_idx if split == "val" else test_idx
    model = build_model(cfg, device, seq_len=base.seq_len)
    load_model_weights(model, cfg.checkpoint_path, device)
    metrics = evaluate_model(
        model,
        loader,
        cfg,
        device,
        all_feature_targets=all_feature_targets,
        all_feature_weights=all_feature_weights,
    )
    graphical_dir = save_prediction_graphs(
        model=model,
        base=base,
        indices=indices,
        output_dir=cfg.prediction_graphical_dir / split,
        split=split,
        plot_count=cfg.prediction_plot_count,
        device=device,
        freq_axis_ghz=freq_axis_ghz,
        export_curve_points=cfg.export_curve_points,
        export_head_points=cfg.export_head_points,
    )
    metrics["split"] = split
    metrics["checkpoint"] = str(cfg.checkpoint_path)
    metrics["graphical_dir"] = str(graphical_dir)
    metrics["prediction_graph_count"] = min(cfg.prediction_plot_count, len(indices))
    metrics.update(
        evaluate_boundary_condition(
            model,
            base,
            boundary_indices(base, cfg.boundary_antenna_id),
            cfg,
            device,
            all_feature_targets,
            all_feature_weights,
        )
    )
    return metrics


def inspect_processed_dataset(cfg: Config) -> None:
    require_torch("inspect")
    base = ProcessedCurveDataset(
        cfg.processed_csv_path,
        cfg.processed_meta_path,
        er_default=cfg.er_default,
        target_curve_points=cfg.target_curve_points,
    )
    print(f"Processed dataset: {cfg.processed_csv_path}")
    print(f"Metadata: {cfg.processed_meta_path}")
    print(f"Antennas: {len(base)}")
    print(f"Sequence length: {base.seq_len}")
    print(f"Frequency axis first 5: {base.freq_axis_ghz[:5].tolist()}")
    print(f"First geometry bits: {base.geometry[0, :10].tolist()}")
    print(f"First curve values: {base.curves_db[0, :5].tolist()}")
    if cfg.use_notch_label_file:
        _targets, weights = build_label_feature_targets(
            base,
            cfg.labels_csv_path,
            min_confidence=cfg.label_min_confidence,
            use_fallbacks=cfg.use_label_fallbacks,
        )
        print(f"Labeled notch rows: {int((weights > 0).sum().item())}/{len(base)}")


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="R6O multi-task notch model (curve MAE + feature loss)."
    )
    parser.add_argument("command", choices=["preprocess", "inspect", "train"], nargs="?", default=cfg.command)
    parser.add_argument("--geometry-catalog-path", type=Path, default=cfg.geometry_catalog_path)
    parser.add_argument("--curves-root", type=Path, default=cfg.curves_root)
    parser.add_argument("--processed-csv-path", type=Path, default=cfg.processed_csv_path)
    parser.add_argument("--processed-meta-path", type=Path, default=cfg.processed_meta_path)
    parser.add_argument("--train-index-path", type=Path, default=cfg.train_index_path)
    parser.add_argument("--val-index-path", type=Path, default=cfg.val_index_path)
    parser.add_argument("--labels-csv-path", type=Path, default=cfg.labels_csv_path)
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
    parser.add_argument(
        "--target-curve-points",
        type=int,
        default=cfg.target_curve_points,
        help="Resample loaded target curves to this many points using shape-preserving PCHIP.",
    )
    parser.add_argument(
        "--export-curve-points",
        type=int,
        default=cfg.export_curve_points,
        help="Export/plot curve points; native sequence is linearly interpolated when needed.",
    )
    parser.add_argument(
        "--output-curve-points",
        type=int,
        default=cfg.output_curve_points,
        help="Fixed R6O curve output size. Use 501 for feed-first then 60k training.",
    )
    parser.add_argument(
        "--curve-head-points",
        type=str,
        default=cfg.curve_head_points,
        help="Deprecated compatibility arg; ignored by the single-head R6O.",
    )
    parser.add_argument(
        "--active-curve-points",
        type=int,
        default=cfg.active_curve_points,
        help="Deprecated compatibility arg; loss resamples the fixed output to target length.",
    )
    parser.add_argument(
        "--export-head-points",
        type=int,
        default=cfg.export_head_points,
        help="Deprecated compatibility arg; ignored by the single-head R6O.",
    )
    parser.add_argument("--init-checkpoint-path", type=Path, default=cfg.init_checkpoint_path)
    parser.add_argument("--strict-init", action="store_true")
    parser.add_argument("--loss-feature-weight", type=float, default=cfg.loss_feature_weight)
    parser.add_argument("--loss-curve-weight", type=float, default=cfg.loss_curve_weight)
    parser.add_argument("--loss-focus-alpha", type=float, default=cfg.loss_focus_alpha,
                        help="Gaussian amplification factor near resonance target (0=plain MSE).")
    parser.add_argument("--loss-focus-bw-norm", type=float, default=cfg.loss_focus_bw_norm,
                        help="Normalized resonance bandwidth for adaptive focus (freq_span units).")
    parser.add_argument("--disable-material-channel", action="store_true")
    parser.add_argument("--use-feed-channel", action="store_true",
                        help="Add a fixed binary feed-structure channel from meta feed_cells.")
    parser.add_argument("--disable-index-files", action="store_true")
    parser.add_argument("--disable-notch-label-file", action="store_true")
    parser.add_argument(
        "--label-min-confidence",
        type=float,
        default=cfg.label_min_confidence,
        help="Minimum resonance-label confidence used by the R6O feature loss.",
    )
    parser.add_argument(
        "--use-label-fallbacks",
        action="store_true",
        help="Also train the feature head on global-min fallback labels.",
    )
    parser.add_argument(
        "--boundary-antenna-id",
        type=int,
        default=cfg.boundary_antenna_id,
        help="Antenna id used as a repeated boundary/anchor condition during R6O training.",
    )
    parser.add_argument(
        "--boundary-loss-weight",
        type=float,
        default=cfg.boundary_loss_weight,
        help="Extra per-batch anchor weight for the boundary antenna. Set 0 to disable.",
    )
    parser.add_argument("--er-default", type=float, default=cfg.er_default)
    parser.add_argument("--overwrite-processed", action="store_true")
    parser.add_argument(
        "--feature-finetune-epochs",
        type=int,
        default=cfg.feature_finetune_epochs,
        help="After main training, freeze encoder and fine-tune feature head for N epochs (0 = disabled).",
    )
    parser.add_argument(
        "--feature-finetune-min-confidence",
        type=float,
        default=cfg.feature_finetune_min_confidence,
        help="Only include samples with label confidence >= this value for feature fine-tuning.",
    )
    parser.add_argument(
        "--feature-finetune-lr",
        type=float,
        default=cfg.feature_finetune_lr,
        help="Learning rate for the feature-head fine-tune phase.",
    )
    args = parser.parse_args()

    cfg.command = args.command
    cfg.geometry_catalog_path = args.geometry_catalog_path
    cfg.curves_root = args.curves_root
    cfg.processed_csv_path = args.processed_csv_path
    cfg.processed_meta_path = args.processed_meta_path
    cfg.train_index_path = args.train_index_path
    cfg.val_index_path = args.val_index_path
    cfg.labels_csv_path = args.labels_csv_path
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
    cfg.target_curve_points = max(0, args.target_curve_points)
    cfg.export_curve_points = max(0, args.export_curve_points)
    cfg.output_curve_points = max(0, args.output_curve_points)
    cfg.curve_head_points = args.curve_head_points
    cfg.active_curve_points = max(0, args.active_curve_points)
    cfg.export_head_points = max(0, args.export_head_points)
    cfg.init_checkpoint_path = args.init_checkpoint_path
    cfg.strict_init = args.strict_init
    cfg.loss_feature_weight = max(0.0, args.loss_feature_weight)
    cfg.loss_curve_weight = max(0.0, args.loss_curve_weight)
    cfg.loss_focus_alpha = max(0.0, args.loss_focus_alpha)
    cfg.loss_focus_bw_norm = max(1e-6, args.loss_focus_bw_norm)
    cfg.use_material_channel = not args.disable_material_channel
    cfg.use_feed_channel = args.use_feed_channel
    cfg.use_index_files = not args.disable_index_files
    cfg.use_notch_label_file = not args.disable_notch_label_file
    cfg.label_min_confidence = float(np.clip(args.label_min_confidence, 0.0, 1.0))
    cfg.use_label_fallbacks = args.use_label_fallbacks
    cfg.boundary_antenna_id = int(args.boundary_antenna_id)
    cfg.boundary_loss_weight = max(0.0, float(args.boundary_loss_weight))
    cfg.er_default = float(np.clip(args.er_default, cfg.er_min, cfg.er_max))
    cfg.overwrite_processed = args.overwrite_processed
    cfg.feature_finetune_epochs = max(0, args.feature_finetune_epochs)
    cfg.feature_finetune_min_confidence = float(np.clip(args.feature_finetune_min_confidence, 0.0, 1.0))
    cfg.feature_finetune_lr = max(0.0, args.feature_finetune_lr)
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
