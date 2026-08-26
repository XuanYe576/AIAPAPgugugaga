from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .config import DataConfig, ExperimentConfig, PhysicsConfig


C0 = 299_792_458.0


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    freq_axis_ghz: torch.Tensor
    source_name: str
    num_samples: int
    num_labeled_samples: int


def uniform_freq_axis(freq_start_ghz: float, freq_stop_ghz: float, seq_len: int) -> np.ndarray:
    return np.linspace(freq_start_ghz, freq_stop_ghz, seq_len, dtype=np.float32)


def load_frequency_axis(data_cfg: DataConfig) -> np.ndarray:
    meta_path = Path(data_cfg.processed_meta_path)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        freq_axis = np.asarray(meta.get("freq_axis_ghz", []), dtype=np.float32)
        if freq_axis.size > 0:
            return freq_axis
    return uniform_freq_axis(data_cfg.freq_start_ghz, data_cfg.freq_stop_ghz, data_cfg.seq_len)


def _find_curve_dips(
    curve: np.ndarray,
    min_depth_db: float,
    min_separation_bins: int,
) -> list[int]:
    """Locate every genuine resonance dip in a dense S11 curve.

    A bin is a candidate if it is a local minimum; its "depth" is measured
    against the higher of its two local shoulders within `min_separation_bins`,
    which rejects sub-dB ripple noise while keeping real, possibly multiple,
    resonances. Candidates are kept deepest-first, discarding anything within
    `min_separation_bins` of an already-kept dip so overlapping detections on
    the same physical notch collapse to one bin.
    """
    seq_len = int(curve.shape[0])
    if seq_len < 3 or not np.isfinite(curve).all():
        return []
    separation = max(1, int(min_separation_bins))
    candidates: list[tuple[float, int]] = []
    for i in range(1, seq_len - 1):
        if curve[i] > curve[i - 1] or curve[i] > curve[i + 1]:
            continue
        if curve[i] == curve[i - 1] and curve[i] == curve[i + 1]:
            continue
        left = float(curve[max(0, i - separation) : i + 1].max())
        right = float(curve[i : min(seq_len, i + separation + 1)].max())
        depth = min(left, right) - float(curve[i])
        if depth >= min_depth_db:
            candidates.append((depth, i))
    candidates.sort(key=lambda item: -item[0])
    kept: list[int] = []
    for _depth, idx in candidates:
        if all(abs(idx - existing) >= separation for existing in kept):
            kept.append(idx)
    kept.sort()
    return kept


def _make_multi_dip_targets(
    curve: np.ndarray,
    freq_axis_ghz: np.ndarray,
    target_radius_bins: int,
    min_depth_db: float,
    min_separation_bins: int,
) -> dict[str, np.ndarray | float | int]:
    """Build dense, multi-resonance supervision targets directly from a curve.

    Unlike a single sparse (freq, depth) label, this scans the full curve for
    every dip that clears `min_depth_db` of local prominence, so antennas with
    several real resonances (common in this dataset) get credit/supervision
    for all of them instead of only the single deepest one.
    """
    seq_len = int(freq_axis_ghz.size)
    dip_mask = np.zeros(seq_len, dtype=np.float32)
    dip_anchor_mask = np.zeros(seq_len, dtype=np.float32)
    dip_offset_ghz = np.zeros(seq_len, dtype=np.float32)
    dip_depth_db = np.zeros(seq_len, dtype=np.float32)
    dip_supervision_mask = np.ones(seq_len, dtype=np.float32)

    dip_indices = _find_curve_dips(curve, min_depth_db=min_depth_db, min_separation_bins=min_separation_bins)
    sigma = max(1.0, float(target_radius_bins))
    for dip_index in dip_indices:
        dip_anchor_mask[dip_index] = 1.0
        for offset in range(-target_radius_bins, target_radius_bins + 1):
            idx = dip_index + offset
            if 0 <= idx < seq_len:
                dip_mask[idx] = max(dip_mask[idx], math.exp(-0.5 * (offset / sigma) ** 2))
        # The curve is already sampled on freq_axis_ghz, so the dip sits
        # exactly on its bin; there is no sub-bin frequency to regress to.
        dip_depth_db[dip_index] = float(max(0.0, -float(curve[dip_index])))

    label_available = float(curve.shape == (seq_len,) and np.isfinite(curve).all())
    if dip_indices:
        primary_index = max(dip_indices, key=lambda idx: dip_depth_db[idx])
    else:
        primary_index = -1

    return {
        "dip_mask": dip_mask,
        "dip_anchor_mask": dip_anchor_mask,
        "dip_offset_ghz": dip_offset_ghz,
        "dip_depth_db": dip_depth_db,
        "dip_supervision_mask": dip_supervision_mask,
        "dip_index": int(primary_index),
        "dip_count": float(len(dip_indices)),
        "label_available": label_available,
        "label_confidence": label_available,
    }


def _active_extent(bits: np.ndarray, axis: int) -> tuple[int, int]:
    occupied = bits.max(axis=axis) > 0.5
    if not occupied.any():
        size = bits.shape[1 - axis]
        return 0, size - 1
    first = int(np.argmax(occupied))
    last = int(occupied.size - np.argmax(occupied[::-1]) - 1)
    return first, last


def _estimate_mode_frequencies_ghz(
    geometry: np.ndarray,
    data_cfg: DataConfig,
    physics_cfg: PhysicsConfig,
) -> list[float]:
    rows = int(data_cfg.grid_height)
    cols = int(data_cfg.grid_width)
    grid = geometry.reshape(rows, cols)
    row0, row1 = _active_extent(grid, axis=1)
    col0, col1 = _active_extent(grid, axis=0)
    length_m = max(1, row1 - row0 + 1) * physics_cfg.cell_size_y_m
    width_m = max(1, col1 - col0 + 1) * physics_cfg.cell_size_x_m
    h = max(physics_cfg.substrate_height_m, 1.0e-6)
    er = max(1.0, physics_cfg.relative_permittivity)
    eps_eff = (er + 1.0) / 2.0 + ((er - 1.0) / 2.0) * (1.0 + 12.0 * h / max(width_m, 1.0e-6)) ** -0.5
    delta_l = 0.412 * h * (
        ((eps_eff + 0.3) * ((width_m / h) + 0.264))
        / max((eps_eff - 0.258) * ((width_m / h) + 0.8), 1.0e-6)
    )
    length_eff = length_m + 2.0 * delta_l
    width_eff = width_m + 2.0 * delta_l
    mode_freqs_ghz: list[float] = []
    for mode_m, mode_n in physics_cfg.physics_modes:
        mode_term = (mode_m / max(length_eff, 1.0e-6)) ** 2 + (mode_n / max(width_eff, 1.0e-6)) ** 2
        freq_hz = 0.5 * C0 * math.sqrt(mode_term / max(eps_eff, 1.0e-6))
        mode_freqs_ghz.append(freq_hz / 1.0e9)
    return mode_freqs_ghz


def generate_synthetic_curve_sample(
    rng: np.random.Generator,
    data_cfg: DataConfig,
    physics_cfg: PhysicsConfig,
) -> dict[str, np.ndarray | int | float]:
    geometry = (rng.random((data_cfg.grid_height, data_cfg.grid_width)) > 0.45).astype(np.float32)
    if geometry.sum() < 4:
        geometry[rng.integers(0, data_cfg.grid_height), rng.integers(0, data_cfg.grid_width)] = 1.0
    geometry_flat = geometry.reshape(-1)
    freq_axis = uniform_freq_axis(data_cfg.freq_start_ghz, data_cfg.freq_stop_ghz, data_cfg.seq_len)
    mode_freqs = _estimate_mode_frequencies_ghz(geometry_flat, data_cfg, physics_cfg)
    valid_modes = [freq for freq in mode_freqs if freq_axis[0] <= freq <= freq_axis[-1]]
    if not valid_modes:
        valid_modes = [float(freq_axis[rng.integers(0, data_cfg.seq_len)])]
    dominant_freq = min(valid_modes, key=lambda freq: abs(freq - 3.5))
    depth_db = 8.0 + 18.0 * float(geometry.mean())
    slope = -2.0 - 3.0 * float(geometry.mean())
    ripple = 1.2 * np.sin(np.linspace(0.0, 2.0 * math.pi, data_cfg.seq_len, dtype=np.float32))
    baseline = slope - 2.0 * np.linspace(0.0, 1.0, data_cfg.seq_len, dtype=np.float32) + ripple
    sigma = 0.12 + 0.06 * float(abs(geometry[:, : geometry.shape[1] // 2].mean() - geometry[:, geometry.shape[1] // 2 :].mean()))
    dominant = depth_db * np.exp(-0.5 * ((freq_axis - dominant_freq) / sigma) ** 2)
    secondary_freq = valid_modes[-1]
    secondary = 0.35 * depth_db * np.exp(-0.5 * ((freq_axis - secondary_freq) / (sigma * 1.35)) ** 2)
    curve = baseline - dominant - secondary
    dense = _make_multi_dip_targets(
        curve=curve,
        freq_axis_ghz=freq_axis,
        target_radius_bins=data_cfg.dip_target_radius_bins,
        min_depth_db=data_cfg.min_dip_depth_db,
        min_separation_bins=data_cfg.dip_min_separation_bins,
    )
    return {
        "geometry": geometry.astype(np.float32),
        "curve": curve.astype(np.float32),
        **dense,
        "antenna_id": 0,
    }


class RealAntennaDataset(Dataset):
    def __init__(self, cfg: ExperimentConfig) -> None:
        data_cfg = cfg.data
        csv_path = Path(data_cfg.processed_csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Processed CSV not found: {csv_path}")
        matrix = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] < data_cfg.input_dim + 1:
            raise ValueError(
                f"Processed CSV has {matrix.shape[1]} columns, expected at least {data_cfg.input_dim + 1}."
            )
        if data_cfg.max_samples > 0:
            matrix = matrix[: data_cfg.max_samples]
        finite_mask = np.isfinite(matrix).all(axis=1)
        dropped_samples = int((~finite_mask).sum())
        if dropped_samples > 0:
            matrix = matrix[finite_mask]
        self.data_cfg = data_cfg
        self.freq_axis_ghz = load_frequency_axis(data_cfg)
        self.seq_len = int(self.freq_axis_ghz.size)
        self.geometry = matrix[:, : data_cfg.input_dim].reshape(-1, data_cfg.grid_height, data_cfg.grid_width)
        self.curves = matrix[:, data_cfg.input_dim : data_cfg.input_dim + self.seq_len]
        if self.curves.shape[1] != self.seq_len:
            raise ValueError(
                f"Curve width mismatch: CSV provides {self.curves.shape[1]} samples, meta says {self.seq_len}."
            )
        meta = {}
        meta_path = Path(data_cfg.processed_meta_path)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        antenna_ids = meta.get("matched_antenna_ids")
        if antenna_ids and len(antenna_ids) >= len(finite_mask):
            selected_ids = [int(value) for value in antenna_ids[: len(finite_mask)]]
            self.antenna_ids = [antenna_id for antenna_id, keep in zip(selected_ids, finite_mask) if keep]
        else:
            self.antenna_ids = list(range(1, len(self.geometry) + 1))
        if dropped_samples > 0:
            print(f"[data] dropped {dropped_samples} non-finite samples from {csv_path.name}")
        self.num_labeled_samples = 0
        self.targets: list[dict[str, np.ndarray | float | int]] = []
        for curve in self.curves:
            dense = _make_multi_dip_targets(
                curve=curve,
                freq_axis_ghz=self.freq_axis_ghz,
                target_radius_bins=data_cfg.dip_target_radius_bins,
                min_depth_db=data_cfg.min_dip_depth_db,
                min_separation_bins=data_cfg.dip_min_separation_bins,
            )
            self.num_labeled_samples += int(float(dense["label_available"]) > 0.0)
            self.targets.append(dense)

    def __len__(self) -> int:
        return int(self.geometry.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        dense = self.targets[index]
        return {
            "geometry": torch.from_numpy(self.geometry[index].astype(np.float32)),
            "curve": torch.from_numpy(self.curves[index].astype(np.float32)),
            "dip_mask": torch.from_numpy(np.asarray(dense["dip_mask"], dtype=np.float32)),
            "dip_anchor_mask": torch.from_numpy(np.asarray(dense["dip_anchor_mask"], dtype=np.float32)),
            "dip_offset_ghz": torch.from_numpy(np.asarray(dense["dip_offset_ghz"], dtype=np.float32)),
            "dip_depth_db": torch.from_numpy(np.asarray(dense["dip_depth_db"], dtype=np.float32)),
            "dip_supervision_mask": torch.from_numpy(np.asarray(dense["dip_supervision_mask"], dtype=np.float32)),
            "dip_index": torch.tensor(int(dense["dip_index"]), dtype=torch.long),
            "dip_count": torch.tensor(float(dense["dip_count"]), dtype=torch.float32),
            "label_available": torch.tensor(float(dense["label_available"]), dtype=torch.float32),
            "label_confidence": torch.tensor(float(dense["label_confidence"]), dtype=torch.float32),
            "antenna_id": torch.tensor(self.antenna_ids[index], dtype=torch.long),
        }


class SyntheticAntennaDataset(Dataset):
    def __init__(self, cfg: ExperimentConfig, size: int | None = None) -> None:
        self.cfg = cfg
        self.size = int(size or cfg.data.synthetic_size)
        self.freq_axis_ghz = uniform_freq_axis(cfg.data.freq_start_ghz, cfg.data.freq_stop_ghz, cfg.data.seq_len)
        rng = np.random.default_rng(cfg.train.random_seed)
        self.samples = [
            generate_synthetic_curve_sample(rng=rng, data_cfg=cfg.data, physics_cfg=cfg.physics)
            for _ in range(self.size)
        ]
        self.num_labeled_samples = self.size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "geometry": torch.from_numpy(np.asarray(sample["geometry"], dtype=np.float32)),
            "curve": torch.from_numpy(np.asarray(sample["curve"], dtype=np.float32)),
            "dip_mask": torch.from_numpy(np.asarray(sample["dip_mask"], dtype=np.float32)),
            "dip_anchor_mask": torch.from_numpy(np.asarray(sample["dip_anchor_mask"], dtype=np.float32)),
            "dip_offset_ghz": torch.from_numpy(np.asarray(sample["dip_offset_ghz"], dtype=np.float32)),
            "dip_depth_db": torch.from_numpy(np.asarray(sample["dip_depth_db"], dtype=np.float32)),
            "dip_supervision_mask": torch.from_numpy(np.asarray(sample["dip_supervision_mask"], dtype=np.float32)),
            "dip_index": torch.tensor(int(sample["dip_index"]), dtype=torch.long),
            "dip_count": torch.tensor(float(sample["dip_count"]), dtype=torch.float32),
            "label_available": torch.tensor(float(sample["label_available"]), dtype=torch.float32),
            "label_confidence": torch.tensor(float(sample["label_confidence"]), dtype=torch.float32),
            "antenna_id": torch.tensor(int(index + 1), dtype=torch.long),
        }


def build_dataset(cfg: ExperimentConfig) -> tuple[Dataset, torch.Tensor, str, int]:
    real_ready = cfg.data.use_real_dataset and Path(cfg.data.processed_csv_path).exists()
    if real_ready:
        dataset = RealAntennaDataset(cfg)
        return dataset, torch.from_numpy(dataset.freq_axis_ghz.copy()), "real", dataset.num_labeled_samples
    if cfg.data.use_synthetic_if_missing:
        dataset = SyntheticAntennaDataset(cfg)
        return dataset, torch.from_numpy(dataset.freq_axis_ghz.copy()), "synthetic", dataset.num_labeled_samples
    raise FileNotFoundError(
        "Neither the real processed dataset nor the synthetic fallback is available. "
        "Enable use_synthetic_if_missing or provide processed CSV/meta files."
    )


def split_indices(size: int, train_ratio: float, val_ratio: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    indices = list(range(size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_end = max(1, int(size * train_ratio))
    val_end = min(size, train_end + max(1, int(size * val_ratio)))
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    if not val_idx:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    if not test_idx:
        test_idx = val_idx[-1:]
        val_idx = val_idx[:-1]
    return train_idx, val_idx, test_idx


def build_dataloaders(cfg: ExperimentConfig) -> DatasetBundle:
    dataset, freq_axis_ghz, source_name, num_labeled_samples = build_dataset(cfg)
    train_idx, val_idx, test_idx = split_indices(
        size=len(dataset),
        train_ratio=cfg.train.train_ratio,
        val_ratio=cfg.train.val_ratio,
        seed=cfg.train.random_seed,
    )
    loaders = {
        "train": DataLoader(
            Subset(dataset, train_idx),
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
        ),
        "val": DataLoader(
            Subset(dataset, val_idx),
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
        ),
        "test": DataLoader(
            Subset(dataset, test_idx),
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
        ),
    }
    return DatasetBundle(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        freq_axis_ghz=freq_axis_ghz.float(),
        source_name=source_name,
        num_samples=len(dataset),
        num_labeled_samples=num_labeled_samples,
    )
