"""Build robust resonance labels and write train.txt / val.txt splits.

Pipeline:
1) optional Savitzky-Golay smoothing (fallback to moving average)
2) local minima search on smoothed curve
3) constraints: depth threshold, minimum prominence, minimum spacing
4) fallback to global minimum when no valid local minima
5) family-aware split (same idea as R5PINN/R6P)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GRID_H = 10
GRID_W = 10
NUM_CELLS = GRID_H * GRID_W


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_meta_path() -> Path:
    return _repo_root() / "Data" / "processed" / "Full_60000Data_61dB.meta.json"


def _default_csv_path() -> Path:
    return _repo_root() / "Data" / "processed" / "Full_60000Data_61dB.csv"


def _default_train_txt() -> Path:
    return _repo_root() / "Data" / "processed" / "train.txt"


def _default_val_txt() -> Path:
    return _repo_root() / "Data" / "processed" / "val.txt"


def _default_labels_csv() -> Path:
    return _repo_root() / "Data" / "processed" / "resonance_labels.csv"


@dataclass(frozen=True)
class LabelRow:
    row_index: int
    antenna_id: int
    resonant_freq_ghz: float
    depth_db: float
    prominence_db: float
    confidence: float
    method: str
    is_valid_local_min: bool


def smooth_curve(values: np.ndarray, method: str, window: int, polyorder: int) -> np.ndarray:
    if window < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    if method == "savgol":
        try:
            from scipy.signal import savgol_filter  # type: ignore

            if window > values.size:
                window = values.size if values.size % 2 == 1 else values.size - 1
            if window < 3:
                return values.copy()
            poly = max(1, min(polyorder, window - 1))
            return savgol_filter(values, window_length=window, polyorder=poly, mode="interp")
        except Exception:
            pass

    # moving average fallback
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _local_minima_indices(y: np.ndarray) -> list[int]:
    mins: list[int] = []
    for i in range(1, y.size - 1):
        if y[i] < y[i - 1] and y[i] <= y[i + 1]:
            mins.append(i)
    return mins


def _prominence(y: np.ndarray, idx: int) -> float:
    left_peak = float(np.max(y[:idx])) if idx > 0 else float(y[idx])
    right_peak = float(np.max(y[idx + 1 :])) if idx + 1 < y.size else float(y[idx])
    return min(left_peak, right_peak) - float(y[idx])


def pick_resonance(
    freq_ghz: np.ndarray,
    s11_db: np.ndarray,
    depth_threshold_db: float,
    min_prominence_db: float,
    min_spacing_ghz: float,
    smooth_method: str,
    smooth_window: int,
    smooth_polyorder: int,
) -> tuple[float, float, float, float, str, bool]:
    ys = smooth_curve(s11_db, smooth_method, smooth_window, smooth_polyorder)
    candidate_ids = _local_minima_indices(ys)

    # rank candidates by depth (more negative first)
    ranked = sorted(candidate_ids, key=lambda i: float(ys[i]))
    accepted: list[int] = []
    for idx in ranked:
        depth = float(ys[idx])
        prom = _prominence(ys, idx)
        if depth > depth_threshold_db:
            continue
        if prom < min_prominence_db:
            continue
        f = float(freq_ghz[idx])
        too_close = any(abs(f - float(freq_ghz[j])) < min_spacing_ghz for j in accepted)
        if too_close:
            continue
        accepted.append(idx)

    if accepted:
        best = min(accepted, key=lambda i: float(ys[i]))
        method = "local_min"
        valid = True
    else:
        best = int(np.argmin(ys))
        method = "global_min_fallback"
        valid = False

    best_depth = float(s11_db[best])
    best_prom = float(_prominence(ys, best))
    # confidence in [0, 1]
    depth_margin = max(0.0, depth_threshold_db - best_depth)
    depth_score = np.tanh(depth_margin / 10.0)
    prom_score = np.tanh(max(0.0, best_prom) / 10.0)
    if 0 < best < s11_db.size - 1:
        curvature = max(0.0, float(s11_db[best - 1] - 2.0 * s11_db[best] + s11_db[best + 1]))
    else:
        curvature = 0.0
    curv_score = np.tanh(curvature / 5.0)
    conf = 0.45 * depth_score + 0.45 * prom_score + 0.10 * curv_score
    if method == "global_min_fallback":
        conf *= 0.7

    return float(freq_ghz[best]), best_depth, best_prom, float(np.clip(conf, 0.0, 1.0)), method, valid


def infer_family_key(bits: np.ndarray) -> tuple[int, ...]:
    grid = bits.reshape(GRID_H, GRID_W)
    metal = int(grid.sum() // 5)
    quadrants = (
        int(grid[:5, :5].sum() // 3),
        int(grid[:5, 5:].sum() // 3),
        int(grid[5:, :5].sum() // 3),
        int(grid[5:, 5:].sum() // 3),
    )
    row_bands = tuple(int(grid[r * 2 : (r + 1) * 2, :].sum() // 4) for r in range(5))
    col_bands = tuple(int(grid[:, c * 2 : (c + 1) * 2].sum() // 4) for c in range(5))
    h_sym = int((grid == np.fliplr(grid)).sum() // 10)
    v_sym = int((grid == np.flipud(grid)).sum() // 10)
    return (metal, *quadrants, *row_bands, *col_bands, h_sym, v_sym)


def family_aware_split(
    geometry_bits: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    families: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i in range(geometry_bits.shape[0]):
        families[infer_family_key(geometry_bits[i])].append(i)

    keys = list(families.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    total = geometry_bits.shape[0]
    train_target = int(total * train_ratio)
    val_target = int(total * val_ratio)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for key in keys:
        members = families[key]
        if len(train_idx) < train_target:
            train_idx.extend(members)
        elif len(val_idx) < val_target:
            val_idx.extend(members)
        else:
            test_idx.extend(members)

    if not val_idx:
        cut = max(1, len(train_idx) // 10)
        val_idx = train_idx[-cut:]
        train_idx = train_idx[:-cut]
    if not test_idx:
        cut = max(1, len(val_idx) // 2)
        test_idx = val_idx[-cut:]
        val_idx = val_idx[:-cut]
    if not test_idx:
        cut = max(1, len(train_idx) // 10)
        test_idx = train_idx[-cut:]
        train_idx = train_idx[:-cut]
    return train_idx, val_idx, test_idx


def write_split_txt(path: Path, rows: list[LabelRow], indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("antenna_id\tresonant_freq_ghz\tdepth_db\tprominence_db\tconfidence\tmethod\n")
        for i in indices:
            r = rows[i]
            f.write(
                f"{r.antenna_id}\t{r.resonant_freq_ghz:.9f}\t{r.depth_db:.6f}\t"
                f"{r.prominence_db:.6f}\t{r.confidence:.6f}\t{r.method}\n"
            )


def write_labels_csv(path: Path, rows: list[LabelRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "row_index",
                "antenna_id",
                "resonant_freq_ghz",
                "depth_db",
                "prominence_db",
                "confidence",
                "method",
                "is_valid_local_min",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.row_index,
                    r.antenna_id,
                    f"{r.resonant_freq_ghz:.9f}",
                    f"{r.depth_db:.6f}",
                    f"{r.prominence_db:.6f}",
                    f"{r.confidence:.6f}",
                    r.method,
                    int(r.is_valid_local_min),
                ]
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build robust resonance labels + train/val txt.")
    p.add_argument("--processed-csv-path", type=Path, default=_default_csv_path())
    p.add_argument("--processed-meta-path", type=Path, default=_default_meta_path())
    p.add_argument("--train-txt", type=Path, default=_default_train_txt())
    p.add_argument("--val-txt", type=Path, default=_default_val_txt())
    p.add_argument("--labels-csv", type=Path, default=_default_labels_csv())
    p.add_argument("--depth-threshold-db", type=float, default=-10.0)
    p.add_argument("--min-prominence-db", type=float, default=2.0)
    p.add_argument("--min-spacing-ghz", type=float, default=0.10)
    p.add_argument("--smooth-method", choices=["savgol", "moving"], default="savgol")
    p.add_argument("--smooth-window", type=int, default=5)
    p.add_argument("--smooth-polyorder", type=int, default=2)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.processed_meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {args.processed_meta_path}")
    if not args.processed_csv_path.exists():
        raise FileNotFoundError(
            f"Processed CSV not found: {args.processed_csv_path}. "
            "Run preprocessing first to build the processed CSV."
        )

    meta = json.loads(args.processed_meta_path.read_text(encoding="utf-8"))
    freq_ghz = np.asarray(meta.get("freq_axis_ghz", []), dtype=np.float32)
    if freq_ghz.size == 0:
        raise ValueError("Missing freq_axis_ghz in meta.")

    values = np.loadtxt(args.processed_csv_path, delimiter=",", dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] < NUM_CELLS + 1:
        raise ValueError(
            f"Expected at least {NUM_CELLS + 1} columns, got {values.shape[1]}."
        )
    seq_len = values.shape[1] - NUM_CELLS
    if seq_len != freq_ghz.size:
        raise ValueError(f"Frequency axis length {freq_ghz.size} != seq_len {seq_len}.")

    geom_all = (values[:, :NUM_CELLS] > 0.5).astype(np.float32)
    curves_all = values[:, NUM_CELLS:]
    antenna_ids_all = meta.get("matched_antenna_ids", list(range(1, values.shape[0] + 1)))
    if len(antenna_ids_all) != values.shape[0]:
        antenna_ids_all = list(range(1, values.shape[0] + 1))

    # Drop rows with NaN/Inf S11 before building labels and split indices.
    valid_mask = np.all(np.isfinite(curves_all), axis=1)
    nan_skipped = int((~valid_mask).sum())
    valid_orig_indices = np.where(valid_mask)[0]
    geom = geom_all[valid_orig_indices]
    curves = curves_all[valid_orig_indices]
    antenna_ids = [antenna_ids_all[j] for j in valid_orig_indices]

    rows: list[LabelRow] = []
    fallback_count = 0
    for i in range(len(curves)):
        f_res, depth, prom, conf, method, valid = pick_resonance(
            freq_ghz=freq_ghz,
            s11_db=curves[i],
            depth_threshold_db=args.depth_threshold_db,
            min_prominence_db=args.min_prominence_db,
            min_spacing_ghz=args.min_spacing_ghz,
            smooth_method=args.smooth_method,
            smooth_window=args.smooth_window,
            smooth_polyorder=args.smooth_polyorder,
        )
        if method == "global_min_fallback":
            fallback_count += 1
        rows.append(
            LabelRow(
                row_index=int(valid_orig_indices[i]),
                antenna_id=int(antenna_ids[i]),
                resonant_freq_ghz=f_res,
                depth_db=depth,
                prominence_db=prom,
                confidence=conf,
                method=method,
                is_valid_local_min=valid,
            )
        )

    train_idx, val_idx, test_idx = family_aware_split(
        geometry_bits=geom,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    write_labels_csv(args.labels_csv, rows)
    write_split_txt(args.train_txt, rows, train_idx)
    write_split_txt(args.val_txt, rows, val_idx)

    low_conf = sum(1 for r in rows if r.confidence < 0.35)
    print(f"total_samples={len(rows)}")
    print(f"nan_skipped={nan_skipped}")
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"fallback_global_min={fallback_count}")
    print(f"low_confidence(<0.35)={low_conf}")
    print(f"labels_csv={args.labels_csv}")
    print(f"train_txt={args.train_txt}")
    print(f"val_txt={args.val_txt}")


if __name__ == "__main__":
    main()

