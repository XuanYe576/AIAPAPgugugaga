"""Utilities for converting the raw antenna folder layout into processed CSV data."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _default_geometry_catalog_path() -> Path:
    candidates = [
        RAW_DATA_ROOT / "30000 Patch Antenna File" / "patch_antennas_updated4.csv",
        RAW_DATA_ROOT / "19001-20000" / "patch_antennas_updated3.csv",
        RAW_DATA_ROOT / "15000 Patch Antenna File" / "patch_antennas_updated2.csv",
        REPO_ROOT / "patch_antennas_updated.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _default_processed_csv() -> Path:
    return REPO_ROOT / "Data" / "processed" / "Full_30000Data_61dB.csv"


def _default_processed_meta() -> Path:
    return REPO_ROOT / "Data" / "processed" / "Full_30000Data_61dB.meta.json"


def parse_geometry_catalog(
    path: Path,
    grid_height: int = 10,
    grid_width: int = 10,
) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    geometries: dict[int, np.ndarray] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header_rows = [next(reader) for _ in range(3)]
        meta = {
            "num_antennas": int(header_rows[0][1]),
            "grid_width": int(header_rows[1][1]),
            "grid_height": int(header_rows[2][1]),
        }
        while True:
            try:
                marker = next(reader)
            except StopIteration:
                break
            if not marker:
                continue
            match = re.search(r"Patch Antenna(\d+)", marker[0])
            if not match:
                continue
            antenna_id = int(match.group(1))
            grid_rows: list[list[float]] = []
            for _ in range(grid_height):
                grid_row = next(reader)
                if len(grid_row) != grid_width:
                    raise ValueError(
                        f"Unexpected geometry row width in {path}: "
                        f"antenna {antenna_id} has {len(grid_row)} columns."
                    )
                grid_rows.append([float(value) for value in grid_row])
            geometries[antenna_id] = np.asarray(grid_rows, dtype=np.float32).reshape(-1)
    return geometries, meta


def collect_curve_paths(root: Path) -> tuple[dict[int, Path], dict[int, list[Path]]]:
    selected: dict[int, Path] = {}
    duplicates: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("patch_*_s11_plot.csv")):
        match = re.search(r"patch_(\d+)_s11_plot\.csv$", path.name)
        if not match:
            continue
        antenna_id = int(match.group(1))
        if antenna_id in selected:
            duplicates[antenna_id].append(path)
            continue
        selected[antenna_id] = path
    return selected, duplicates


def load_curve_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    freq_ghz: list[float] = []
    s11_db: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or len(header) < 2:
            raise ValueError(f"Curve file is missing a 2-column header: {path}")
        for row in reader:
            if len(row) < 2:
                continue
            freq_ghz.append(float(row[0]))
            s11_db.append(float(row[1]))
    if not freq_ghz:
        raise ValueError(f"Curve file has no samples: {path}")
    return np.asarray(freq_ghz, dtype=np.float32), np.asarray(s11_db, dtype=np.float32)


def build_processed_csv_from_raw_data(
    *,
    geometry_catalog_path: Path,
    curves_root: Path,
    processed_csv_path: Path,
    processed_meta_path: Path,
    overwrite_processed: bool = False,
    max_antennas: int = 0,
    grid_height: int = 10,
    grid_width: int = 10,
) -> dict[str, object]:
    processed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if processed_csv_path.exists() and processed_meta_path.exists() and not overwrite_processed:
        with processed_meta_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    geometries, geom_meta = parse_geometry_catalog(
        geometry_catalog_path,
        grid_height=grid_height,
        grid_width=grid_width,
    )
    curve_paths, duplicates = collect_curve_paths(curves_root)
    geometry_ids = set(geometries)
    curve_ids = set(curve_paths)
    scanned_curve_paths = len(curve_paths) + sum(len(paths) for paths in duplicates.values())
    missing_curve_ids = sorted(geometry_ids - curve_ids)

    matched_ids = sorted(geometry_ids & curve_ids)
    if max_antennas > 0:
        matched_ids = matched_ids[:max_antennas]
    if not matched_ids:
        raise RuntimeError(
            "No matching antenna ids were found between geometry and S11 files. "
            f"geometry_catalog_path={geometry_catalog_path}, "
            f"curves_root={curves_root}, "
            f"num_geometries={len(geometries)}, "
            f"num_curve_files={len(matched_ids)}, "
            f"num_curve_files_discovered={len(curve_paths)}"
        )

    freq_reference: np.ndarray | None = None
    skipped_ids: list[int] = []
    valid_ids: list[int] = []
    valid_rows = 0

    with processed_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for antenna_id in matched_ids:
            freq_ghz, s11_db = load_curve_csv(curve_paths[antenna_id])
            if freq_reference is None:
                freq_reference = freq_ghz
            elif len(freq_ghz) != len(freq_reference) or not np.allclose(
                freq_ghz,
                freq_reference,
                atol=1e-6,
            ):
                skipped_ids.append(antenna_id)
                continue

            geom_bits = geometries[antenna_id].astype(np.int32).tolist()
            row = geom_bits + [float(value) for value in s11_db.tolist()]
            writer.writerow(row)
            valid_rows += 1
            valid_ids.append(antenna_id)

    if freq_reference is None:
        raise RuntimeError("Failed to build a reference frequency axis from the S11 files.")

    summary = {
        "geometry_catalog_path": str(geometry_catalog_path),
        "curves_root": str(curves_root),
        "processed_csv_path": str(processed_csv_path),
        "num_geometries": len(geometries),
        "num_curve_files": len(matched_ids),
        "num_curve_files_discovered": len(curve_paths),
        "num_curve_paths_scanned": scanned_curve_paths,
        "num_duplicate_curve_ids": len(duplicates),
        "duplicate_curve_ids": sorted(duplicates),
        "missing_curve_ids": missing_curve_ids,
        "matched_antennas": valid_rows,
        "skipped_antennas": skipped_ids,
        "seq_len": int(len(freq_reference)),
        "freq_axis_ghz": [float(value) for value in freq_reference.tolist()],
        "grid_width": geom_meta["grid_width"],
        "grid_height": geom_meta["grid_height"],
        "matched_antenna_ids": valid_ids,
    }
    with processed_meta_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def preprocess_uploaded_dataset(
    *,
    geometry_catalog_path: Path,
    curves_root: Path,
    processed_csv_path: Path,
    processed_meta_path: Path,
    overwrite_processed: bool = False,
    max_antennas: int = 0,
    grid_height: int = 10,
    grid_width: int = 10,
) -> dict[str, object]:
    """Backward-compatible alias used by the PINN entrypoint."""

    return build_processed_csv_from_raw_data(
        geometry_catalog_path=geometry_catalog_path,
        curves_root=curves_root,
        processed_csv_path=processed_csv_path,
        processed_meta_path=processed_meta_path,
        overwrite_processed=overwrite_processed,
        max_antennas=max_antennas,
        grid_height=grid_height,
        grid_width=grid_width,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the raw patch antenna folder dataset into one processed CSV.",
    )
    parser.add_argument(
        "--geometry-catalog-path",
        type=Path,
        default=_default_geometry_catalog_path(),
    )
    parser.add_argument(
        "--curves-root",
        type=Path,
        default=RAW_DATA_ROOT,
    )
    parser.add_argument(
        "--processed-csv-path",
        type=Path,
        default=_default_processed_csv(),
    )
    parser.add_argument(
        "--processed-meta-path",
        type=Path,
        default=_default_processed_meta(),
    )
    parser.add_argument("--max-antennas", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--grid-height", type=int, default=10)
    parser.add_argument("--grid-width", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_processed_csv_from_raw_data(
        geometry_catalog_path=args.geometry_catalog_path,
        curves_root=args.curves_root,
        processed_csv_path=args.processed_csv_path,
        processed_meta_path=args.processed_meta_path,
        overwrite_processed=args.overwrite,
        max_antennas=args.max_antennas,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
