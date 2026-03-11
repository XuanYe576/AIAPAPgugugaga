"""Create result visualizations from training history and summary files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "metrics" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize training history and summary metrics for saved runs.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        help="Specific run directory under results/ to visualize. Can be used multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures will be saved.",
    )
    return parser.parse_args()


def discover_run_dirs(explicit: list[Path] | None) -> list[Path]:
    if explicit:
        return [path if path.is_absolute() else (REPO_ROOT / path) for path in explicit]

    run_dirs: list[Path] = []
    for path in sorted(RESULTS_ROOT.iterdir() if RESULTS_ROOT.exists() else []):
        if not path.is_dir():
            continue
        if (path / "history.csv").exists() or (path / "summary.json").exists():
            run_dirs.append(path)
    return run_dirs


def load_history(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []

    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
            if parsed:
                rows.append(parsed)
    return rows


def load_summary(path: Path) -> dict[str, float | str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required to create visualizations. "
            "Install it with: pip install matplotlib"
        ) from exc
    return plt


def grouped_metric_columns(rows: list[dict[str, float]], prefix: str) -> list[str]:
    if not rows:
        return []
    columns = [key for key in rows[0].keys() if key.startswith(prefix)]
    return [key for key in columns if key not in {f"{prefix}total"}]


def plot_history_panel(ax, rows: list[dict[str, float]], title: str) -> None:
    if not rows:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No history.csv found", ha="center", va="center")
        ax.axis("off")
        return

    epochs = [row["epoch"] for row in rows if "epoch" in row]
    train_total = [row.get("train_total", math.nan) for row in rows]
    val_total = [row.get("val_total", math.nan) for row in rows]
    ax.plot(epochs, train_total, label="Train total", linewidth=2.0)
    ax.plot(epochs, val_total, label="Val total", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_metric_panel(
    ax,
    rows: list[dict[str, float]],
    prefix: str,
    title: str,
) -> None:
    if not rows:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return

    columns = grouped_metric_columns(rows, prefix)
    epochs = [row["epoch"] for row in rows if "epoch" in row]
    if not columns:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No matching metrics", ha="center", va="center")
        ax.axis("off")
        return

    for column in columns:
        ax.plot(epochs, [row.get(column, math.nan) for row in rows], label=column)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_summary_panel(ax, summary: dict[str, float | str], title: str) -> None:
    numeric = {
        key: value
        for key, value in summary.items()
        if isinstance(value, (int, float))
    }
    if not numeric:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No summary.json found", ha="center", va="center")
        ax.axis("off")
        return

    keys = list(numeric.keys())
    values = [float(numeric[key]) for key in keys]
    ax.barh(keys, values, color="#2f6db3")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)


def plot_run_dashboard(run_dir: Path, output_dir: Path) -> Path:
    plt = load_pyplot()
    history_rows = load_history(run_dir / "history.csv")
    summary = load_summary(run_dir / "summary.json")

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    figure.suptitle(f"Run Dashboard: {run_dir.name}", fontsize=14)

    plot_history_panel(axes[0, 0], history_rows, "Train vs Validation Loss")
    plot_metric_panel(axes[0, 1], history_rows, "train_", "Training Metrics")
    plot_metric_panel(axes[1, 0], history_rows, "val_", "Validation Metrics")
    plot_summary_panel(axes[1, 1], summary, "Final Summary Metrics")

    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_dir.name}_dashboard.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_comparison(run_dirs: list[Path], output_dir: Path) -> Path | None:
    plt = load_pyplot()
    summaries = []
    for run_dir in run_dirs:
        summary = load_summary(run_dir / "summary.json")
        if summary:
            summaries.append((run_dir.name, summary))

    if len(summaries) < 2:
        return None

    preferred_keys = [
        "best_val_total",
        "test_total",
        "test_mae",
        "test_db_mae",
        "test_complex_mse",
        "test_passive",
    ]
    available_keys = [
        key
        for key in preferred_keys
        if any(isinstance(summary.get(key), (int, float)) for _, summary in summaries)
    ]
    if not available_keys:
        return None

    figure, axes = plt.subplots(1, len(available_keys), figsize=(5 * len(available_keys), 4))
    if len(available_keys) == 1:
        axes = [axes]

    run_names = [name for name, _ in summaries]
    for ax, key in zip(axes, available_keys):
        values = []
        for _, summary in summaries:
            value = summary.get(key)
            values.append(float(value) if isinstance(value, (int, float)) else math.nan)
        ax.bar(run_names, values, color="#c96f32")
        ax.set_title(key)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)

    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "run_comparison.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    run_dirs = discover_run_dirs(args.run_dir)
    if not run_dirs:
        raise SystemExit("No run directories with history.csv or summary.json were found.")

    outputs = []
    for run_dir in run_dirs:
        outputs.append(plot_run_dashboard(run_dir, args.output_dir))

    comparison = plot_comparison(run_dirs, args.output_dir)
    if comparison is not None:
        outputs.append(comparison)

    print("Saved visualizations:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
