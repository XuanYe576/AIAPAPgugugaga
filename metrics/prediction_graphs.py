"""Prediction and training-curve plotting helpers."""

from __future__ import annotations

from pathlib import Path

from metrics.plotting import load_pyplot


def _plot_prediction_pair(
    plt,
    x_values,
    target_values,
    pred_values,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(x_values, target_values, label="Target", linewidth=2.0)
    plt.plot(x_values, pred_values, label="Prediction", linewidth=2.0, linestyle="--")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_loss_curve(
    history_rows: list[dict[str, float]],
    output_path: Path,
    title: str = "R5 Physics-Informed Training Curve",
) -> Path:
    plt = load_pyplot()
    epochs = [row["epoch"] for row in history_rows]
    train_loss = [row["train_total"] for row in history_rows]
    val_loss = [row["val_total"] for row in history_rows]
    plt.figure()
    plt.plot(epochs, train_loss, label="Train total loss")
    plt.plot(epochs, val_loss, label="Validation total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def save_complex_prediction_graphs(
    model,
    loader,
    freq_axis_hz,
    gamma_db_fn,
    output_dir: Path,
    split: str,
    plot_count: int,
    device: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if plot_count <= 0:
        return output_dir

    import torch

    plt = load_pyplot()
    x_values = (freq_axis_hz.detach().cpu().numpy() / 1.0e9).tolist()
    saved = 0
    model.eval()

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb, freq_axis_hz)
            pred_db = gamma_db_fn(outputs["gamma"]).detach().cpu().numpy()
            target_db = gamma_db_fn(yb).detach().cpu().numpy()

            for sample_idx in range(pred_db.shape[0]):
                if saved >= plot_count:
                    return output_dir
                _plot_prediction_pair(
                    plt=plt,
                    x_values=x_values,
                    target_values=target_db[sample_idx],
                    pred_values=pred_db[sample_idx],
                    output_path=output_dir / f"{split}_sample_{saved + 1:04d}.png",
                    title=f"{split.upper()} Prediction {saved + 1:04d}",
                    ylabel="S11 (dB)",
                )
                saved += 1

    return output_dir


def save_real_channel_prediction_graphs(
    model,
    loader,
    freq_axis_hz,
    output_dir: Path,
    split: str,
    plot_count: int,
    device: str,
    channel_idx: int = 0,
    ylabel: str = "S11 (dB)",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if plot_count <= 0:
        return output_dir

    import torch

    plt = load_pyplot()
    x_values = (freq_axis_hz.detach().cpu().numpy() / 1.0e9).tolist()
    saved = 0
    model.eval()

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb, freq_axis_hz)
            pred_values = outputs["gamma"][..., channel_idx].detach().cpu().numpy()
            target_values = yb[..., channel_idx].detach().cpu().numpy()

            for sample_idx in range(pred_values.shape[0]):
                if saved >= plot_count:
                    return output_dir
                _plot_prediction_pair(
                    plt=plt,
                    x_values=x_values,
                    target_values=target_values[sample_idx],
                    pred_values=pred_values[sample_idx],
                    output_path=output_dir / f"{split}_sample_{saved + 1:04d}.png",
                    title=f"{split.upper()} Prediction {saved + 1:04d}",
                    ylabel=ylabel,
                )
                saved += 1

    return output_dir


def save_scalar_prediction_graphs(
    model,
    base,
    antenna_indices: list[int],
    output_dir: Path,
    split: str,
    plot_count: int,
    device: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if plot_count <= 0:
        return output_dir

    import torch

    plt = load_pyplot()
    freq_norm = base.freq_axis_norm.to(device).unsqueeze(-1)
    x_values = base.freq_axis_ghz.detach().cpu().numpy()
    saved = 0
    model.eval()

    with torch.no_grad():
        for antenna_idx in antenna_indices:
            if saved >= plot_count:
                break
            geom_bits = base.geometry[antenna_idx].to(device)
            geom_batch = geom_bits.unsqueeze(0).expand(base.seq_len, -1)
            pred_db = model(geom_batch, freq_norm).detach().cpu().numpy()
            target_db = base.curves_db[antenna_idx].detach().cpu().numpy()
            antenna_id = base.antenna_ids[antenna_idx]
            _plot_prediction_pair(
                plt=plt,
                x_values=x_values,
                target_values=target_db,
                pred_values=pred_db,
                output_path=output_dir / f"{split}_antenna_{int(antenna_id):05d}.png",
                title=f"{split.upper()} Antenna {antenna_id}",
                ylabel="S11 (dB)",
            )
            saved += 1

    return output_dir
