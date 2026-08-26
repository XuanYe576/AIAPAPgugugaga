from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig
from .data import DatasetBundle, build_dataloaders
from .model import R6P_v4Model
from .train import embedding_variation, get_device, set_seed


def _restore_value(value: Any, current: Any) -> Any:
    if isinstance(current, Path):
        return Path(value)
    if isinstance(current, tuple):
        return tuple(tuple(item) if isinstance(item, list) else item for item in value)
    return value


def load_run_config(results_dir: Path) -> ExperimentConfig:
    cfg = ExperimentConfig()
    summary_path = results_dir / cfg.train.summary_filename
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        saved_config = summary.get("config", {})
        for section_name in ("data", "physics", "model", "loss", "train"):
            section = getattr(cfg, section_name)
            for key, value in saved_config.get(section_name, {}).items():
                if hasattr(section, key):
                    setattr(section, key, _restore_value(value, getattr(section, key)))
    cfg.train.output_dir = results_dir
    return cfg


def tensor_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
        "p05": float(torch.quantile(values, 0.05)),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
    }


def correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.detach().float()
    second = second.detach().float()
    if first.std(unbiased=False) == 0 or second.std(unbiased=False) == 0:
        return 0.0
    return float(torch.corrcoef(torch.stack((first, second)))[0, 1])


def normalized_distribution(values: torch.Tensor, classes: int) -> list[float]:
    counts = torch.bincount(values.long(), minlength=classes).float()
    return (counts / counts.sum().clamp_min(1.0)).tolist()


def minimum_summary(curves: torch.Tensor, freq_axis_ghz: torch.Tensor) -> dict[str, Any]:
    indices = curves.argmin(dim=1)
    frequencies = freq_axis_ghz[indices]
    histogram = torch.bincount(indices, minlength=freq_axis_ghz.numel())
    mode_index = int(histogram.argmax())
    return {
        "frequency_ghz": tensor_stats(frequencies),
        "unique_bins": int(indices.unique().numel()),
        "mode_frequency_ghz": float(freq_axis_ghz[mode_index]),
        "mode_fraction": float(histogram[mode_index] / indices.numel()),
    }


def select_loader(bundle: DatasetBundle, split: str) -> torch.utils.data.DataLoader:
    return {
        "train": bundle.train_loader,
        "val": bundle.val_loader,
        "test": bundle.test_loader,
    }[split]


def collect_predictions(
    model: R6P_v4Model,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    collected: dict[str, list[torch.Tensor]] = {
        "geometry": [],
        "z_g": [],
        "coarse": [],
        "soft_final": [],
        "hard_final": [],
        "existence": [],
        "location": [],
        "pred_count": [],
        "target_count": [],
        "label_available": [],
        "antenna_id": [],
    }
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            geometry = batch["geometry"].to(device)
            outputs = model(geometry, hard_inference=False)
            outputs_hard = model(geometry, hard_inference=True)
            existence_prob = outputs.dip_existence_logits.sigmoid()
            collected["geometry"].append(batch["geometry"].flatten(start_dim=1).cpu())
            collected["z_g"].append(outputs.geometry_embedding.cpu())
            collected["coarse"].append(outputs.coarse_line.cpu())
            collected["soft_final"].append(outputs.final_curve.cpu())
            collected["hard_final"].append(outputs_hard.final_curve.cpu())
            collected["existence"].append(existence_prob.cpu())
            collected["location"].append(outputs.dip_location.cpu())
            collected["pred_count"].append((existence_prob > 0.5).sum(dim=-1).cpu())
            collected["target_count"].append(batch["dip_count"].round().long().cpu())
            collected["label_available"].append(batch["label_available"].cpu())
            collected["antenna_id"].append(batch["antenna_id"].cpu())
            if (batch_index + 1) % 25 == 0:
                print(f"[preflight] batches={batch_index + 1}/{len(loader)}")
    return {key: torch.cat(parts) for key, parts in collected.items()}


def build_diagnostic(
    tensors: dict[str, torch.Tensor],
    freq_axis_ghz: torch.Tensor,
    cfg: ExperimentConfig,
    checkpoint_path: Path,
    split: str,
    device: torch.device,
    pair_count: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    z_g = tensors["z_g"]
    dimension_std, effective_dims_99 = embedding_variation(z_g)
    centered = z_g - z_g.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    variance_ratio = singular_values.square()
    variance_ratio = variance_ratio / variance_ratio.sum().clamp_min(1.0e-12)

    generator = torch.Generator().manual_seed(cfg.train.random_seed + 20_260_720)
    requested_pairs = min(pair_count, max(1, z_g.shape[0] * 2))
    first = torch.randint(z_g.shape[0], (requested_pairs,), generator=generator)
    second = torch.randint(z_g.shape[0], (requested_pairs,), generator=generator)
    distinct = first != second
    first = first[distinct]
    second = second[distinct]
    geometry_distance = (
        tensors["geometry"][first] != tensors["geometry"][second]
    ).float().mean(dim=1)
    z_distance = (z_g[first] - z_g[second]).square().mean(dim=1).sqrt()
    output_distances = {
        name: (tensors[name][first] - tensors[name][second]).square().mean(dim=1).sqrt()
        for name in ("coarse", "soft_final", "hard_final", "existence")
    }

    max_count_class = int(cfg.model.num_queries)
    target_count = tensors["target_count"]
    clipped_target_count = target_count.clamp(0, max_count_class)
    zero_dip = target_count == 0

    mean_std_passed = bool(dimension_std.mean() > cfg.train.embedding_std_abort_threshold)
    rank_passed = bool(effective_dims_99 >= cfg.train.embedding_min_effective_dims_99)
    payload: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "num_samples": int(z_g.shape[0]),
        "device": str(device),
        "gate": {
            "passed": mean_std_passed and rank_passed,
            "mean_std_threshold": cfg.train.embedding_std_abort_threshold,
            "min_effective_dims_99": cfg.train.embedding_min_effective_dims_99,
            "passed_mean_std": mean_std_passed,
            "passed_effective_dims": rank_passed,
        },
        "z_g": {
            "dimension_std": dimension_std.tolist(),
            "std_mean": float(dimension_std.mean()),
            "std_min": float(dimension_std.min()),
            "std_max": float(dimension_std.max()),
            "dimensions_above_1e-3": int((dimension_std > 1.0e-3).sum()),
            "dimensions_above_1e-2": int((dimension_std > 1.0e-2).sum()),
            "effective_dims_99": effective_dims_99,
            "top_variance_ratios": variance_ratio[:10].tolist(),
        },
        "pairwise_sensitivity": {
            "pairs": int(first.numel()),
            "geometry_hamming_fraction": tensor_stats(geometry_distance),
            "z_rms_distance": tensor_stats(z_distance),
            "coarse_rms_distance_db": tensor_stats(output_distances["coarse"]),
            "soft_final_rms_distance_db": tensor_stats(output_distances["soft_final"]),
            "hard_final_rms_distance_db": tensor_stats(output_distances["hard_final"]),
            "existence_rms_distance": tensor_stats(output_distances["existence"]),
            "corr_geometry_z": correlation(geometry_distance, z_distance),
            "corr_geometry_coarse": correlation(geometry_distance, output_distances["coarse"]),
            "corr_geometry_soft_final": correlation(geometry_distance, output_distances["soft_final"]),
            "corr_geometry_hard_final": correlation(geometry_distance, output_distances["hard_final"]),
            "corr_geometry_existence": correlation(geometry_distance, output_distances["existence"]),
        },
        "prediction_variation": {
            "coarse_min": minimum_summary(tensors["coarse"], freq_axis_ghz),
            "soft_final_min": minimum_summary(tensors["soft_final"], freq_axis_ghz),
            "hard_final_min": minimum_summary(tensors["hard_final"], freq_axis_ghz),
            "coarse_pointwise_std_mean_db": float(tensors["coarse"].std(dim=0, unbiased=False).mean()),
            "soft_final_pointwise_std_mean_db": float(
                tensors["soft_final"].std(dim=0, unbiased=False).mean()
            ),
            "hard_final_pointwise_std_mean_db": float(
                tensors["hard_final"].std(dim=0, unbiased=False).mean()
            ),
            "existence_pointwise_std_mean": float(tensors["existence"].std(dim=0, unbiased=False).mean()),
            "existence_mean": float(tensors["existence"].mean()),
            "existence_max_mean": float(tensors["existence"].max(dim=1).values.mean()),
        },
        "label_and_count": {
            "zero_dip_fraction": float(zero_dip.float().mean()),
            "zero_dip_marked_available_fraction": float(
                tensors["label_available"][zero_dip].float().mean()
            ),
            "unavailable_fraction": float((tensors["label_available"] <= 0.5).float().mean()),
            "target_overflow_fraction": float((target_count > max_count_class).float().mean()),
            "target_clipped_0_to_max": normalized_distribution(
                clipped_target_count, max_count_class + 1
            ),
            "predicted_0_to_max": normalized_distribution(
                tensors["pred_count"].clamp(0, max_count_class), max_count_class + 1
            ),
            "raw_target_0_to_max_observed": normalized_distribution(
                target_count, int(target_count.max()) + 1
            ),
        },
    }
    plot_tensors = {
        "dimension_std": dimension_std,
        "geometry_distance": geometry_distance,
        "z_distance": z_distance,
        "coarse": tensors["coarse"],
        "soft_final": tensors["soft_final"],
        "hard_final": tensors["hard_final"],
        "existence": tensors["existence"],
        "location": tensors["location"],
        "target_count": clipped_target_count,
        "pred_count": tensors["pred_count"],
    }
    return payload, plot_tensors


def save_plot(
    path: Path,
    payload: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    freq_axis_ghz: torch.Tensor,
    num_queries: int,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(17, 10))
    std = tensors["dimension_std"].sort().values
    axes[0, 0].plot(std.numpy(), color="tab:blue")
    axes[0, 0].axhline(payload["gate"]["mean_std_threshold"], color="tab:red", linestyle="--")
    axes[0, 0].set_title("Sorted z_g std(dim=0)")
    axes[0, 0].set_xlabel("Embedding dimension")
    axes[0, 0].set_ylabel("Std")

    pair_limit = min(3000, tensors["geometry_distance"].numel())
    axes[0, 1].scatter(
        tensors["geometry_distance"][:pair_limit].numpy(),
        tensors["z_distance"][:pair_limit].numpy(),
        s=6,
        alpha=0.25,
    )
    axes[0, 1].set_title(
        f"Geometry vs z_g distance (r={payload['pairwise_sensitivity']['corr_geometry_z']:.3f})"
    )
    axes[0, 1].set_xlabel("Geometry Hamming fraction")
    axes[0, 1].set_ylabel("z_g RMS distance")

    for key, label in (
        ("coarse", "Stage 1"),
        ("soft_final", "Soft fused"),
        ("hard_final", "Hard fused"),
    ):
        axes[0, 2].plot(
            freq_axis_ghz.numpy(),
            tensors[key].std(dim=0, unbiased=False).numpy(),
            label=label,
        )
    axes[0, 2].set_title("Across-geometry output std")
    axes[0, 2].set_xlabel("Frequency (GHz)")
    axes[0, 2].set_ylabel("Std (dB)")
    axes[0, 2].legend()

    classes = torch.arange(num_queries + 1)
    target_distribution = torch.tensor(payload["label_and_count"]["target_clipped_0_to_max"])
    predicted_distribution = torch.tensor(payload["label_and_count"]["predicted_0_to_max"])
    axes[1, 0].bar(classes.numpy() - 0.18, target_distribution.numpy(), width=0.36, label="Target")
    axes[1, 0].bar(classes.numpy() + 0.18, predicted_distribution.numpy(), width=0.36, label="Predicted")
    axes[1, 0].set_title("Count distribution (last class is overflow-clipped)")
    axes[1, 0].set_xlabel("Dip count class")
    axes[1, 0].set_ylabel("Fraction")
    axes[1, 0].legend()

    bins = torch.arange(freq_axis_ghz.numel() + 1).numpy() - 0.5
    for key, label in (
        ("coarse", "Stage 1"),
        ("soft_final", "Soft fused"),
        ("hard_final", "Hard fused"),
    ):
        axes[1, 1].hist(
            tensors[key].argmin(dim=1).numpy(),
            bins=bins,
            histtype="step",
            density=True,
            label=label,
        )
    axes[1, 1].set_title("Predicted minimum-frequency bins")
    axes[1, 1].set_xlabel("Frequency-bin index")
    axes[1, 1].set_ylabel("Fraction")
    axes[1, 1].legend()

    # existence is per query slot, not per frequency bin, so there is no fixed
    # x-axis to plot it against directly — instead scatter each query's own
    # predicted location (converted to GHz) against its existence probability,
    # pooled across the whole sample set.
    freq_min = float(freq_axis_ghz[0].item())
    freq_max = float(freq_axis_ghz[-1].item())
    location_ghz = freq_min + tensors["location"] * (freq_max - freq_min)
    scatter_limit = min(20_000, location_ghz.numel())
    flat_location = location_ghz.reshape(-1)[:scatter_limit]
    flat_existence = tensors["existence"].reshape(-1)[:scatter_limit]
    axes[1, 2].scatter(flat_location.numpy(), flat_existence.numpy(), s=4, alpha=0.15, color="tab:purple")
    axes[1, 2].axhline(0.5, color="black", linestyle=":", linewidth=1.0)
    axes[1, 2].set_title("Query existence probability vs. predicted location")
    axes[1, 2].set_xlabel("Predicted location (GHz)")
    axes[1, 2].set_ylabel("Existence probability")

    gate_status = "PASS" if payload["gate"]["passed"] else "FAIL"
    figure.suptitle(
        f"R6P_v4 Geometry Sensitivity Preflight: {gate_status} | "
        f"std_mean={payload['z_g']['std_mean']:.3f}, "
        f"effective_dims_99={payload['z_g']['effective_dims_99']}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the R6P_v4 geometry-sensitivity preflight gate.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pair-count", type=int, default=20_000)
    parser.add_argument("--output-prefix", default="r6p_v4_geometry_sensitivity_preflight")
    parser.add_argument("--no-fail-on-collapse", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_run_config(args.results_dir)
    set_seed(cfg.train.random_seed)
    bundle = build_dataloaders(cfg)
    loader = select_loader(bundle, args.split)
    device = get_device(args.device)
    checkpoint_path = args.checkpoint or cfg.train.checkpoint_path
    print(f"[preflight] device={device} split={args.split} samples={len(loader.dataset)}")

    model = R6P_v4Model(cfg, bundle.freq_axis_ghz).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    tensors = collect_predictions(model, loader, device)
    payload, plot_tensors = build_diagnostic(
        tensors=tensors,
        freq_axis_ghz=bundle.freq_axis_ghz,
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        split=args.split,
        device=device,
        pair_count=args.pair_count,
    )

    output_dir = args.results_dir / "figures"
    json_path = output_dir / f"{args.output_prefix}.json"
    plot_path = output_dir / f"{args.output_prefix}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_plot(plot_path, payload, plot_tensors, bundle.freq_axis_ghz, cfg.model.num_queries)

    status = "PASS" if payload["gate"]["passed"] else "FAIL"
    print(
        f"[preflight] {status}: std_mean={payload['z_g']['std_mean']:.6g} "
        f"std_min={payload['z_g']['std_min']:.6g} "
        f"effective_dims_99={payload['z_g']['effective_dims_99']}"
    )
    print(
        "[preflight] sensitivity: "
        f"corr(geometry,z_g)={payload['pairwise_sensitivity']['corr_geometry_z']:.4f} "
        f"soft_final_pair_rms={payload['pairwise_sensitivity']['soft_final_rms_distance_db']['mean']:.4f} dB"
    )
    print(f"[preflight] wrote {json_path}")
    print(f"[preflight] wrote {plot_path}")
    if not payload["gate"]["passed"] and not args.no_fail_on_collapse:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
