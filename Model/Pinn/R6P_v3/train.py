from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[4]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from mainPAP.Model.Pinn.R6P_v3.config import ExperimentConfig, to_serializable
    from mainPAP.Model.Pinn.R6P_v3.data import DatasetBundle, _make_multi_dip_targets, build_dataloaders
    from mainPAP.Model.Pinn.R6P_v3.losses import LossBreakdown, compute_losses
    from mainPAP.Model.Pinn.R6P_v3.model import R6P_v3Model
else:
    from .config import ExperimentConfig, to_serializable
    from .data import DatasetBundle, _make_multi_dip_targets, build_dataloaders
    from .losses import LossBreakdown, compute_losses
    from .model import R6P_v3Model


@dataclass(frozen=True)
class RunMode:
    name: str
    stage1_only: bool
    use_adapter: bool


RUN_MODES: dict[str, RunMode] = {
    "stage1": RunMode(name="stage1", stage1_only=True, use_adapter=False),
    "dip_no_adapter": RunMode(name="dip_no_adapter", stage1_only=False, use_adapter=False),
    "full": RunMode(name="full", stage1_only=False, use_adapter=True),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def configure_cuda_attention(device: torch.device) -> None:
    if device.type != "cuda" or not hasattr(torch.backends, "cuda"):
        return
    cuda_backends = torch.backends.cuda
    for fn_name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp"):
        fn = getattr(cuda_backends, fn_name, None)
        if callable(fn):
            fn(False)
    math_fn = getattr(cuda_backends, "enable_math_sdp", None)
    if callable(math_fn):
        math_fn(True)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        if hasattr(torch, "amp"):
            return torch.amp.autocast("cuda")
        return torch.cuda.amp.autocast()
    return nullcontext()


def build_grad_scaler(device: torch.device, use_amp: bool) -> torch.cuda.amp.GradScaler | None:
    if use_amp and device.type == "cuda":
        if hasattr(torch, "amp"):
            return torch.amp.GradScaler("cuda")
        return torch.cuda.amp.GradScaler()
    return None


def embedding_variation(geometry_embedding: torch.Tensor) -> tuple[torch.Tensor, int]:
    embedding = geometry_embedding.detach().float()
    dimension_std = embedding.std(dim=0, unbiased=False)
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    variance = singular_values.square()
    total_variance = variance.sum()
    if not torch.isfinite(total_variance) or total_variance <= 0:
        return dimension_std, 0
    cumulative = variance.cumsum(dim=0) / total_variance
    effective_dimensions_99 = int((cumulative < 0.99).sum().item()) + 1
    return dimension_std, effective_dimensions_99


def log_embedding_variation(
    geometry_embedding: torch.Tensor,
    cfg: ExperimentConfig,
    run_mode: RunMode,
    epoch: int,
) -> None:
    dimension_std, effective_dimensions_99 = embedding_variation(geometry_embedding)
    std_values = dimension_std.cpu().tolist()
    print(f"[z_g:{run_mode.name}] epoch={epoch} std(dim=0)={std_values}")
    print(
        f"[z_g:{run_mode.name}] epoch={epoch} "
        f"std_mean={dimension_std.mean().item():.6g} "
        f"std_min={dimension_std.min().item():.6g} "
        f"std_max={dimension_std.max().item():.6g} "
        f"effective_dims_99={effective_dimensions_99}"
    )
    if not torch.isfinite(dimension_std).all():
        raise FloatingPointError("Non-finite values detected in z_g batch variation.")
    if dimension_std.mean().item() <= cfg.train.embedding_std_abort_threshold:
        raise RuntimeError(
            "Geometry encoder collapsed: mean z_g dimension std is below "
            f"{cfg.train.embedding_std_abort_threshold:.3g}."
        )
    required_dimensions = min(
        cfg.train.embedding_min_effective_dims_99,
        max(geometry_embedding.shape[0] - 1, 1),
    )
    if effective_dimensions_99 < required_dimensions:
        raise RuntimeError(
            "Geometry encoder is low-rank: "
            f"effective_dims_99={effective_dimensions_99}, required={required_dimensions}."
        )


def train_one_epoch(
    model: R6P_v3Model,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: ExperimentConfig,
    run_mode: RunMode,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
) -> dict[str, float]:
    model.train()
    totals = {
        "total": 0.0,
        "line": 0.0,
        "mask": 0.0,
        "iou": 0.0,
        "dip": 0.0,
        "count": 0.0,
        "geometry_reconstruction": 0.0,
        "embedding_variance": 0.0,
        "embedding_covariance": 0.0,
        "final": 0.0,
        "physics_aux": 0.0,
    }
    batch_count = 0
    for batch_idx, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, cfg.train.use_amp):
            outputs = model(
                batch["geometry"],
                stage1_only=run_mode.stage1_only,
                use_adapter=run_mode.use_adapter,
                hard_inference=False,
            )
            losses = compute_losses(outputs, batch, cfg.loss, stage1_only=run_mode.stage1_only)
        if batch_idx == 0:
            log_embedding_variation(outputs.geometry_embedding, cfg, run_mode, epoch)
        if not torch.isfinite(losses.total):
            components = losses.as_float_dict()
            raise FloatingPointError(
                f"Non-finite training loss at epoch={epoch} batch={batch_idx + 1}: {components}"
            )
        if scaler is not None:
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=cfg.train.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.total.backward()
            clip_grad_norm_(model.parameters(), max_norm=cfg.train.gradient_clip)
            optimizer.step()
        for key, value in losses.as_float_dict().items():
            totals[key] += value
        batch_count += 1
        if cfg.train.log_every_batches > 0 and (batch_idx + 1) % cfg.train.log_every_batches == 0:
            avg_total = totals["total"] / max(batch_count, 1)
            print(f"[train:{run_mode.name}] batch={batch_idx + 1} avg_total={avg_total:.6f}")
    return {key: value / max(batch_count, 1) for key, value in totals.items()}


def _dense_mask_iou_sum(pred_mask: torch.Tensor, target_mask: torch.Tensor, supervision: torch.Tensor) -> tuple[float, int]:
    """Sum of per-sample hard IoU (over supervised bins) for samples that have >=1 true dip bin."""
    pred = (pred_mask > 0.5).float() * supervision
    target = (target_mask > 0.5).float() * supervision
    intersection = (pred * target).sum(dim=1)
    union = ((pred + target) > 0.5).float().sum(dim=1)
    has_positive = target.sum(dim=1) > 0
    iou = torch.where(union > 0, intersection / union.clamp_min(1.0), torch.zeros_like(union))
    return float(iou[has_positive].sum().item()), int(has_positive.sum().item())


def _naive_dip_mask(coarse_line: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    """Baseline dip mask built the same way as the training targets, but read
    straight off the Stage-1 coarse line instead of the true curve — this is
    what "no dip specialist at all, just look at Stage 1" would predict."""
    coarse_np = coarse_line.detach().cpu().numpy()
    dummy_freq = np.arange(coarse_np.shape[1], dtype=np.float32)
    masks = np.stack(
        [
            _make_multi_dip_targets(
                curve=row,
                freq_axis_ghz=dummy_freq,
                target_radius_bins=cfg.data.dip_target_radius_bins,
                min_depth_db=cfg.data.min_dip_depth_db,
                min_separation_bins=cfg.data.dip_min_separation_bins,
            )["dip_mask"]
            for row in coarse_np
        ]
    )
    return torch.from_numpy(masks).to(coarse_line.device)


def evaluate(
    model: R6P_v3Model,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: ExperimentConfig,
    run_mode: RunMode,
) -> dict[str, float]:
    model.eval()
    loss_totals = {
        "total": 0.0,
        "line": 0.0,
        "mask": 0.0,
        "iou": 0.0,
        "dip": 0.0,
        "count": 0.0,
        "geometry_reconstruction": 0.0,
        "embedding_variance": 0.0,
        "embedding_covariance": 0.0,
        "final": 0.0,
        "physics_aux": 0.0,
    }
    batches = 0
    curve_sse = 0.0
    line_sse = 0.0
    point_count = 0
    expert_iou_sum = 0.0
    naive_iou_sum = 0.0
    iou_sample_count = 0
    expert_depth_abs_error = 0.0
    depth_anchor_count = 0.0
    count_correct = 0.0
    count_total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                batch["geometry"],
                stage1_only=run_mode.stage1_only,
                use_adapter=run_mode.use_adapter,
                hard_inference=True,
            )
            losses = compute_losses(outputs, batch, cfg.loss, stage1_only=run_mode.stage1_only)
            for key, value in losses.as_float_dict().items():
                loss_totals[key] += value
            batches += 1

            target_curve = batch["curve"]
            curve_sse += (outputs.final_curve - target_curve).pow(2).sum().item()
            line_sse += (outputs.coarse_line - target_curve).pow(2).sum().item()
            point_count += int(target_curve.numel())

            if run_mode.stage1_only:
                continue

            target_mask = batch["dip_mask"]
            supervision = batch["dip_supervision_mask"] * batch["label_available"].unsqueeze(-1)
            expert_prob = torch.sigmoid(outputs.dip_presence_logits)
            iou_sum, count = _dense_mask_iou_sum(expert_prob, target_mask, supervision)
            expert_iou_sum += iou_sum
            iou_sample_count += count

            naive_mask = _naive_dip_mask(outputs.coarse_line, cfg)
            naive_iou_sum_batch, _ = _dense_mask_iou_sum(naive_mask, target_mask, supervision)
            naive_iou_sum += naive_iou_sum_batch

            anchor_mask = batch["dip_anchor_mask"] * batch["label_available"].unsqueeze(-1)
            expert_depth_abs_error += ((outputs.dip_depth_db - batch["dip_depth_db"]).abs() * anchor_mask).sum().item()
            depth_anchor_count += anchor_mask.sum().item()

            max_count = outputs.count_logits.shape[-1] - 1
            target_count = batch["dip_count"].round().long().clamp(0, max_count)
            pred_count = outputs.count_logits.argmax(dim=-1)
            count_correct += (pred_count == target_count).float().sum().item()
            count_total += float(target_count.shape[0])

    summary = {key: value / max(batches, 1) for key, value in loss_totals.items()}
    summary.update(
        {
            "line_mse": line_sse / max(point_count, 1),
            "final_mse": curve_sse / max(point_count, 1),
            "dip_mask_iou_expert": expert_iou_sum / max(iou_sample_count, 1),
            "dip_mask_iou_naive": naive_iou_sum / max(iou_sample_count, 1),
            "dip_depth_mae_expert": expert_depth_abs_error / max(depth_anchor_count, 1.0),
            "dip_count_acc": count_correct / max(count_total, 1.0),
            "num_labeled_eval_samples": float(iou_sample_count),
        }
    )
    return summary


def save_history(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the R6P_v3 two-stage surrogate.")
    parser.add_argument("--mode", choices=sorted(RUN_MODES), default="full")
    parser.add_argument("--processed-csv", type=Path, default=None)
    parser.add_argument("--processed-meta", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--synthetic-size", type=int, default=None)
    parser.add_argument("--min-dip-depth-db", type=float, default=None)
    parser.add_argument("--dip-target-radius-bins", type=int, default=None)
    parser.add_argument("--mask-positive-weight", type=float, default=None)
    parser.add_argument("--use-synthetic-only", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--use-physics-aux", action="store_true")
    parser.add_argument(
        "--prune-dip-bins-from-line",
        action="store_true",
        help="Exclude dip-touched bins from Stage 1's line loss, so it only ever fits the smooth baseline.",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Warm-start all weights from a checkpoint (e.g. a --prune-dip-bins-from-line stage1 pretrain run).",
    )
    parser.add_argument(
        "--freeze-stage1",
        action="store_true",
        help="Freeze encoder+decoder parameters (use with --init-from to only train Stage 2 on a pretrained baseline).",
    )
    return parser


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if args.processed_csv is not None:
        cfg.data.processed_csv_path = args.processed_csv
    if args.processed_meta is not None:
        cfg.data.processed_meta_path = args.processed_meta
    if args.results_dir is not None:
        cfg.train.output_dir = args.results_dir
    if args.epochs is not None:
        cfg.train.epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.train.batch_size = int(args.batch_size)
    if args.lr is not None:
        cfg.train.lr = float(args.lr)
    if args.weight_decay is not None:
        cfg.train.weight_decay = float(args.weight_decay)
    if args.device is not None:
        cfg.train.device = args.device
    if args.max_samples is not None:
        cfg.data.max_samples = int(args.max_samples)
    if args.synthetic_size is not None:
        cfg.data.synthetic_size = int(args.synthetic_size)
    if args.min_dip_depth_db is not None:
        cfg.data.min_dip_depth_db = float(args.min_dip_depth_db)
    if args.dip_target_radius_bins is not None:
        cfg.data.dip_target_radius_bins = int(args.dip_target_radius_bins)
    if args.mask_positive_weight is not None:
        cfg.loss.mask_positive_weight = float(args.mask_positive_weight)
    if args.use_synthetic_only:
        cfg.data.use_real_dataset = False
    if args.disable_amp:
        cfg.train.use_amp = False
    if args.use_physics_aux:
        cfg.loss.use_physics_aux = True
    if args.prune_dip_bins_from_line:
        cfg.loss.prune_dip_bins_from_line = True
    return cfg


def run_training(
    cfg: ExperimentConfig,
    run_mode: RunMode,
    *,
    init_from: Path | None = None,
    freeze_stage1: bool = False,
) -> dict[str, Any]:
    set_seed(cfg.train.random_seed)
    bundle: DatasetBundle = build_dataloaders(cfg)
    device = get_device(cfg.train.device)
    configure_cuda_attention(device)
    model = R6P_v3Model(cfg, freq_axis_ghz=bundle.freq_axis_ghz).to(device)
    if init_from is not None:
        state = torch.load(init_from, map_location=device)
        model.load_state_dict(state, strict=True)
        print(f"[{run_mode.name}] warm-started all weights from {init_from}")
    if freeze_stage1:
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = False
        print(f"[{run_mode.name}] encoder+decoder frozen; only Stage 2 (+ count adapter/fusion) will train")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = build_grad_scaler(device, cfg.train.use_amp)
    cfg.train.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience_count = 0

    for epoch in range(cfg.train.epochs):
        train_metrics = train_one_epoch(
            model,
            bundle.train_loader,
            optimizer,
            device,
            cfg,
            run_mode,
            scaler,
            epoch=epoch + 1,
        )
        val_metrics = evaluate(model, bundle.val_loader, device, cfg, run_mode)
        record = {"epoch": float(epoch + 1)}
        record.update({f"train_{k}": v for k, v in train_metrics.items()})
        record.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(record)
        print(
            f"[{run_mode.name}] epoch={epoch + 1}/{cfg.train.epochs} "
            f"train_total={train_metrics['total']:.6f} val_total={val_metrics['total']:.6f} "
            f"val_final_mse={val_metrics['final_mse']:.6f}"
        )
        save_history(cfg.train.history_path, history)
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            patience_count = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, cfg.train.checkpoint_path)
        else:
            patience_count += 1
            if patience_count >= cfg.train.patience:
                print(f"[{run_mode.name}] early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, bundle.test_loader, device, cfg, run_mode)
    payload = {
        "model_name": cfg.name,
        "run_mode": run_mode.name,
        "device": str(device),
        "data_source": bundle.source_name,
        "num_samples": bundle.num_samples,
        "num_labeled_samples": bundle.num_labeled_samples,
        "best_val_total": best_val,
        "test_metrics": test_metrics,
        "config": to_serializable(cfg),
    }
    save_history(cfg.train.history_path, history)
    save_json(cfg.train.summary_path, payload)
    return payload


def format_metrics_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "name",
        "final_mse",
        "line_mse",
        "dip_mask_iou_expert",
        "dip_mask_iou_naive",
        "dip_depth_mae_expert",
        "dip_count_acc",
    ]
    widths = {header: len(header) for header in headers}
    printable: list[dict[str, str]] = []
    for row in rows:
        metrics = row["test_metrics"]
        printable_row = {
            "name": str(row["run_mode"]),
            "final_mse": f"{metrics['final_mse']:.6f}",
            "line_mse": f"{metrics['line_mse']:.6f}",
            "dip_mask_iou_expert": f"{metrics['dip_mask_iou_expert']:.4f}",
            "dip_mask_iou_naive": f"{metrics['dip_mask_iou_naive']:.4f}",
            "dip_depth_mae_expert": f"{metrics['dip_depth_mae_expert']:.4f}",
            "dip_count_acc": f"{metrics.get('dip_count_acc', 0.0):.4f}",
        }
        printable.append(printable_row)
        for header, value in printable_row.items():
            widths[header] = max(widths[header], len(value))
    line = " | ".join(header.ljust(widths[header]) for header in headers)
    sep = "-+-".join("-" * widths[header] for header in headers)
    body = [
        " | ".join(row[header].ljust(widths[header]) for header in headers)
        for row in printable
    ]
    return "\n".join([line, sep, *body])


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = build_config_from_args(args)
    run_mode = RUN_MODES[args.mode]
    summary = run_training(cfg, run_mode, init_from=args.init_from, freeze_stage1=args.freeze_stage1)
    print(format_metrics_table([summary]))


if __name__ == "__main__":
    main()
