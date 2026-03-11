"""R5.5B-s: 61-point complex entry point for the full R5 architecture.

This keeps the cleaned data/training pipeline from `patch_antenna_ai_r5.py`,
but freezes the model defaults to the formal spec-oriented setup instead of
the broader backward-compatible baseline configuration.

Key choices locked here:
- 61 frequency points
- complex output only: [Re(Gamma), Im(Gamma)]
- 3-layer GATv2 encoder with 4 heads
- k-NN graph with K=8 and geometric edge features
- geometry-conditioned FEDformer decoder
- spec-sized hidden layers, adapted to native 61-point runs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import patch_antenna_ai_r5 as base


def _default_output_dir() -> Path:
    return base.REPO_ROOT / "results" / "R5.5B-s"


def make_spec_config() -> base.Config:
    cfg = base.Config()

    # Freeze the architecture, but target the available 61-point complex simulations.
    cfg.seq_len = 61
    cfg.output_mode = "complex_61"
    cfg.batch_size = 32
    cfg.lr = 2e-3
    cfg.optimizer_name = "adamw"
    cfg.weight_decay = 1e-4
    cfg.min_lr_scale = 0.05
    cfg.epochs = 320
    cfg.train_ratio = 0.70
    cfg.val_ratio = 0.15

    cfg.geometry_embedding_dim = 256
    cfg.gnn_hidden_dim = 128
    cfg.gnn_layers = 3
    cfg.gnn_heads = 4
    cfg.edge_mlp_hidden = 64
    cfg.node_update_hidden = 128
    cfg.graph_k = 8

    cfg.dmodel = 256
    cfg.decoder_heads = 4
    cfg.decoder_blocks = 2
    cfg.refinement_layers = 1
    cfg.decoder_ffn_hidden = 512
    cfg.dropout = 0.10

    cfg.freq_hz_start = 1.0e9
    cfg.freq_hz_stop = 6.0e9
    cfg.use_augmentation = True
    cfg.bit_flip_prob = 0.01
    cfg.block_jitter_prob = 0.15
    cfg.max_block_size = 2

    # 61 points are the final target here, not a curriculum stage on the way to 122.
    cfg.curriculum_epochs = 0
    cfg.curriculum_stride = 1
    cfg.patience = 30
    cfg.gradient_clip = 1.0
    cfg.log_every_batches = 10
    cfg.warmup_ratio = 0.10
    cfg.use_amp = True

    cfg.loss_complex = 1.0
    cfg.loss_db = 0.25
    cfg.loss_notch = 0.10
    cfg.loss_passive = 0.10
    cfg.loss_smooth = 0.02
    cfg.notch_sigma_db = 4.0
    cfg.passive_limit = 1.0

    cfg.output_dir = _default_output_dir()
    cfg.weights_filename = "r55bs_best.pt"
    cfg.checkpoint_filename = "r55bs_best.ckpt"
    cfg.history_filename = "history.csv"
    cfg.loss_plot_filename = "loss_curve.png"
    cfg.summary_filename = "summary.json"
    cfg.config_filename = "config.json"
    return cfg


def parse_args() -> base.Config:
    cfg = make_spec_config()
    parser = argparse.ArgumentParser(
        description="Train the spec-locked R5.5B-s GATv2 + FEDformer model."
    )
    parser.add_argument("--csv-path", type=Path, default=cfg.csv_path)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--optimizer", choices=["adamw", "adagrad"], default=cfg.optimizer_name)
    parser.add_argument("--device", type=str, default=cfg.device)
    parser.add_argument("--output-dir", type=Path, default=cfg.output_dir)
    parser.add_argument("--seed", type=int, default=cfg.random_seed)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--graph-k", type=int, default=cfg.graph_k)
    parser.add_argument("--log-every-batches", type=int, default=cfg.log_every_batches)
    parser.add_argument("--eval-split", choices=["val", "test"])
    parser.add_argument("--prediction-plot-count", type=int, default=cfg.prediction_plot_count)
    args = parser.parse_args()

    cfg.csv_path = args.csv_path
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.optimizer_name = args.optimizer
    cfg.device = args.device
    cfg.output_dir = args.output_dir
    cfg.random_seed = args.seed
    cfg.use_augmentation = not args.no_augmentation
    cfg.graph_k = args.graph_k
    cfg.log_every_batches = args.log_every_batches
    cfg.eval_split = args.eval_split or ""
    cfg.prediction_plot_count = max(0, args.prediction_plot_count)
    return cfg


def main() -> None:
    cfg = parse_args()
    base.set_seed(cfg.random_seed)

    if cfg.eval_split:
        metrics = base.run_saved_evaluation(cfg, cfg.eval_split)
        print(f"Evaluation split: {cfg.eval_split}")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        return

    print("R5.5B-s configuration:")
    for key, value in asdict(cfg).items():
        print(f"  {key}: {value}")

    dataset = base.AntennaDataset(cfg.csv_path, input_dim=cfg.input_dim, output_mode=cfg.output_mode)
    base.sync_config_from_dataset(cfg, dataset)
    if dataset.layout.name != "complex_61":
        raise ValueError(
            "R5.5B-s is locked to 61-point complex targets. "
            f"Loaded layout: {dataset.layout.name}"
        )

    base.inspect_dataset(dataset)
    device = base.get_device(cfg.device)
    print(f"Using device: {device}")
    base.inspect_model(cfg, device)

    summary, history_rows = base.train_model(cfg)
    base.plot_loss_curve(history_rows, cfg.loss_plot_path)
    print(f"Loss curve saved to: {cfg.loss_plot_path}")
    print(f"Prediction graphs saved to: {summary['prediction_graphical_dir']}")
    print("Training summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
