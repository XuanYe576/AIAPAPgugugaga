"""Diagnostic: does the presence head know WHERE dips are, or just HOW MANY?

Plots sigmoid(dip_presence_logits) across all 61 bins against the true dip
bins for a handful of labeled real samples.

- Real (if diffuse, sub-0.5) bumps near the true bins -> calibration/width
  problem; the Gaussian-target + pos_weight fixes should help.
- A flat/unrelated map -> the bug is upstream of fusion entirely (label-bin
  alignment in data.py, or the head is geometry-blind); rebuilding fusion
  is premature.

Run from the repo root, e.g.:
    python3 mainPAP/Model/Pinn/R6P_v3/diagnose_presence.py \\
        --checkpoint /path/to/r6p_v3_best.pt --num-samples 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[4]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from mainPAP.Model.Pinn.R6P_v3.config import ExperimentConfig
    from mainPAP.Model.Pinn.R6P_v3.data import build_dataset, split_indices
    from mainPAP.Model.Pinn.R6P_v3.model import R6P_v3Model
else:
    from .config import ExperimentConfig
    from .data import build_dataset, split_indices
    from .model import R6P_v3Model


def plot_presence_vs_truth(
    model: R6P_v3Model,
    batch: dict[str, torch.Tensor],
    out_dir: Path,
    max_samples: int,
) -> None:
    model.eval()
    with torch.no_grad():
        outputs = model(batch["geometry"], stage1_only=False, use_adapter=True, hard_inference=True)
    probs = torch.sigmoid(outputs.dip_presence_logits)
    labeled = (batch["label_available"] > 0.5).nonzero(as_tuple=True)[0]

    shown = 0
    for idx in labeled.tolist():
        if shown >= max_samples:
            break
        true_bins = (batch["dip_anchor_mask"][idx] > 0.5).nonzero(as_tuple=True)[0].tolist()
        p = probs[idx]
        max_bin = int(p.argmax().item())
        # 0 = the true bin IS the top-scoring bin; larger = further down the ranking.
        rank_of_true = [int((p > p[b]).sum().item()) for b in true_bins]
        prob_at_true = [round(float(p[b].item()), 3) for b in true_bins]
        print(
            f"sample {idx}: true_bins={true_bins} prob_at_true={prob_at_true} "
            f"rank_of_true(0=top)={rank_of_true} max_prob={p.max().item():.3f} at bin {max_bin}"
        )

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(p.shape[0]), p.numpy(), color="steelblue", width=0.9)
        for b in true_bins:
            ax.axvline(b, color="red", linestyle="--", linewidth=1.5)
        ax.axhline(0.5, color="black", linestyle=":", linewidth=1.0)
        ax.set_title(f"sample {idx}: sigmoid(presence logit) per bin (red = true dip bin)")
        ax.set_xlabel("frequency bin")
        ax.set_ylabel("presence probability")
        ax.set_ylim(0.0, 1.0)
        fig.tight_layout()
        fig.savefig(out_dir / f"presence_map_sample_{idx}.png", dpi=120)
        plt.close(fig)
        shown += 1

    if shown == 0:
        print("No labeled samples in this batch — nothing to plot.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot presence-head score maps against true dip bins.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed-csv", type=Path, default=None)
    parser.add_argument("--processed-meta", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=Path("presence_diagnostic"))
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.processed_csv is not None:
        cfg.data.processed_csv_path = args.processed_csv
    if args.processed_meta is not None:
        cfg.data.processed_meta_path = args.processed_meta

    dataset, freq_axis_ghz, source_name, _ = build_dataset(cfg)
    print(f"data source: {source_name}, {len(dataset)} samples")
    _, _, test_idx = split_indices(len(dataset), cfg.train.train_ratio, cfg.train.val_ratio, cfg.train.random_seed)
    loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    model = R6P_v3Model(cfg, freq_axis_ghz=freq_axis_ghz.float())
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    batch = next(iter(loader))
    plot_presence_vs_truth(model, batch, args.out_dir, max_samples=args.num_samples)
    print(f"plots written to {args.out_dir}")


if __name__ == "__main__":
    main()
