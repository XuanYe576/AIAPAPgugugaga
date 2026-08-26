# R6P_v3

`R6P_v3` is a two-stage surrogate for patch-antenna S11 prediction from binary metallization geometry. It supersedes `R6P_v2`: that checkpoint and code are left in place under `R6P_v2/` for reference, but its architecture cannot load `R6P_v3` weights or vice versa.

## What changed vs. R6P_v2

`R6P_v2` supervised only the single dominant resonance per antenna (one row per `antenna_id` in `resonance_labels_60k.csv`), routed dip presence through a global softmax over all frequency bins (so multiple simultaneous resonances competed for one shared probability budget), and gated Stage-2 fusion with a fixed `sigmoid(logits) > 0.5` threshold. In practice, on a full 40-epoch real-data run, that threshold never fired on any test sample — Stage 2 contributed nothing and `final_curve == coarse_line` exactly.

`R6P_v3` fixes this at the source:

- **Multi-resonance labels from the curve itself.** `data.py` scans every real S11 curve for local minima whose prominence clears `min_dip_depth_db` (default 2.0 dB), instead of relying on the single-label CSV. Real antennas commonly have 3-5+ genuine resonances; all of them are now supervised.
- **Residual U-Net dip head with skip connections**, replacing the global self-attention block in `dip_experts.py`. Dip presence is an independent per-bin `sigmoid`, not a `softmax` over the sequence, so multiple confident dips no longer dilute each other.
- **Soft IoU loss** (`losses.py`) over the dense multi-dip mask, combined with BCE — rewards getting the whole set of dip bins right, not just per-bin averages that a handful of positives can barely move.
- **Learnable presence bias.** The dip-presence bias is an `nn.Parameter` (init -2.0), not a fixed constant, so training can correct a miscalibrated threshold instead of getting stuck under it.
- **`DipCountAdapter`** (`count_adapter.py`): a small head reading the pooled geometry embedding `z_g` that classifies how many resonances an antenna has (`0..max_dip_count`), supervised by cross-entropy against the count of curve-detected dips.
- **Top-k fusion gating.** At hard inference, `fusion.py` no longer thresholds at a fixed 0.5 — it selects the top-k highest-logit bins, where k is the count adapter's prediction. Rank-based selection stays correct even if the whole logit distribution is shifted, which is exactly the failure mode that made `R6P_v2`'s fusion never fire.

## Assumptions

- Framework: PyTorch, with optional PyTorch Geometric `GraphConv`. A plain PyTorch fallback graph convolution is included so the model still runs without PyG.
- Geometry input: `B x H x W`, binary grid, where `1` means metallization.
- Output: `B x F` S11 magnitude in dB. Default `F=61` for the processed `1.0..6.0 GHz` data already in this repo.
- Real label source: none — dip targets are derived directly from the processed S11 curves (see `_find_curve_dips` / `_make_multi_dip_targets` in `data.py`). `resonance_labels_60k.csv` is no longer read by the training pipeline.

## Real Data Contract

- Processed curves + meta: `Trainzip/Data/60k61db.csv`, `Trainzip/Data/60k61db.meta.json` (`freq_axis_ghz`, `matched_antenna_ids`).
- Format: each row is `100 geometry bits + 61 S11(dB) samples`.

`data.py` derives, per sample, from the dense curve:

- `dip_mask[F]`: union of Gaussian-spread bumps around every detected dip bin.
- `dip_anchor_mask[F]`: one bin per detected dip (not just one per sample).
- `dip_offset_ghz[F]`: 0 at every anchor bin — curve-derived dips sit exactly on a bin, no sub-bin frequency is available.
- `dip_depth_db[F]`: positive dip magnitude at each anchor bin.
- `dip_supervision_mask[F]`: all-ones; class imbalance is handled by loss weighting, not by masking out samples.
- `dip_count`: scalar, the number of detected dips — target for `DipCountAdapter`.

If the processed CSV or meta are missing, the loader falls back to a synthetic dataset (with a real dominant + secondary resonance baked in) so the full pipeline still runs.

## Architecture

### Stage 1

- Grid to graph: node per cell, 4- or 8-neighbor edges, node feature dimension = 1.
- Encoder: `GraphConv(1 -> 32) + ReLU`, `GraphConv(32 -> 64) + ReLU`, global mean over nodes, MLP to `z_g in R^d_model`.
- Decoder: token expansion from `z_g` to `B x F x d_model`, sinusoidal positional encoding, spectral FFT mixing blocks, local-window Transformer blocks, linear head to coarse `B x F` S11 line.

### Physics Adapter

Computes approximate cavity-model features and injects them as soft conditioning into the dip experts (never clamps predictions to the closed-form values):

- Effective permittivity: `eps_eff = (er + 1)/2 + ((er - 1)/2) * (1 + 12 h / W)^(-1/2)`
- Fringing extension: `delta_L = 0.412 h * ((eps_eff + 0.3)(W/h + 0.264)) / ((eps_eff - 0.258)(W/h + 0.8))`
- Cavity-mode resonance estimate: `f_mn = c / (2 sqrt(eps_eff)) * sqrt((m / L_eff)^2 + (n / W_eff)^2)`, `L_eff = L + 2 delta_L`

### Stage 2

- Input: Stage-1 decoder features, coarse S11 line, physics adapter conditioning.
- `DipResUNet`: 1D residual U-Net (encoder/decoder with skip connections) over the frequency axis, replacing the old global self-attention block.
- MoE experts on top of the U-Net features output, per bin: dip-presence logit, dip-center offset (GHz), dip depth (positive dB).
- Dip curve synthesis: each bin emits a Gaussian notch weighted by its own independent `sigmoid` confidence (not a shared softmax); these sum into a dip-only curve.
- `DipCountAdapter`: reads `z_g` directly, predicts how many resonances the antenna has.
- Fusion: training uses a soft sigmoid gate; hard inference selects the top-k highest-logit bins, k = predicted count.

## Shapes

- Geometry input: `B x H x W`
- Graph encoder pooled output: `B x d_model`
- Stage-1 decoder features: `B x F x d_model`
- Coarse line: `B x F`
- Physics conditioning: `B x F x conditioning_dim`
- Dip logits, offsets, depths: `B x F`
- Count logits: `B x (max_dip_count + 1)`
- Final curve: `B x F`

## Files

- `config.py`, `data.py`, `encoder.py`, `decoder.py`, `physics_adapter.py`, `dip_experts.py`, `count_adapter.py`, `fusion.py`, `model.py`, `losses.py`, `train.py`, `ablations.py`

## Run

Train the full model on the real dataset:

```bash
python3 Model/Pinn/R6P_v3/train.py --mode full
```

Train Stage-1 only:

```bash
python3 Model/Pinn/R6P_v3/train.py --mode stage1
```

Run ablations A-D:

```bash
python3 Model/Pinn/R6P_v3/ablations.py
```

Run synthetic only:

```bash
python3 Model/Pinn/R6P_v3/train.py --use-synthetic-only --synthetic-size 256
```

## Config Knobs

Important parameters live in `config.py`:

- `data.grid_height`, `data.grid_width`, `data.seq_len`
- `data.min_dip_depth_db`, `data.dip_min_separation_bins`
- `physics.relative_permittivity`, `physics.substrate_height_m`
- `model.d_model`, `model.ffn_hidden`, `model.attention_window`
- `model.num_experts`, `model.dip_sigma_bins`, `model.dip_unet_depth`
- `model.max_dip_count`, `model.count_adapter_hidden`
- `loss.line_weight`, `loss.mask_weight`, `loss.iou_weight`, `loss.dip_weight`, `loss.count_weight`, `loss.final_weight`
- `train.batch_size`, `train.epochs`, `train.lr`
