# R6P_v4

`R6P_v4` is a two-stage surrogate for patch-antenna S11 prediction from binary metallization geometry. It supersedes `R6P_v3`: that checkpoint and code are left in place under `R6P_v3/` for reference, but its architecture cannot load `R6P_v4` weights or vice versa.

## What changed vs. R6P_v3

`R6P_v3` supervised dip presence as dense per-bin binary classification: 61 independent bins, each asked "is there a dip here", trained with BCE (+ IoU) against a Gaussian-smeared target. Across every pos_weight tried (16, 24, 32, 48, 128, 160), this either collapsed to "never confidently fires anywhere" (low pos_weight) or "confidently fires in the same fixed frequency band regardless of the input geometry" (high pos_weight) — a real, measured trade-off, not a training-time fluke: on `R6P_v3_full_pw128_20260730`, the top presence bins sat at ~1.9–2.8 GHz with ~0.95 confidence for every geometry that had any predicted dip at all, unrelated to where that antenna's true dips actually were.

`R6P_v4` replaces the dense per-bin classification with **DETR-style set prediction and Hungarian matching** (Carion et al., 2020):

- **`dip_queries.py` (`DipQueryDecoder`)**: a small fixed number of learned query slots (`model.num_queries`, default 10) cross-attend over the Stage-1 decoder features (self-attention between queries, then a shared MLP head), each emitting `(existence_logit, location in [0,1], depth_db)`. There is no per-bin grid at all — a slot either represents a real dip or it doesn't.
- **`matching.py` (`hungarian_match`)**: at every forward pass, predicted query slots are bipartite-matched to the variable-size set of true dips (0 to however many a curve actually has) via `scipy.optimize.linear_sum_assignment`, minimizing a cost over existence, location, and depth. Unmatched slots are simply "no object" — there is no fixed background class to tune a threshold against.
- **`losses.py` (`set_prediction_loss`)**: existence is supervised on every slot (matched=1, unmatched=0, with the "no object" class down-weighted by `loss.no_object_weight` so ~10 mostly-empty slots don't dominate); location and depth are regressed only on matched slots. There is no `mask_positive_weight` to calibrate, because there's no dense background/foreground imbalance to begin with.
- **Evaluation (`train.py`)** reports `dip_precision`/`dip_recall`/`dip_f1` (a slot counts as a true positive only if it's both matched to a real dip *and* crosses existence 0.5), plus `dip_location_mae_ghz`, `dip_depth_mae_db`, `dip_count_mae` (predicted count = number of slots with existence > 0.5).
- `count_adapter.py`/`fusion.py`/`dip_experts.py` from v3 are gone entirely — a separate "how many dips" head and a top-k gating hack are redundant once existence itself is a per-slot yes/no.

Everything from v3's encoder-collapse work carries over unchanged: geometry reconstruction + variance/covariance regularization on `z_g`, and the hard collapse-abort in `train.py` (`embedding_variation`/`log_embedding_variation`).

## Assumptions

- Framework: PyTorch, with optional PyTorch Geometric `GraphConv`. A plain PyTorch fallback graph convolution is included so the model still runs without PyG.
- Geometry input: `B x H x W`, binary grid, where `1` means metallization.
- Output: `B x F` S11 magnitude in dB. Default `F=61` for the processed `1.0..6.0 GHz` data already in this repo.
- Real label source: none — dip targets are derived directly from the processed S11 curves (see `_find_curve_dips` / `_make_multi_dip_targets` in `data.py`, unchanged from v3).

## Real Data Contract

- Processed curves + meta: `Trainzip/Data/60k61db.csv`, `Trainzip/Data/60k61db.meta.json` (`freq_axis_ghz`, `matched_antenna_ids`).
- `data.py` is unchanged from v3: it still produces the dense `dip_mask`/`dip_supervision_mask` arrays (used only by the optional `--prune-dip-bins-from-line` Stage-1 pretraining path) alongside `dip_anchor_mask` and `dip_depth_db`, which `losses.extract_ground_truth_dips` turns into the variable-length (location, depth) sets that matching actually trains against.

## Architecture

### Stage 1 (unchanged from v3)

- Grid to graph, `GraphConv` encoder, spatial-pooling + geometry-reconstruction + variance/covariance regularized `z_g`, spectral/windowed-Transformer decoder to a coarse `B x F` S11 line.

### Stage 2 — set prediction

- `DipQueryDecoder`: `num_queries` learned query embeddings, cross-attend over FiLM-conditioned Stage-1 features, self-attend among themselves, then a shared head outputs `(existence_logit, location, depth)` per query.
- Dip curve synthesis: each query emits a Gaussian notch at its predicted location, weighted by its own existence probability (soft during training, hard-thresholded at 0.5 at inference) and depth; these sum into the residual subtracted from the coarse line — this *is* `final_curve`, there is no separate fusion step.
- Training: Hungarian-matched existence (BCE, no-object down-weighted) + location/depth L1 on matched slots only.

## Shapes

- Geometry input: `B x H x W`
- Coarse line / final curve: `B x F`
- Existence logits, location, depth: `B x num_queries`

## Files

`config.py`, `data.py`, `encoder.py`, `decoder.py`, `physics_adapter.py`, `dip_queries.py`, `matching.py`, `model.py`, `losses.py`, `train.py`, `diagnose_geometry.py`, `ablations.py`

## Run

```bash
python3 -m Model.Pinn.R6P_v4.train --mode full
python3 -m Model.Pinn.R6P_v4.train --mode stage1
python3 -m Model.Pinn.R6P_v4.ablations
python3 -m Model.Pinn.R6P_v4.diagnose_geometry --results-dir weights/R6P_v4_<run_name>
```

Stage-1 pretraining + warm start (carried over from v3, still useful — Stage 1's `line` loss can optionally exclude dip-touched bins so it only ever fits the smooth baseline):

```bash
python3 -m Model.Pinn.R6P_v4.train --mode stage1 --prune-dip-bins-from-line \
  --results-dir weights/R6P_v4_stage1_pretrain

python3 -m Model.Pinn.R6P_v4.train --mode full \
  --init-from weights/R6P_v4_stage1_pretrain/r6p_v4_best.pt --freeze-stage1
```

## Config Knobs

- `model.num_queries`, `model.query_attention_heads`, `model.query_ffn_hidden`, `model.query_head_hidden`
- `loss.existence_weight`, `loss.location_weight`, `loss.depth_weight`, `loss.no_object_weight`
- `loss.match_class_weight`, `loss.match_location_weight`, `loss.match_depth_weight` (Hungarian assignment cost, not training loss)
- `loss.geometry_reconstruction_weight`, `loss.embedding_variance_weight`, `loss.embedding_covariance_weight`, `loss.embedding_target_std`
- `data.min_dip_depth_db`, `data.dip_min_separation_bins` (still govern what counts as a real dip when building ground truth)
- `train.embedding_std_abort_threshold`, `train.embedding_min_effective_dims_99` (encoder-collapse hard gate)
