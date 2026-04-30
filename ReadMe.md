# AI-Assisted Patch Antenna Design Using a Hybrid Encoder-Decoder Architecture

Current repo layout for the R5O, R5.5B-s, PINN, and LightGBM training pipelines.

- `Model/R5O.py`: architecture-only non-PINN baseline
- `Model/R5.5B-s.py`: older notebook-style neural baseline
- `Model/Pinn/R5PINN_perF.py`: per-frequency PINN path with physics-guided loss

## Antenna Setup

- Each antenna geometry is represented as a `10 x 10` binary metal grid, so each sample has `100` geometry bits.
- The current preprocessing treats one cell as `15 mm x 15 mm`, so the represented patch canvas is `150 mm x 150 mm`.
- The processed geometry order follows the historical `data` convention: flatten each `10 x 10` patch from bottom to top, left to right.
- Each antenna has one S11 response file named `patch_<id>_s11_plot.csv`.
- The current processed magnitude dataset stores `61` S11 values in dB per antenna. The exact frequency axis is saved in the matching metadata JSON.
- The PINN resonance helper uses a simple substrate approximation with relative permittivity `er = 4.4` and substrate thickness `h = 1.57 mm`.

## What The Data Is Based On

- Raw geometry catalogs live under `Data/NotprocessedData/.../patch_antennas_updated*.csv`.
- Raw S11 curves live under `Data/NotprocessedData/**/patch_*_s11_plot.csv`.
- The default `60k` build uses `Data/NotprocessedData/60000 Patch Antenna File/patch_antennas_updated5b.csv`.

## How Data Is Processed

The main preprocessing utility is `utils/dataP.py`.

It performs the following steps:

1. Parse a raw geometry catalog and read each antenna as a `10 x 10` binary grid.
2. Flip the grid vertically so the flattened output matches the historical `Ori_30k.csv` orientation.
3. Scan `Data/NotprocessedData` for all `patch_<id>_s11_plot.csv` files.
4. Match geometry IDs to curve IDs.
5. Write one processed row per matched antenna as:
   `[100 geometry bits] + [61 S11 dB samples]`
6. Write a companion metadata JSON with the geometry catalog path, curve root, matched antenna IDs, duplicate IDs, missing IDs, grid size, sequence length, and frequency axis.

The current default processed files are:

- `Data/processed/Full_60000Data_61dB.csv`
- `Data/processed/Full_60000Data_61dB.meta.json`

## Models

| Model | Script | Input and output | Main idea |
| --- | --- | --- | --- |
| R5O | `Model/R5O.py` | Processed CSV to full curve | GATv2-style graph encoder plus geometry-conditioned decoder trained as the non-PINN architecture baseline |
| R5.5B-s | `Model/R5.5B-s.py` | Processed CSV to full curve | Older notebook-style GraphConv plus spectral and Transformer decoder baseline |
| R5 PINN | `Model/Pinn/R5PINN_perF.py` | Processed CSV to one dB value at one frequency | Per-frequency graph model with soft physics constraints |
| R6P Pole-Residue PINN | `Model/Pinn/R6P.py` | Processed CSV to full curve | Shared encoder + pole-residue rational decoder + notch branch + auxiliary transmission-line regularization |
| R6O Multi-Task Notch | `Model/Pinn/R6O.py` | Processed CSV to full curve + notch features | Shared encoder with two heads: curve reconstruction MAE + notch feature regression loss |
| LightGBM baseline | `Model/patch_antenna_ai_colab_GPT_R4_LightGBM_with_validation_excel.ipynb` | Processed CSV to 61 separate regressors | Fast tabular baseline with Excel export |

## Environment Setup

```bash
cd mainPAP
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib torch
```

Notes:

- `numpy` is enough for preprocessing.
- `torch` is required for training and evaluation.
- `matplotlib` is required for plotting utilities.
- `pandas` is used by `R5.5B-s.py`.

## How To Run The Code

### Preprocess The 60k Raw Dataset

```bash
cd mainPAP
python3 utils/dataP.py --overwrite
```

Equivalent PINN entrypoint:

```bash
cd mainPAP
python3 Model/Pinn/R5PINN_perF.py preprocess --overwrite-processed
```

### Inspect The Processed PINN Dataset

```bash
cd mainPAP
python3 Model/Pinn/R5PINN_perF.py inspect
```

### Train The PINN

Direct script:

```bash
cd mainPAP
python3 Model/Pinn/R5PINN_perF.py train \
  --epochs 80 \
  --batch-size 256 \
  --device auto
```

Surface launcher:

```bash
cd mainPAP
python3 Model/main.py --usepinn \
  --epochs 80 \
  --batch-size 256 \
  --device auto
```

Evaluate a saved PINN checkpoint:

```bash
cd mainPAP
python3 Model/Pinn/R5PINN_perF.py --eval-split test
```

### Train R5O On The Magnitude-Only 60k Dataset

```bash
cd mainPAP
python3 Model/R5O.py \
  --csv-path Data/processed/Full_60000Data_61dB.csv \
  --output-mode mag_only \
  --seq-len 61 \
  --epochs 80 \
  --batch-size 32 \
  --device auto
```

Evaluate a saved R5O checkpoint:

```bash
cd mainPAP
python3 Model/R5O.py \
  --csv-path Data/processed/Full_60000Data_61dB.csv \
  --output-mode mag_only \
  --seq-len 61 \
  --eval-split test
```

### Train R5.5B-s On The Magnitude-Only 60k Dataset

```bash
cd mainPAP
python3 Model/R5.5B-s.py \
  --csv-path Data/processed/Full_60000Data_61dB.csv \
  --output-mode mag_only \
  --epochs 50 \
  --batch-size 64 \
  --device auto
```

Evaluate a saved R5.5B-s checkpoint:

```bash
cd mainPAP
python3 Model/R5.5B-s.py \
  --csv-path Data/processed/Full_60000Data_61dB.csv \
  --output-mode mag_only \
  --eval-split test
```

### Launcher Shortcuts

`Model/main.py` now supports all current backends:

- default: `R5O`
- `--model r55bs`: `R5.5B-s`
- `--model pinn` or `--usepinn`: `R5PINN_perF`
- `--model r6p`: `R6P_pole_residue`
- `--model r6o`: `R6O`

Examples:

```bash
cd mainPAP
python3 Model/main.py --epochs 80
python3 Model/main.py --model r55bs --epochs 50
python3 Model/main.py --usepinn --epochs 80 --batch-size 256
python3 Model/main.py --model r6p --epochs 120 --batch-size 128
python3 Model/main.py --model r6o --epochs 120 --batch-size 128
```

## What The PINN Includes

`Model/Pinn/R5PINN_perF.py` is physics-informed in the soft-constraint surrogate sense. It is not a Maxwell-residual PDE PINN.

The current PINN includes these perspectives:

- Geometry perspective: a graph encoder over the `10 x 10` metal grid, with metal and void occupancy plus coordinate features.
- Frequency perspective: normalized frequency is encoded with polynomial and sinusoidal features before fusion with the geometry embedding.
- Data-fit perspective: training predicts one scalar `dB(S11)` value for one frequency point at a time using weighted SmoothL1 loss.
- Notch perspective: samples near the deepest notch receive larger weights during pointwise training, and validation reconstructs full curves to measure notch-frequency and `-10 dB` bandwidth mismatch.
- Resonance perspective: validation also reconstructs full curves and computes a resonance loss from the predicted notch frequency versus a theoretical resonance estimated from extracted patch length.
- Passivity perspective: the loss penalizes `|S11| > 1` in linear magnitude, not just positive dB values.
- Substrate perspective: theoretical resonance uses a simple closed-form patch formula with `er = 4.4` and `h = 1.57 mm`.
- Generalization perspective: train, validation, and test sets are split by inferred geometry families rather than by naive random leakage.

In short, the PINN is best described as:

- `geometry + normalized_frequency -> dB(S11)`
- graph-based
- per-frequency
- physics-informed by passivity and resonance consistency
- not a full electromagnetic field solver

## Results, Utilities, And Metrics

### Main Output Files

`R5PINN_perF.py` writes to `results/R5PINN_perF/`:

- `history.csv`: per-epoch training and validation metrics
- `loss_per_freq.csv`: average pointwise training loss per frequency bin for each epoch
- `summary.json`: best validation metrics and final test metrics
- `config.json`: saved run configuration
- `weight/graphical/*.png`: predicted versus target S11 curves for selected samples

`R5O.py` writes to `results/patch_antenna_ai_r5/`:

- `history.csv`
- `summary.json`
- `loss_curve.png`
- `weight/graphical/*.png`
- `weight/excel/val_predictions.xlsx` or `test_predictions.xlsx`

`R5.5B-s.py` writes to `results/R5.5B-s/`:

- `history.csv`
- `summary.json`
- `loss_curve.png`
- `weight/graphical/*.png`

### What The Metrics Mean

Common metrics:

- `train_total`, `val_total`, `test_total`: overall objective used by that script
- `mae` or `db_mae`: average absolute dB error
- `passive`: penalty for violating passivity
- `complex_mse`: complex-domain reconstruction error for complex-output models

PINN-specific metrics:

- `train_data`, `val_data`, `test_data`: weighted SmoothL1 data-fit term
- `val_notch`, `test_notch`: notch-frequency and bandwidth mismatch term from reconstructed validation/test curves
- `val_resonance_loss`, `test_resonance_loss`: mismatch between predicted notch frequency and the simple substrate-based theoretical resonance
- `passive`: passivity penalty used by `R5PINN_perF.py`
- `val_selection_total`, `test_selection_total`: total metric plus `loss_notch * notch + resonance_loss_weight * resonance_loss`

### Utility Scripts

- `utils/dataP.py`: rebuilds processed CSV and metadata from the raw folder tree
- `metrics/prediction_graphs.py`: saves loss curves and prediction-versus-target S11 plots
- `metrics/visualize_results.py`: reads `history.csv` and `summary.json`, then creates per-run dashboards and cross-run comparison plots
- `metrics/plotting.py`: central `matplotlib` loader with the non-interactive `Agg` backend

### Visualize Saved Runs

```bash
cd mainPAP
python3 metrics/visualize_results.py
```

Optional examples:

```bash
python3 metrics/visualize_results.py --run-dir results/R5PINN_perF
python3 metrics/visualize_results.py --run-dir results/patch_antenna_ai_r5
python3 metrics/visualize_results.py --output-dir metrics/figures
```
