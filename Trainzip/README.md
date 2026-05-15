# Trainzip

Self-contained training bundle for the R6 antenna models.

## Data

The datasets are kept separate:

- `Data/feedpatch_1093_501.csv`: 1093 feedpatch samples, 501 S11 points
- `Data/60k61db.csv`: 60000 geometry samples, 61 S11 points; the loader resamples this to 501 points with shape-preserving PCHIP for R5/R6 training.
- `Data/train_feedpatch_1093.txt`, `Data/val_feedpatch_1093.txt`, `Data/resonance_labels_feedpatch_1093.csv`
- `Data/train_60k.txt`, `Data/val_60k.txt`, `Data/resonance_labels_60k.csv`

## Intended R6 Flow

The training order is:

1. Train on 1093/501 first to learn feed and notch priors.
2. Continue on 60k data interpolated from 61 to 501 points to learn the main geometry-to-curve mapping.
3. Export or infer at 501 points.

`R6O.py` uses one fixed 501-point output head:

- `--output-curve-points 501` keeps the same output head in both stages.
- In the 1093 stage, the 501 output is compared directly with 501-point targets.
- In the 60k stage, the 61-point source data is PCHIP-resampled to 501 points before loss.
- Final prediction/export remains 501 points from the same checkpoint.
- The auxiliary R6O feature is one scalar only: normalized resonance frequency. It is used for feature loss during training; inference output remains the 501-point S11 curve.
- R6O uses only high-confidence local-min resonance labels by default (`--label-min-confidence 0.35`); fallback global-min labels are ignored unless `--use-label-fallbacks` is passed.
- R6O weights the resonance-frequency auxiliary loss at `--loss-feature-weight 5.0` in the streamline run, so the label path is visible in `history.csv` as `*_feature_mae_norm`.
- R6O uses antenna `30002` as a repeated boundary/anchor condition by default (`--boundary-antenna-id 30002`, `--boundary-loss-weight 0.05`). Boundary curve/error/resonance metrics are written to `summary.json`, and a boundary prediction plot is saved under `weight/graphical/boundary/`.
- R6O curve output is constrained to non-positive dB, and `summary.json` records `test_pred_max_db`/`test_pred_min_db` to verify this numerically.
- `--init-checkpoint-path ...` to continue from the previous stage

`R6P.py` is frequency-axis based, so it uses the same PCHIP-resampled 501-point 60k target and `--export-curve-points 501` for prediction export.

`R5O.py` and `R5.5B-s.py` now treat `mag_only` as dynamic-length magnitude data. The `60k` profile also passes `--target-curve-points 501`, so R5 and R6 metrics are comparable on the same 501-point target.

## Run

```bash
cd ~/mainPAP/Trainzip
./run_streamline.sh
```

The script runs:

1. R6O on 1093/501
2. R6O on 60k/61 initialized from step 1, exporting 501
3. R6P on 1093/501
4. R6P on 60k/61 initialized from step 3, exporting 501

Logs are saved under `weights/logs/`.

## R6 vs R5 Evaluation

Train R5 on the same interpolated 501-point 60k target:

```bash
python3 -u main.py --model r5o --dataset-profile 60k --epochs 120 --device cuda --output-dir weights/R5O_60k501
python3 -u main.py --model r55bs --dataset-profile 60k --epochs 120 --device cuda --output-dir weights/R55BS_60k501
```

Evaluate trained checkpoints on the same `60k` profile:

```bash
python3 -u main.py --model r6o --dataset-profile 60k --results-dir weights/R6O_stage2_60k_main_final501 --eval-split test --device cuda
python3 -u main.py --model r6p --dataset-profile 60k --results-dir weights/R6P_stage2_60k_main_final501 --eval-split test --device cuda
python3 -u main.py --model r5o --dataset-profile 60k --output-dir weights/R5O_60k501 --eval-split test --device cuda
python3 -u main.py --model r55bs --dataset-profile 60k --output-dir weights/R55BS_60k501 --eval-split test --device cuda
```
