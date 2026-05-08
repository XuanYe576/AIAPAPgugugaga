# Trainzip

Self-contained training bundle for the R6 antenna models.

## Data

The datasets are kept separate:

- `Data/feedpatch_1093_501.csv`: 1093 feedpatch samples, 501 S11 points
- `Data/60k61db.csv`: 60000 geometry samples, 61 S11 points
- `Data/train_feedpatch_1093.txt`, `Data/val_feedpatch_1093.txt`, `Data/resonance_labels_feedpatch_1093.csv`
- `Data/train_60k.txt`, `Data/val_60k.txt`, `Data/resonance_labels_60k.csv`

## Intended R6 Flow

The training order is:

1. Train on 1093/501 first to learn feed and notch priors.
2. Continue on 60k/61 to learn the main geometry-to-curve mapping.
3. Export or infer at 501 points.

`R6O.py` supports this directly with configurable curve heads:

- `--curve-head-points 501,61`
- `--active-curve-points 501` for the 1093 stage
- `--active-curve-points 61` for the 60k stage
- `--export-head-points 501` for final 501-point output
- `--init-checkpoint-path ...` to continue from the previous stage

`R6P.py` is frequency-axis based, so it does not need separate linear heads. It uses `--init-checkpoint-path ...` for the same staged training flow and `--export-curve-points 501` for 501-point prediction export.

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
