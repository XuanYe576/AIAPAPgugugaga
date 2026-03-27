# Trainzip

This folder is a standalone training bundle meant to be copied by itself onto another machine.

Included:

- `main.py`
- `Model/R5O.py`
- `Model/R5.5B-s.py`
- `Model/Pinn/R5PINN_perF.py`
- `utils/amp.py`
- `utils/adamw.py`
- `metrics/plotting.py`
- `metrics/prediction_graphs.py`
- `metrics/visualize_results.py`
- `Data/60k61db.csv`

## Notes

- This bundle is for training and plotting from the processed `60k` CSV.
- The PINN copy can run from the CSV alone. If no metadata JSON is present, it infers `61` points over `1 GHz` to `6 GHz`.
- Prediction graphs saved after training already compare the model output against the original target S11 curve from the CSV.

## Setup

```bash
cd ~/mainPAP/Trainzip
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib torch
```

## Train With The Launcher

Default `R5O`:

```bash
cd ~/mainPAP/Trainzip
python3 main.py \
  --csv-path Data/60k61db.csv \
  --output-mode mag_only \
  --seq-len 61 \
  --epochs 80 \
  --device auto
```

`R5.5B-s`:

```bash
cd ~/mainPAP/Trainzip
python3 main.py \
  --model r55bs \
  --csv-path Data/60k61db.csv \
  --output-mode mag_only \
  --epochs 50 \
  --device auto
```

`PINN`:

```bash
cd ~/mainPAP/Trainzip
python3 main.py --usepinn \
  --processed-csv-path Data/60k61db.csv \
  --epochs 80 \
  --batch-size 256 \
  --device auto
```

## Plot Saved Results

```bash
cd ~/mainPAP/Trainzip
python3 metrics/visualize_results.py
```

Outputs are written under `results/`.
