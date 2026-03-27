# Trainzip — Patch Antenna Training Pipeline

Training bundle for the three real backends in this repo:

- `R5O`
- `R5.5B-s`
- `PINN`

This folder is meant to run by itself on a CUDA machine such as GH200.

## Quick Start

```bash
cd ~/mainPAP/Trainzip
./run_streamline.sh
```

This runs all three configurations sequentially on CUDA, one at a time:

| Step | Model   | Output Dir           |
|------|---------|----------------------|
| 1/3  | R5O     | `weights/R5O_no_PINN` |
| 2/3  | R5.5B-s | `weights/R55BS_no_PINN` |
| 3/3  | PINN    | `weights/PINN` |

Logs are saved to `weights/logs/`.

Important note:

- This repo has one PINN backend.
- There is not a separate technical path called “R5O with PINN” or “R5.5B-s with PINN”.
- If you want a truthful streamlined runner, the real sequence is `R5O`, `R5.5B-s`, and `PINN`.

## Setup

```bash
cd ~/mainPAP/Trainzip
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib torch
chmod +x run_streamline.sh
```

## Files Included

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
- `Data/60k61db.meta.json`
- `run_streamline.sh`

## Bugs Fixed And CUDA Stabilization

### 1. NumPy Scalar Bug In `build_geometry_graph` (`R5O.py`)

The original R5O graph builder used `(N, 1)` slices:

```python
x01 = coords[:, 1:2] / max(1, width - 1)
y01 = coords[:, 0:1] / max(1, height - 1)
```

Then later it did:

```python
dx = float(x01[src] - x01[dst])
dy = float(y01[src] - y01[dst])
```

That is fragile because `x01[src]` and `y01[src]` are 1-element arrays, not scalars.

The bundle now uses 1D arrays:

```python
x01 = coords[:, 1] / max(1, width - 1)
y01 = coords[:, 0] / max(1, height - 1)
node_static = np.stack([x01, y01, x_center, y_center, radius], axis=1)
```

### 2. Lower Default LR In `R5O.py`

`R5O.py` now defaults to:

```python
lr: float = 5e-4
```

instead of `2e-3`, which is safer for the current loss stack and large-curve training path.

### 3. Non-Finite Batch Guard In `R5O.py`

If a batch produces `NaN` or `Inf` total loss, the training loop now skips the backward and optimizer step for that batch, while still advancing the scheduler:

```python
if not torch.isfinite(losses.total):
    scheduler.step()
    continue
```

That prevents one corrupt batch from poisoning the weights.

### 4. AMP And Gradient Clipping Added To `R5.5B-s.py`

`R5.5B-s.py` now uses the shared AMP helper and gradient clipping:

```python
with autocast_context(device, cfg.use_amp):
    outputs = model(xb, freq_axis_hz)
    loss = complex_mse(outputs["gamma"], yb)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
scaler.step(optimizer)
```

### 5. PINN Passive Physics Term Forced To Float32

The PINN passive term contains `10^(pred_db / 20)`. That exponential is now computed outside mixed precision:

```python
with torch.amp.autocast("cuda", enabled=False):
    pred_f32 = pred_db.float()
    s11_lin = torch.pow(pred_f32.new_tensor(10.0), pred_f32 / 20.0)
```

This reduces overflow risk in the physics term.

### 6. Shared AMP Helpers For CUDA

`utils/amp.py` now centralizes AMP behavior for the bundle.

For GH200-style CUDA usage, it uses CUDA autocast with `bfloat16` and a null scaler fallback:

```python
return torch.amp.autocast("cuda", dtype=torch.bfloat16)
```

### 7. PINN Metadata Included

The bundle includes:

- `Data/60k61db.csv`
- `Data/60k61db.meta.json`

The meta JSON is intentionally minimal. The PINN loader can also infer the frequency axis from the CSV shape if needed.

### 8. Live Logging For tmux Or SSH

`run_streamline.sh` uses unbuffered Python output so logs appear live:

```bash
export PYTHONUNBUFFERED=1
python3 -u main.py ...
```

## Plotting And Comparison

After each run, results are written under `weights/...`.

You can generate dashboards and cross-run comparisons with:

```bash
cd ~/mainPAP/Trainzip
python3 metrics/visualize_results.py --run-dir weights/R5O_no_PINN
python3 metrics/visualize_results.py --run-dir weights/R55BS_no_PINN
python3 metrics/visualize_results.py --run-dir weights/PINN
```

The saved prediction graphs compare:

- model prediction
- target S11 curve from the bundled `60k61db.csv`

So they already give you the direct “predicted vs original patch response” comparison after training.

## Manual Commands

### R5O

```bash
cd ~/mainPAP/Trainzip
python3 main.py \
  --csv-path Data/60k61db.csv \
  --output-mode mag_only \
  --seq-len 61 \
  --epochs 80 \
  --device cuda \
  --output-dir weights/R5O_no_PINN
```

### R5.5B-s

```bash
cd ~/mainPAP/Trainzip
python3 main.py \
  --model r55bs \
  --csv-path Data/60k61db.csv \
  --output-mode mag_only \
  --epochs 50 \
  --device cuda \
  --output-dir weights/R55BS_no_PINN
```

### PINN

```bash
cd ~/mainPAP/Trainzip
python3 main.py --usepinn \
  --processed-csv-path Data/60k61db.csv \
  --processed-meta-path Data/60k61db.meta.json \
  --epochs 80 \
  --batch-size 256 \
  --device cuda \
  --results-dir weights/PINN
```
