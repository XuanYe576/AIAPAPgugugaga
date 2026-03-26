# AI-Assisted Patch Antenna Design Using a Hybrid Encoder-Decoder Architecture

Assuming you mean the physics-informed PINN path, the actual backend is [`mainPAP/Model/Pinn/R5PINN_perF.py`](/Users/zhuowenfeng6626/Downloads/ProjectAntennaPatch/mainPAP/Model/Pinn/R5PINN_perF.py#L1), selected by [`mainPAP/Model/main.py`](/Users/zhuowenfeng6626/Downloads/ProjectAntennaPatch/mainPAP/Model/main.py#L34). The flow is:

- Preprocess raw geometry catalogs plus per-antenna S11 CSVs into one matrix of `[100 geometry bits + 61 dB points]` in [`mainPAP/Model/Pinn/R5PINN_perF.py`](/Users/zhuowenfeng6626/Downloads/ProjectAntennaPatch/mainPAP/Model/Pinn/R5PINN_perF.py#L350).
- Expand each antenna into 61 point samples, so training becomes `(geometry, normalized frequency) -> dB(S11)` in [`mainPAP/Model/Pinn/R5PINN_perF.py`](/Users/zhuowenfeng6626/Downloads/ProjectAntennaPatch/mainPAP/Model/Pinn/R5PINN_perF.py#L418) and [`mainPAP/Model/Pinn/R5PINN_perF.py`](/Users/zhuowenfeng6626/Downloads/ProjectAntennaPatch/mainPAP/Model/Pinn/R5PINN_perF.py#L468).
- Encode the 10x10 patch as a graph with node occupancy plus coordinate features, then fuse that geometry embedding with sinusoidal/polynomial frequency features before predicting one scalar dB value in [`mainPAP/Model/Pinn/R5PINN_perF.py`](/Users/zhuowenfeng6626/Downloads/ProjectAntennaPatch/mainPAP/Model/Pinn/R5PINN_perF.py#L490), [`mainPAP/Model/Pinn/R5PINN_perF.py`](/Users/zhuowenfeng6626/Downloads/ProjectAntennaPatch/mainPAP/Model/Pinn/R5PINN_perF.py#L621), and [`mainPAP/Model/Pinn/R5PINN_perF.py`](/Users/zhuowenfeng6626/Downloads/ProjectAntennaPatch/mainPAP/Model/Pinn/R5PINN_perF.py#L690).

The “physics-informed” part is mostly in the loss, not in the forward model:

- Resonance weighting upweights samples near the deepest S11 notch: [`mainPAP/Model/Pinn/R5PINN_perF.py`](/Users/zhuowenfeng6626/Downloads/ProjectAntennaPatch/mainPAP/Model/Pinn/R5PINN_perF.py#L463).
- Passivity penalty punishes predicted positive dB values with `relu(pred_db)^2`, which encodes that passive antennas should have S11 in dB at or below 0: [`mainPAP/Model/Pinn/R5PINN_perF.py`](/Users/zhuowenfeng6626/Downloads/ProjectAntennaPatch/mainPAP/Model/Pinn/R5PINN_perF.py#L713).
- Total loss is weighted SmoothL1 data fit plus the passivity term: [`mainPAP/Model/Pinn/R5PINN_perF.py`](/Users/zhuowenfeng6626/Downloads/ProjectAntennaPatch/mainPAP/Model/Pinn/R5PINN_perF.py#L725), used in training here: [`mainPAP/Model/Pinn/R5PINN_perF.py`](/Users/zhuowenfeng6626/Downloads/ProjectAntennaPatch/mainPAP/Model/Pinn/R5PINN_perF.py#L916).

Important nuance: this is not a classical PINN with Maxwell/PDE residuals or boundary-condition losses. There is no autograd-based physics residual in this file. It is better described as a physics-informed surrogate with soft physical constraints.

## Repository layout

- `Model/`: My edits
- `Data/`: processed data to run and train.
- `results/`: training outputs and saved analysis artifacts.
- `introduction_and_pdfs/`: specifications, presentation material, reference notes, and images.
- `utils/`: utility scripts.

## Current model families

- `GNN/GCN + FEDformer-style decoder`: graph encoder over the `10 x 10` layout with spectral and Transformer-style decoding.
- `CNN + FEDformer-style decoder`: image-style geometry encoder with spectral sequence decoding.
- `Tiny CNN regressor`: compact direct baseline without the FEDformer-style decoder.
- `LightGBM baseline`: one regressor per frequency point as a non-neural baseline.

## Setup

```bash
cd mainPAP
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib torch
```

## Training Entry Point

Use the surface launcher:

```bash
python3 main.py ...
```

`main.py` forwards to:

- `Model/patch_antenna_ai_r5.py` for the non-PINN path
- `Model/Pinn/R5PINN_perF.py` for the PINN path when `--usepinn` is present

## Data Preparation

For the uploaded `30000`-antenna raw dataset, preprocess once first:

```bash
cd mainPAP
python3 main.py --usepinn preprocess --overwrite-processed
```

-->

- `Data/processed/Full_data_61dB.csv`

## Common Commands

Train without PINN:

```bash
cd mainPAP
python3 main.py \
  --csv-path Data/processed/Full_30000Data_61dB.csv \
  --output-mode mag_only \
  --epochs 80 \
  --batch-size 32 \
  --device auto
```

Train with PINN:

```bash
cd mainPAP
python3 main.py --usepinn \
  --epochs 80 \
  --batch-size 256 \
  --device auto
```

Use Adagrad:

```bash
python3 main.py \
  --csv-path Data/processed/Full_30000Data_61dB.csv \
  --output-mode mag_only \
  --optimizer adagrad
```

Use Adagrad with PINN:

```bash
python3 main.py --usepinn \
  --optimizer adagrad
```

Print one training log every 10 batches:

```bash
python3 main.py \
  --csv-path Data/processed/Full_30000Data_61dB.csv \
  --output-mode mag_only \
  --log-every-batches 10
```

Evaluate a saved checkpoint on validation:

```bash
python3 main.py \
  --csv-path Data/processed/Full_30000Data_61dB.csv \
  --output-mode mag_only \
  --eval-split val
```

Evaluate a saved checkpoint on test:

```bash
python3 main.py --usepinn --eval-split test
```

Rebuild processed data and then train with PINN:

```bash
python3 main.py --usepinn \
  --overwrite-processed \
  --epochs 80 \
  --batch-size 256
```

## Logging And Checkpoints

Both backends now support:

- `--optimizer adamw|adagrad`
- `--log-every-batches N`
- `--eval-split val|test`
- checkpoint saving
- history CSV logging
- JSON summary output

Output locations:

- non-PINN: `results/patch_antenna_ai_r5/`
- PINN: `results/R5PINN_perF/`

Checkpoint behavior:

- the best model is chosen by validation loss
- a checkpoint file and model weights are saved when validation improves
- validation or test can be rerun later from the saved checkpoint

## Metrics Visualization

Use the metrics script to turn saved `history.csv` and `summary.json` files into figures:

```bash
cd mainPAP
python3 metrics/visualize_results.py
```

Optional:

- `--run-dir results/patch_antenna_ai_r5`
- `--run-dir results/R5PINN_perF`
- `--output-dir metrics/figures`

The script creates:

- one dashboard PNG per run
- one comparison PNG when multiple runs are available

## GitHub setup

GitHub usually recommends every repository include a `README`, `LICENSE`, and `.gitignore`. This repo now has a readable project README. A `LICENSE` can still be added later if you want the repository to be publicly reusable under explicit terms.

### Create a new repository on the command line

```bash
echo "# -AI-Assisted-Patch-Antenna-Design-Using-a-Hybrid-Encoder-Decoder-Architecture" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:XuanYe576/-AI-Assisted-Patch-Antenna-Design-Using-a-Hybrid-Encoder-Decoder-Architecture.git
git push -u origin main
```

### Push an existing repository from the command line

```bash
git remote add origin git@github.com:XuanYe576/-AI-Assisted-Patch-Antenna-Design-Using-a-Hybrid-Encoder-Decoder-Architecture.git
git branch -M main
git push -u origin main
```

## Citations

Main method families referenced by the codebase:

1. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. "Attention Is All You Need." arXiv, 2017. https://arxiv.org/abs/1706.03762
2. Thomas N. Kipf and Max Welling. "Semi-Supervised Classification with Graph Convolutional Networks." OpenReview, 2017. https://openreview.net/forum?id=SJU4ayYgl
3. Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting." Proceedings of Machine Learning Research, 2022. https://proceedings.mlr.press/v162/zhou22g.html
4. Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS, 2017. https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

## Current baseline note

`Model/patch_antenna_ai_r5.py` is the current cleaned baseline in this repo snapshot. It trains a graph-based encoder plus FEDformer-style decoder for `61` S11 samples and supports `mag_only` or `complex_61` output formatting depending on the CSV layout.




"""
Summary of changes for the PINN NaN loss:
Sanitize curves_db in ProcessedCurveDataset
After loading the CSV, non-finite values in the dB columns are replaced and then clamped:
torch.where(torch.isfinite(...), curves_db, -80.0) so nan/-inf become -80 dB
torch.clamp(..., min=-80, max=0) so all values stay in [-80, 0] and +inf becomes 0
That way targets and weights are always finite even if the source data had log(0) or bad values.
Safe resonance_weights_db
The divisor is now max(sigma_db, 1e-6) so the weight computation never divides by zero.
Defensive pointwise_physics_loss
The data term is wrapped in torch.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0) so any remaining non-finite value in the loss is replaced and training doesn’t get a NaN loss.
Re-run PINN training (e.g. cd /home/ubuntu/A && ./run_pinn_then_nopinn_tmux.sh, then tmux attach -t antenna). The data loss and total loss should stay finite. If the CSV was already written with -inf/nan, the in-memory sanitization will fix it; if you prefer a clean file on disk, you can re-run preprocessing after these code changes so the processed CSV itself has no non-finite values.
"""
