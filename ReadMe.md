# AI-Assisted Patch Antenna Design Using a Hybrid Encoder-Decoder Architecture

This project explores data-driven patch antenna design with hybrid encoder-decoder models that predict S11 responses from `10 x 10` binary patch geometries.

The main runnable entry point in the current repository snapshot is `main.py`. Older notebooks, archived experiments, and supporting notes are kept under `old/`.

## Repository layout

- `Model/`: current training code and model implementations.
- `Data/`: intended location for active datasets such as `Full_1000Data.csv`.
- `patch_antennas_updated.csv`: currently available CSV in this snapshot.
- `results/`: training outputs and saved analysis artifacts.
- `old/`: archived notebooks, experiment notes, and previous documentation.
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
- `Model/PINN/R5PINN_perF.py` for the PINN path when `--usepinn` is present

## Data Preparation

For the uploaded `30000`-antenna raw dataset, preprocess once first:

```bash
cd mainPAP
python3 main.py --usepinn preprocess --overwrite-processed
```

This builds:

- `Data/processed/Full_30000Data_61dB.csv`
- `Data/processed/Full_30000Data_61dB.meta.json`

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
