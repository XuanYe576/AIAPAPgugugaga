# mainPAP

- `model_training/`: notebooks and Python scripts used to train or evaluate models.
- `data/`: CSV datasets, zipped HFSS exports, and dataset-generation files.
- `introduction_and_pdfs/`: design specs, presentation material, notes, PDFs, and reference images.

## Folder layout

```text
mainPAP/
├── model_training/
│   ├── notebooks/
│   └── scripts/
├── data/
└── introduction_and_pdfs/
```

## Data

- `data/Full_1000Data.csv`: main training dataset.
- `data/HFSS CSV Files.zip`: zipped HFSS-exported CSV files.
- `data/AI Patch Antenna csv Files/`: extra CSV datasets plus `creating_csv_of_patch_antennas.m` for dataset generation.

## Introduction and reference files

- `introduction_and_pdfs/AI Antenna_Project Design specification_R1.2.pdf`: project design specification in PDF form.
- `introduction_and_pdfs/AI Antenna_Project Design specification_R1.docx`: editable design specification document.
- `introduction_and_pdfs/AI Patch Antenna Design Results.pptx`: presentation of project results.
- `introduction_and_pdfs/MOO_Student_Guide.pdf`: reference/student guide.
- `introduction_and_pdfs/AI code Explanation.txt`: model/code notes.
- `introduction_and_pdfs/AI code Explanation_R2.txt`: additional code notes for the R2 line.
- `introduction_and_pdfs/AI code Colab_with graph.txt`: notes for the Colab version with plotted loss graphs.
- `introduction_and_pdfs/IMG_4300 (1).jpeg` and `IMG_4302.jpeg`: image references.

## Model training files

Version note:

- The `R...` labels below are local experiment version tags taken from the filenames.
- A true publication date is usually not recorded in these files.
- The only explicit calendar date I found in the model files is `12/19/2025` inside `AI_Patch_Antenna_GPT_R3_4.ipynb`.
- A few exported scripts and notebooks have version-name mismatches between the filename and the internal title; those are called out below.

Published method-family reference:

- `Transformer`: 2017.
- `Graph Convolutional Network (GCN)`: 2017.
- `FEDformer`: 2022.
- `LightGBM`: 2017.
- `CNN`: Proposed CNN encoders/regressors;

### Notebooks

| File | Version tag | Date/publish clue found locally | What it does |
| --- | --- |
| `AI_Patch_Antenna_Gemini_R1_0.ipynb` | `R1.0` | Ours | TensorFlow/Keras model that uses a CNN encoder and a simplified FEDformer-like decoder to predict 61 S11 values from a 10x10 geometry, with flips, rotations, and noise augmentation. |
| `AI_Patch_Antenna_GPT_R3_1.ipynb` | `R3.1` | Ours | PyTorch baseline that loads CSV data, applies input augmentation, and trains a CNN encoder plus simplified FEDformer decoder for 61-point S11 regression. |
| `AI_Patch_Antenna_GPT_R3_2.ipynb` | `R3.2` | Ours | Same CNN plus FEDformer-style pipeline as R3.1, but with augmentation disabled for a cleaner baseline run. |
| `AI_Patch_Antenna_GPT_R3_2_With results.ipynb` | `R3.2` results copy | Ours | Executed/results-preserved copy of the R3.2 CNN plus FEDformer notebook. |
| `AI_Patch_Antenna_GPT_R3_3_Tiny CNN.ipynb` | `R3.3` | Ours | Lightweight pure CNN regressor that predicts the 61 S11 points directly and serves as a compact baseline without the FEDformer decoder. |
| `AI_Patch_Antenna_GPT_R3_4.ipynb` | `R3.4` | Explicit note in file: `12/19/2025` | Improved CNN plus FEDformer notebook with S11 normalization, de-normalized inspection, longer training, learning-rate scheduling, gradient clipping, and augmentation disabled by default. |
| `AI_Patch_Antenna_GPT_R3_4_.ipynb` | `R3.4` alternate copy | No explicit date stated in file | Alternate saved copy of the R3.4 CNN plus FEDformer experiment with outputs retained. |
| `AI_Patch_Antenna_GPT_R3_5_Resonance_weighted_loss.ipynb` | `R3.5` | Ours | CNN plus FEDformer model that changes the loss function to weight resonance/notch regions more heavily during training. |
| `AI_Patch_Antenna_GPT_R3_6_2DPositional.ipynb` | `R3.6` | Ours | CNN plus FEDformer model that augments the geometry input with 2D row and column positional channels. |
| `Patch Antenna_Colab_R2-1.ipynb` | `R2.1` | Ours | Early PyTorch model that uses a GNN-Transformer encoder and a FEDformer-style Fourier decoder for normalized S11 prediction. |
| `R2-3_With Augmentation.ipynb` | `R2.3` | Ours | R2-series GNN-Transformer plus FEDformer model with training-time augmentation applied only to the binary geometry input. |
| `patchAntennaAI_R1-6.ipynb` | `R1.6` | Ours | GNN encoder plus FEDformer decoder notebook for complex or magnitude S11 outputs, with augmentation preview, logging, and loss plotting. |
| `PatchAntennaAI_R1_7.ipynb` | `R1.7` | Ours | Refined GNN plus FEDformer notebook that defaults to magnitude-only output and keeps augmentation on the input geometry during training. |
| `PatchAntennaAI_R1_9.ipynb` | Filename says `R1.9`; internal title says `R1.8_CNN` | Ours | Magnitude-only model that replaces the GNN encoder with a CNN encoder and keeps the FEDformer-style decoder. |
| `antenna_sparams_colab.ipynb` | No explicit `R` tag | Ours | End-to-end S-parameter predictor with GNN plus FEDformer training, evaluation metrics, optional tiny synthetic dataset creation, and a Gradio interface for inference. |
| `patch_antenna_ai_colab_GPT_R4_LightGBM.ipynb` | `R4` | Ours | Non-neural baseline that trains 61 separate LightGBM regressors, one model per S11 frequency point. |
| `patch_antenna_ai_colab_R5.ipynb` | `R5` | Ours | Later GNN plus FEDformer notebook that keeps both augmentation and no-augmentation configuration sections and trains mainly in magnitude-only mode. |
| `patch_antenna_ai_colab_R5_section results.ipynb` | `R5` section-results copy | Ours | R5-style GNN plus FEDformer notebook with sectioned result cells and recorded validation-loss outputs. |
| `patch_antenna_ai_colab_ReduceModelComplexity.ipynb` | No explicit `R` tag in filename | Ours | Reduced-complexity GNN plus FEDformer version intended to make training lighter and simpler. |
| `patch_antenna_ai_colab_withLossGraph.ipynb` | No explicit `R` tag in filename | Ours | GNN plus FEDformer notebook that emphasizes saved loss curves, evaluation, and a Gradio-based prediction UI. |

### Scripts

| File | Version tag | Date/publish clue found locally | What it does |
| --- | --- |
| `model_training/scripts/ai_patch_antenna_gemini_r1_0.py` | `R1.0` | Ours | Python script export of the Gemini R1.0 TensorFlow CNN plus FEDformer-style notebook. |
| `model_training/scripts/ai_patch_antenna_gpt_r3_1.py` | `R3.1` | Ours | Python script export of the GPT R3.1 PyTorch CNN plus FEDformer baseline. |
| `model_training/scripts/patchantennaai_r1_8.py` | Filename says `R1.8`; internal title says `R1.7` | Ours | Python script version of the magnitude-only graph-based encoder plus FEDformer model. |
| `model_training/scripts/patchantennaai_r1_9.py` | Filename says `R1.9`; internal title says `R1.8_CNN` | Ours | Python script version of the magnitude-only CNN encoder plus FEDformer model. |

## Notes

- `/content/...`, locally in csv.
- Several files are iterative variants of encode a 10x10 binary patch geometry, then predict a 61-point S11 response curve.
