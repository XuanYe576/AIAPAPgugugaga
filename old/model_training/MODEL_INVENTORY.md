# Project task

- Input: a `10 x 10` binary patch-antenna geometry, flattened to `100` values
- Output: an S11 response curve with `61` frequency samples

Depending on the experiment, the target is one of:

- `61` real S11 values
- `61 x 2` complex outputs represented as real + imaginary parts
- `61` magnitude-only outputs

## Important clarification

Some filenames contain `GPT` or `Gemini`, but the repository does not contain OpenAI or Google Gemini API usage.
Those names appear to be experiment labels or notebook naming choices, not deployed foundation-model dependencies.

### UI / demo layer

- `Gradio` is used in some notebooks for inference demos, but it is not a model

## Model families

### 1. GNN / GCN encoder + FEDformer-style decoder

This is the main early model family.

Typical structure:

- build a `10 x 10` grid graph from the geometry
- run graph convolutions over the `100` nodes
- pool to a global geometry embedding
- expand that embedding into a sequence of `61` tokens
- decode with a spectral / Fourier-style block plus Transformer-style layers

Common files in this family:

- `model_training/notebooks/patchAntennaAI_R1-6.ipynb`
- `model_training/notebooks/PatchAntennaAI_R1_7.ipynb`
- `model_training/notebooks/Patch Antenna_Colab_R2-1.ipynb`
- `model_training/notebooks/R2-3_With Augmentation.ipynb`
- `model_training/notebooks/antenna_sparams_colab.ipynb`
- `model_training/notebooks/patch_antenna_ai_colab_R5.ipynb`
- `model_training/notebooks/patch_antenna_ai_colab_R5_section results.ipynb`
- `model_training/notebooks/patch_antenna_ai_colab_ReduceModelComplexity.ipynb`
- `model_training/notebooks/patch_antenna_ai_colab_withLossGraph.ipynb`
- `model_training/scripts/patchantennaai_r1_8.py`

Key concepts used in this family:

- graph adjacency over the `10 x 10` layout
- `GraphConv` / `GridGraphEncoder`
- Transformer encoder layers
- FEDformer-style spectral decoding
- positional encoding over frequency sequence

### 2. CNN encoder + FEDformer-style decoder

This is the later and more common family in the repo.

Typical structure:

- reshape the `100` geometry bits into a `10 x 10` image
- encode with stacked `Conv2d` layers
- map to a latent embedding
- create `61` sequence tokens
- decode with a simplified FEDformer-style spectral block and Transformer-style layers

Common files in this family:

- `model_training/notebooks/AI_Patch_Antenna_Gemini_R1_0.ipynb`
- `model_training/notebooks/PatchAntennaAI_R1_9.ipynb`
- `model_training/notebooks/AI_Patch_Antenna_GPT_R3_1.ipynb`
- `model_training/notebooks/AI_Patch_Antenna_GPT_R3_2.ipynb`
- `model_training/notebooks/AI_Patch_Antenna_GPT_R3_2_With results.ipynb`
- `model_training/notebooks/AI_Patch_Antenna_GPT_R3_4.ipynb`
- `model_training/notebooks/AI_Patch_Antenna_GPT_R3_4_.ipynb`
- `model_training/notebooks/AI_Patch_Antenna_GPT_R3_5_Resonance_weighted_loss.ipynb`
- `model_training/notebooks/AI_Patch_Antenna_GPT_R3_6_2DPositional.ipynb`
- `model_training/scripts/ai_patch_antenna_gemini_r1_0.py`
- `model_training/scripts/ai_patch_antenna_gpt_r3_1.py`
- `model_training/scripts/patchantennaai_r1_9.py`

Notable variants inside this family:

- `R3.1`: CNN + FEDformer baseline with augmentation
- `R3.2`: same baseline with augmentation disabled
- `R3.4`: normalization, longer training, LR scheduling, gradient clipping
- `R3.5`: resonance-weighted loss
- `R3.6`: extra 2D positional channels for row/column coordinates

### 3. Tiny CNN direct regressor

- small CNN over the `10 x 10` geometry
- direct regression head to the `61` S11 values
- no FEDformer-style decoder

Main file:

- `model_training/notebooks/AI_Patch_Antenna_GPT_R3_3_Tiny CNN.ipynb`

### 4. LightGBM baseline

This is the main non-neural baseline.

Typical structure:

- flatten the geometry input
- train `61` separate `LightGBM` regressors
- each regressor predicts one frequency point of the S11 curve

Main file:

- `model_training/notebooks/patch_antenna_ai_colab_GPT_R4_LightGBM.ipynb`

## Common training techniques and experiment knobs

Across the notebooks, the following ML ideas appear repeatedly:

- data augmentation on the geometry input
- random flips and rotations
- additive Gaussian noise
- train/validation splits
- early stopping or best-checkpoint selection
- `Adam` or `AdamW` optimization
- MSE-based losses
- resonance-weighted MSE in `R3.5`
- normalization / de-normalization of S11 targets
- magnitude-only versus complex-output training

## What is not present in the repo

I did not find repository-local evidence of:

- OpenAI API usage
- Google Gemini API usage
- Hugging Face `transformers`
- pretrained LLMs or VLMs
- ONNX or other exported deployment model files committed into the repo

The notebooks save checkpoints under Colab-style paths such as `/content/...`, but those trained weight files are not present in this repository snapshot.

## Short answer

If you need the one-line summary:

- Main ML stack: `PyTorch`, `TensorFlow/Keras`, `LightGBM`
- Main model families: `GNN/GCN + FEDformer-style decoder`, `CNN + FEDformer-style decoder`, `Tiny CNN regressor`, and `61-per-frequency LightGBM regressors`
