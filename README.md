# FloodDiffusion 2

**Efficient and Path Controllable Streaming Motion Generation**

FloodDiffusion 2 (FD2) improves the efficiency and controllability of
FloodDiffusion while preserving continuous motion generation from changing text
prompts. The paper develops three parts:

1. **Partial Attention** makes finalized history independent of the active
   denoising window, enabling KV-cached inference and shared-history training.
2. **Diffusion-compatible geometric supervision** uses the necessary-and-sufficient
   Bregman criterion for conditional-mean regression, under its stated regularity
   conditions, to design an FK-induced quadratic loss without online FK evaluation.
3. **Root-path conditioning** controls the trajectory while generating natural
   body motion. On SEED, the representation separates the ground-projected body
   center of mass from root-relative pelvis translation and rotation.

This repository contains FD2 networks, training and evaluation entries, motion
recovery and rendering.

## Installation

```bash
git clone https://github.com/caiyy17/FloodDiffusion2.git
cd FloodDiffusion2
conda create -n motion_gen python=3.10
conda activate motion_gen
python setup_project.py
```

From a Python 3.10+ environment, `python setup_project.py` installs the Python
dependencies and downloads the four paper checkpoints, their normalization/FK assets,
and the shared UMT5, T2M and GloVe dependencies. Downloads resume after interruption
and files are checked against the release manifest. To also download the prepared
datasets, run `python setup_project.py --with-data`.

Use `python setup_project.py --skip-install` with an existing environment, or
`python setup_project.py --check` to verify installed assets without downloading.

The environment uses PyTorch 2.9.1 (CUDA 12.8 on Linux). Attention uses PyTorch
SDPA; a separate Flash Attention package is not required.
We have tested training on a single NVIDIA H200 and inference on an NVIDIA RTX 4090.
The tested Python dependency versions are recorded in [requirements-tested.txt](requirements-tested.txt) for reference.

Linux with an NVIDIA GPU is recommended; Windows can use WSL2. Mesh rendering
uses SMPL-H and an OpenGL/EGL runtime.
For headless NVIDIA rendering, the driver must provide `libEGL_nvidia.so.0`.
If the driver is present but its EGL vendor discovery files are missing, the
renderer uses the bundled vendor description for that process. Explicit EGL
settings are preserved. On multi-GPU systems, select the rendering GPU separately
with `EGL_DEVICE_ID`; `CUDA_VISIBLE_DEVICES` does not renumber EGL devices.

## Configurations

The configurations use one GPU. Diffusion configurations use partial attention;
training windows are set in `schedule_config.train_n_windows`.

| Configuration | Model / data |
|---|---|
| `configs/df_humanml3d_263.yaml` | HumanML3D, 263D, text-conditioned |
| `configs/df_humanml3d_263_path.yaml` | HumanML3D, 263D, path-conditioned |
| `configs/df_seed_138.yaml` | SEED, 138D, text-conditioned |
| `configs/df_seed_138_path.yaml` | SEED, 138D, path-conditioned |

`configs/paths_default.yaml` defines relative asset locations. Put machine-specific
paths in `configs/paths.yaml`, which is excluded from Git:

```yaml
dirs:
  deps: ./deps
  raw_data: ./data
  outputs: ./outputs
  checkpoints: ./checkpoints
```

## Weights and dependencies

The [Hugging Face release](https://huggingface.co/caiyiyi1998/FloodDiffusion2)
contains the four paper checkpoints below, each with its matching config and assets.
The setup command places each at the path already specified in its configuration.

| Configuration | Checkpoint step | CFG |
|---|---:|---:|
| `df_humanml3d_263.yaml` | 60,000 | 4 |
| `df_humanml3d_263_path.yaml` | 55,000 | 3 |
| `df_seed_138.yaml` | 300,000 | 2 |
| `df_seed_138_path.yaml` | 300,000 | 2 |

Additional checkpoints and their configs are available in the Hugging Face release.
The shared UMT5, T2M and GloVe dependencies are hosted in the
[FloodDiffusion 2 release](https://huggingface.co/caiyiyi1998/FloodDiffusion2/blob/main/deps.zip).

Text generation uses the UMT5-XXL encoder:

```text
deps/t5_umt5-xxl-enc-bf16/
├── models_t5_umt5-xxl-enc-bf16.pth
└── google/umt5-xxl/
    ├── spiece.model
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

The setup command also installs T2M evaluator weights/statistics under `deps/t2m/`
and GloVe assets under `deps/glove/`.

For SMPL-H mesh rendering, obtain the neutral `model.npz` from the
[official SMPL+H download](https://mano.is.tue.mpg.de/) and import it during setup:

```bash
python setup_project.py --smplh /path/to/neutral/model.npz
```

It is placed at `deps/smplh/neutral/model.npz`. The body model requires its own
license and is not redistributed here. Numerical motion generation, skeleton
recovery and metrics use the supplied representation code. Recomputing the
SEED mesh-induced FK matrix also uses this body model.

Each checkpoint folder includes `assets/Mean.npy`, `assets/Std.npy`, and `assets/W.npy`. The configurations use these released assets for training;
inference also restores the corresponding buffers from the checkpoint.

To recompute a matrix from training data, use the matching configuration:

```bash
python prepare_fk_matrix.py --config configs/df_humanml3d_263.yaml --output outputs/recomputed_W.npy
```

The configuration's `fk_matrix.recipe` selects the released construction:
HumanML3D uses the full 64D position block (root height and 21 joint positions)
normalized to trace 64, with unit
weights on the remaining channels. Its path variant uses the 260D principal
block, normalized to trace 260. SEED uses the mesh pullback estimated from two
500-clip training samples, five noninitial frames per clip; its path variant
uses the 135D principal block normalized to trace 135.

## Data and training

The [prepared datasets](https://huggingface.co/datasets/caiyiyi1998/FloodDiffusion2-Data)
are available as separate ZIP archives. `python setup_project.py --with-data`
downloads and extracts the released data packs. The paper configurations use:

```text
data/
├── HumanML3D/HumanML3D263/  # new_joint_vecs/, texts/, split files
└── SEED/MEI138/            # new_joint_vecs_uni/, texts/, split files
```

A new training run checks that its configured normalization statistics and FK
loss matrix exist before loading the model. Evaluation and full-state resume
can restore those buffers from a matching checkpoint. The released asset
statistics match the released checkpoints.

```bash
python train_df.py --config configs/df_humanml3d_263.yaml
```

Use the corresponding path or SEED configuration with `train_df.py` to train
those variants. W&B logging is optional and reads `WANDB_API_KEY` and
`WANDB_ENTITY` from the environment.

## Evaluation and test rendering

Evaluate the checkpoint already selected by a configuration:

```bash
python evaluate.py --config configs/df_humanml3d_263.yaml
```

The same command supports the four paper configurations. The YAML's `val_meta_paths`
select metric evaluation data; `test_meta_paths`
select generation and rendering examples (normally `test_min.txt`). Results
are written below `dirs.outputs`. For generation and rendering without T2M
metrics, add `metrics.t2m=null` to the overrides. Path configurations read the
root conditions from the input motion data.

For SEED evaluation with the official Kimodo TMR and physical metrics, use
[tools/seed_official](tools/seed_official/README.md). It supports both released
SEED configurations, including MEI138-to-SOMA conversion and path conditioning.

## Code layout

- `models/`: diffusion, VAE, text encoder, attention, and loss implementations.
- `datasets/`: loaders and batching for the configured representations.
- `metrics/`: evaluation implementations.
- `visualization/`: representation recovery and rendering.
- `train_df.py`, `train_ldf.py`, `train_vae.py`: training and evaluation entries.
- `evaluate.py`: configuration-selected evaluation entry.
- `prepare_fk_matrix.py`, `tools/fk/`: configuration-selected FK matrix preparation.
- `tools/seed_official/`: adapter for the official SEED evaluation pipeline.

## License

This project is licensed under the [MIT License](LICENSE).
See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for third-party components.
Third-party model weights and datasets retain their respective licenses.
