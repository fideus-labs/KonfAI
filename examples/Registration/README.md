# Registration Example

This example provides a **simple deformable image registration baseline** for KonfAI.

It is intentionally small and self-contained, and is meant to be:

- easy to read
- easy to adapt
- easy to use as a first registration template

The current baseline uses:

- the built-in diffeomorphic `VoxelMorph` model (`konfai/models/registration`)
- a 2D slice-wise setup
- patch-based training
- an `MSE` image-similarity loss during training
- `MAE` / `MSE` evaluation, comparing the registered image against the fixed image
- a fully synthetic dataset generated locally by `make_dataset.py`

The task is deliberately transparent: each `MOVING` image is the `FIXED` image translated by a
**known** shift, so the registered `MOVED` image can be checked numerically against `FIXED`.

## What you will find in this folder

```text
examples/Registration/
├── make_dataset.py
├── Config.yml
├── Prediction.yml
├── Evaluation.yml
└── README.md
```

- `make_dataset.py`: builds the synthetic `FIXED` / `MOVING` dataset (no download needed)
- `Config.yml`: training workflow
- `Prediction.yml`: inference workflow (produces the registered image `MOVED`)
- `Evaluation.yml`: evaluation workflow (before/after registration error)

## How the model is wired

`VoxelMorph` takes **two inputs** and the order of the input groups is load-bearing:

- the first `is_input` group (`FIXED`) is branch `0`
- the second `is_input` group (`MOVING`) is branch `1`

Internally it concatenates the two images, runs a small UNet, predicts a **diffeomorphic
deformation field**, and warps `MOVING` with it. The warped image is exposed by the module
`MovingImageResample`, which is:

- the target of the training loss (`MovingImageResample` vs `FIXED`)
- the model output saved at prediction time (as the `MOVED` group)

### Model form: Python by necessity

The Segmentation example ships its UNet in both a YAML and a Python form; `VoxelMorph`,
by contrast, is a **built-in Python model with no YAML twin**. Its scaling-and-squaring
integration of the velocity field and its spatial-transformer warp are a custom `forward`,
which the declarative YAML builder (a feed-forward `add_module` graph over curated block
types) cannot express. Custom-`forward` models — registration warps, diffusion samplers,
adversarial loops — stay in Python; standard feed-forward graphs can be YAML.

> `VoxelMorph` currently supports `dim: 2` only (its warping components are 2D-hardcoded),
> so this example is slice-wise. Keep `shape` equal to the `(Y, X)` size of the training patch.

## Dataset

This example does **not** need any external download. The dataset is generated procedurally
(Gaussian blobs, no patient data). Run all commands from this directory:

```bash
cd examples/Registration
python make_dataset.py
```

This writes:

```text
examples/Registration/
└── Dataset/
    ├── CASE_000/
    │   ├── FIXED.mha
    │   └── MOVING.mha
    ├── CASE_001/
    │   ├── FIXED.mha
    │   └── MOVING.mha
    └── ...
```

- `FIXED`: reference image
- `MOVING`: `FIXED` translated by a known shift (default `(5, 4)` voxels in `(Y, X)`)

## Quick start

Run all commands from this directory. Everything is CPU-friendly and finishes in a few minutes.

### 1. Train

```bash
konfai TRAIN -y --gpu 0 --config Config.yml
```

If you do not have a GPU available, use `--cpu 1` instead of `--gpu 0`.

This creates:

- `Checkpoints/REG_BASELINE/`
- `Statistics/REG_BASELINE/`

### 2. Predict

Use one checkpoint from `Checkpoints/REG_BASELINE/`:

```bash
konfai PREDICTION -y --gpu 0 --config Prediction.yml --models Checkpoints/REG_BASELINE/<checkpoint>.pt
```

This creates `Predictions/REG_BASELINE/`, where each case now has a `MOVED.mha` (the registered image).

### 3. Evaluate

```bash
konfai EVALUATION -y --config Evaluation.yml
```

This creates `Evaluations/REG_BASELINE/Metric_TRAIN.json`.

## Reading the metrics

`Evaluation.yml` computes two comparisons against `FIXED` so the improvement is visible in a single
JSON file:

- `MOVING:FIXED:MAE` / `MOVING:FIXED:MSE` — the error **before** registration (baseline)
- `MOVED:FIXED:MAE` / `MOVED:FIXED:MSE` — the error **after** registration

A successful run shows the `MOVED` error clearly below the `MOVING` error. On the shipped
synthetic dataset (60 epochs, CPU) a typical result is:

| Metric | Before (`MOVING` vs `FIXED`) | After (`MOVED` vs `FIXED`) |
|--------|------------------------------|----------------------------|
| MAE    | ~0.078                       | ~0.017                     |
| MSE    | ~0.021                       | ~0.0008                    |

The registered image is several times closer to the fixed image than the moving image was.

## Why training uses MSE here

This example optimizes only an image-similarity term (`MSE`) on purpose:

- the diffeomorphic velocity-field integration in `VoxelMorph` already regularizes the deformation
- keeping a single similarity loss makes the example easy to read and stable to train

For real deformable registration you will usually add a **smoothness regularizer** on the
deformation field (and often switch the similarity term to normalized cross-correlation).

## What to adapt first

For a real project, you will usually want to update:

1. `dataset_filenames`
2. `train_name`
3. the input groups (`FIXED` / `MOVING`) and their order
4. patch size and `shape` (keep them aligned)
5. batch size
6. preprocessing transforms (normalization for real intensities)
7. the similarity loss and, ideally, a deformation-field regularizer

## Recommended usage

Use this example when you want to:

- bootstrap a new registration experiment quickly
- understand the minimal KonfAI structure for image registration
- see how a two-input model is wired and how its warped output is scored against a reference
