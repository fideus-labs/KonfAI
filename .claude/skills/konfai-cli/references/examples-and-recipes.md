# Verified end-to-end recipes

These are the exact command sequences from the runnable examples in `examples/`. Run every
command **from inside the example directory**: KonfAI resolves relative paths (`Dataset/`,
`Checkpoints/`, config files) against the current working directory.

The `train_name` set inside each `Config.yml` is what keys every output folder: the
examples below use `SEG_BASELINE`, `TRAIN_01` and `REG_BASELINE`.

Every example also ships a `*_demo.ipynb` that runs the whole sequence and plots the result;
their `epochs` are **demo-sized** (5, 1 and 400 respectively) so a first run finishes in
minutes. Raise `epochs` before reading anything into the scores.

## Segmentation (`examples/Segmentation`)

2D slice-wise multiclass segmentation baseline; model graph declared in `UNet.yml`, and
**`CrossEntropyLoss` + `Dice` together** for training, Dice for evaluation. The two losses read
different outputs of the same head: `UNetBlock_0:Head:Conv` wants logits, `UNetBlock_0:Head:Softmax`
wants probabilities. CrossEntropy alone is minimised by predicting background almost everywhere when
the 40 foreground labels each cover under 3% of a volume.

Dataset layout (one input image + one label map per case):

```text
Dataset/
├── 1PC006/
│   ├── CT.mha      # CT: input image
│   └── SEG.mha     # SEG: label map (0 = background, 1..40 = classes)
└── ...
```

```bash
cd examples/Segmentation

# 1. Train  ->  Checkpoints/SEG_BASELINE/ , Statistics/SEG_BASELINE/
konfai TRAIN -y --gpu 0 --config Config.yml

# 2. Predict  ->  Predictions/SEG_BASELINE/    (pick a checkpoint written by step 1)
konfai PREDICTION -y --gpu 0 --config Prediction.yml --models Checkpoints/SEG_BASELINE/<checkpoint>.pt

# 3. Evaluate  ->  Evaluations/SEG_BASELINE/
konfai EVALUATION -y --config Evaluation.yml
```

First things to adapt for a real project: `dataset_filenames`, `train_name`, patch size,
batch size, `nb_class`, preprocessing transforms, and `Dice.labels` in `Evaluation.yml`.

## Synthesis (`examples/Synthesis`)

MR→CT synthesis with a **local custom model** (`Model.py` defines `UNetpp5`, `Discriminator`,
`Gan`) and a custom post-processing transform (`UnNormalize.py`), referenced from YAML via
`classpath: Model:UNetpp5`. Dataset groups: `MR` (input), `CT` (target), `MASK` (masked
evaluation).

`Model.py` imports `segmentation_models_pytorch`, which the base install does not pull in: `pip install konfai[smp]` first.

**The preprocessing in `Prediction.yml` must mirror `Config.yml` exactly.** Standardizing the MR
inside the body mask at prediction time while training standardized on whole-volume statistics
feeds the network a scale it never saw: same checkpoint, 409 HU of MAE instead of 98, with no
error anywhere. This bites any two-config workflow, not just this example.

```bash
cd examples/Synthesis

# 1. Train  ->  Checkpoints/TRAIN_01/ , Statistics/TRAIN_01/   (use --cpu 1 if no GPU)
konfai TRAIN -y --gpu 0 --config Config.yml

# 2. Predict  ->  Predictions/TRAIN_01/
konfai PREDICTION -y --gpu 0 --config Prediction.yml --models Checkpoints/TRAIN_01/<checkpoint>.pt

# 3. Evaluate  ->  Evaluations/TRAIN_01/
konfai EVALUATION -y --config Evaluation.yml
```

### GAN variant

`Config_GAN.yml` trains `Model:Gan` (a 2.5D generator + 3D discriminator sharing the same
`UNetpp5`). Because the generator class name is unchanged, the same `Prediction.yml` reloads
either a baseline or a GAN checkpoint.

```bash
konfai TRAIN -y --gpu 0 --config Config_GAN.yml     # -> Checkpoints/TRAIN_GAN_01/
```

**Important:** before predicting/evaluating a *different* checkpoint source, set `train_name`
(and the prediction folder in `Evaluation.yml`) to match (e.g. `TRAIN_GAN_01`), so outputs
land in the right folder and evaluation reads the right predictions.

## Registration (`examples/Registration`)

Two-input deformable registration with the built-in diffeomorphic `VoxelMorph` (Python-only: its
custom `forward` has no YAML twin). `make_dataset.py` builds 30 `FIXED`/`MOVING` pairs from real
pelvis CT slices, `MOVING` being its `FIXED` pushed through a known smooth field, so the result is
checkable. The order of the `is_input` groups is load-bearing: `FIXED` is branch `0`, `MOVING`
branch `1`.

```bash
cd examples/Registration
python make_dataset.py                       # -> Dataset/<patient>_z<slice>/{FIXED,MOVING}.mha

konfai TRAIN -y --gpu 0 --config Config.yml   # -> Checkpoints/REG_BASELINE/
konfai PREDICTION -y --gpu 0 --config Prediction.yml --models Checkpoints/REG_BASELINE/<checkpoint>.pt
konfai EVALUATION -y --config Evaluation.yml  # MOVING:FIXED:* is the before, MOVED:FIXED:* the after
```

`VoxelMorph`'s `shape`, the training patch size, the prediction patch size and `CROP` in
`make_dataset.py` must all agree: a mismatch surfaces as a `state_dict` load error at PREDICTION,
not as a config error.

For registration between two *different patients* (a real anatomical difference with no
ground-truth field, scored by propagating one patient's reference labels) see
`examples/ImpactReg`, which drives the `impact-reg-konfai` app instead of training anything.

## The local-custom-code pattern

The Synthesis example is the template for custom architectures/transforms:

1. Put a `.py` next to the configs (e.g. `Model.py`, `UnNormalize.py`).
2. Reference it from YAML by classpath: a local file uses `File:Class` (e.g. `Model:UNetpp5`,
   `UnNormalize:UnNormalize`). KonfAI prepends the CWD to `sys.path`, so a module beside the
   config resolves.
3. Run the normal `konfai TRAIN/PREDICTION/EVALUATION` commands unchanged.

## Demo data

Segmentation and Synthesis pull a public subset from `huggingface.co/datasets/VBoussot/konfai-demo`:
the subset named after the example, flattened into `Dataset/` because the configs read `./Dataset`
directly.

```bash
hf download VBoussot/konfai-demo --repo-type dataset --include "Segmentation/**" --local-dir Dataset
mv Dataset/Segmentation/* Dataset/ && rmdir Dataset/Segmentation && rm -rf Dataset/.cache
```

Synthesis is the same two lines with `Synthesis` in place of `Segmentation`; Registration fetches its
subset through `make_dataset.py` instead. The `*_demo.ipynb` notebooks automate clone + install +
download + the whole workflow for a fresh machine or Colab: run every cell, nothing is gated behind
a flag.

## When to stop using raw YAML

Once a workflow is mature, the next step is to package it as a **KonfAI App** (see
`apps/impact_synth`) for a simpler user-facing interface, that is beyond this skill's scope.
