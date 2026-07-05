# Verified end-to-end recipes

These are the exact command sequences from the runnable examples in `examples/`. Run every
command **from inside the example directory** — KonfAI resolves relative paths (`Dataset/`,
`Checkpoints/`, config files) against the current working directory.

The `train_name` set inside each `Config.yml` is what keys every output folder — the
examples below use `SEG_BASELINE` and `TRAIN_01`.

## Segmentation (`examples/Segmentation`)

2D slice-wise multiclass segmentation baseline; model graph declared in `UNet.yml`,
`CrossEntropyLoss` for training, Dice for evaluation.

Dataset layout (one input image + one label map per case):

```text
Dataset/
├── CASE_000/
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
(and the prediction folder in `Evaluation.yml`) to match — e.g. `TRAIN_GAN_01` — so outputs
land in the right folder and evaluation reads the right predictions.

## The local-custom-code pattern

The Synthesis example is the template for custom architectures/transforms:

1. Put a `.py` next to the configs (e.g. `Model.py`, `UnNormalize.py`).
2. Reference it from YAML by classpath — a local file uses `File:Class` (e.g. `Model:UNetpp5`,
   `UnNormalize:UnNormalize`). KonfAI prepends the CWD to `sys.path`, so a module beside the
   config resolves.
3. Run the normal `konfai TRAIN/PREDICTION/EVALUATION` commands unchanged.

## Demo data

Both examples pull a public subset from `huggingface.co/datasets/VBoussot/konfai-demo`
(`hf download VBoussot/konfai-demo --repo-type dataset --include "Segmentation/**" --local-dir Dataset`).
The `*_demo.ipynb` notebooks automate clone + install + download for a fresh machine or Colab.

## When to stop using raw YAML

Once a workflow is mature, the next step is to package it as a **KonfAI App** (see
`apps/impact_synth`) for a simpler user-facing interface — that is beyond this skill's scope.
