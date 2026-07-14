# Examples

This directory contains the **low-level KonfAI workflows** used to design, test, and understand the framework directly from YAML configuration files.

If you are new to KonfAI, this is the best place to start before moving to packaged **KonfAI Apps**.

Both example notebooks are designed to be friendly to a fresh environment, including **Google Colab**:

- they can bootstrap KonfAI from the repository
- they can download the public demo data automatically
- they expose the standard `TRAIN -> PREDICTION -> EVALUATION` loop

## What is inside

### `Synthesis`

Medical image synthesis example based on:

- `Config.yml` for training
- `Prediction.yml` for inference
- `Evaluation.yml` for evaluation
- `Model.py` and `UnNormalize.py` for local custom Python modules

Use this example if you want to understand how KonfAI combines:

- YAML configuration
- local Python model definitions
- preprocessing and postprocessing
- train / prediction / evaluation loops

### `Segmentation`

Multiclass segmentation baseline based on the built-in UNet.

Use this example if you want a simple and conservative starting point for:

- segmentation datasets
- patch-based training
- prediction export
- Dice-based evaluation

### `Registration`

Deformable image registration baseline based on the built-in diffeomorphic `VoxelMorph`.

Use this example if you want a simple starting point for:

- a two-input (`FIXED` / `MOVING`) model
- patch-based training with an image-similarity loss
- exporting the registered image
- checking the registration numerically against a reference

It ships a `make_dataset.py` that builds a fully synthetic dataset locally, so it runs end to end
without any download.

## Demo data

The public demo dataset is available on Hugging Face:

- `https://huggingface.co/datasets/VBoussot/konfai-demo`

It currently provides:

- `Synthesis/`
- `Segmentation/`

## Recommended order

If you are discovering KonfAI, a good progression is:

1. start with `examples/Synthesis`
2. read the notebook and run the workflow once
3. inspect the generated checkpoints, predictions, and logs
4. move to `examples/Segmentation`
5. adapt one example to your own dataset

## Notebooks

Each example now includes a notebook intended as an onboarding companion:

- `examples/Synthesis/Synthesis_demo.ipynb`
- `examples/Segmentation/Segmentation_demo.ipynb`

These notebooks focus on:

- dataset preparation
- empty-environment bootstrap
- folder layout
- configuration files
- example commands
- practical tips for adapting the workflow
