# Segmentation Example

This example provides a **simple multiclass segmentation baseline** for KonfAI.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fideus-labs/KonfAI/blob/main/examples/Segmentation/Segmentation_demo.ipynb)

**Fastest way to run it:** open `Segmentation_demo.ipynb` and run every cell. It downloads the demo
data, trains, predicts, evaluates and plots the prediction next to the reference — about 7 minutes
on a GPU.

It is intentionally conservative and is meant to be:

- easy to read
- easy to adapt
- easy to use as a first segmentation template

The current baseline uses:

- the routed KonfAI model graph declared in `UNet.yml`
- a 2D slice-wise setup
- patch-based training
- `CrossEntropyLoss` + `Dice` during training
- Dice evaluation after prediction
- `41` classes in total (`0` for background, `1..40` for labels)

## What you will find in this folder

```text
examples/Segmentation/
├── Config.yml
├── UNet.yml
├── Model.py
├── Prediction.yml
├── Evaluation.yml
├── README.md
└── Segmentation_demo.ipynb
```

- `Config.yml`: training workflow
- `UNet.yml`: UNet modules, nested skip connections, parameters, and routing
- `Model.py`: the **same** UNet written as a Python class
- `Prediction.yml`: inference workflow
- `Evaluation.yml`: evaluation workflow
- `Segmentation_demo.ipynb`: guided onboarding notebook

## Two ways to define the model

KonfAI accepts a model as a declarative **YAML graph** or as a **Python class**, and this
example ships both — they build the same network, so they are interchangeable. Pick one in
`Config.yml`/`Prediction.yml`:

```yaml
Model:
  classpath: UNet.yml      # declarative form (this is the default here)
  # classpath: Model:UNet  # the Python form in Model.py — same architecture
```

Use the **YAML form** for a no-code, shareable model (safe by construction — it can only
reference a curated set of block types). Reach for the **Python form** when a model needs a
custom `forward` or logic a declarative graph cannot express (the Synthesis example is such a
case). Because both expose the same named output (`Config.yml` keys the loss on
`UNetBlock_0:Head:Conv`), `outputs_criterions` is unchanged when you swap between them.

The notebook is designed to work from a **fresh environment**, including **Google Colab**. Its setup cells can:

- clone the KonfAI repository if needed
- install KonfAI and its Python dependencies
- download the public segmentation demo subset automatically

## Expected dataset layout

```text
Dataset/
├── CASE_000/
│   ├── CT.mha
│   └── SEG.mha
├── CASE_001/
│   ├── CT.mha
│   └── SEG.mha
└── ...
```

- `CT`: input image
- `SEG`: segmentation label map

The default template assumes:

- a multiclass task
- label `0` as background
- labels `1..40` as foreground classes
- one input image per case
- `SEG` stored as a label map with integer values

## Demo data

The public Hugging Face demo dataset is available at:

- `https://huggingface.co/datasets/VBoussot/konfai-demo`

If you want the easiest first run, use `Segmentation_demo.ipynb`.

If you prefer to fetch the demo subset manually, use the Hugging Face CLI:

```bash
python -m pip install -U "huggingface_hub[cli]"
hf download VBoussot/konfai-demo \
  --repo-type dataset \
  --include "Segmentation/**" \
  --local-dir Dataset
mv Dataset/Segmentation/* Dataset/
rmdir Dataset/Segmentation
rm -rf Dataset/.cache
```

After that, your local `Dataset/` folder should already match the structure expected by this example.

## Quick start

`Segmentation_demo.ipynb` runs all three steps below and plots the result. To do it by hand instead,
run every command from this directory, with `Dataset/` in place:

```bash
cd examples/Segmentation
```

### 1. Train

```bash
konfai TRAIN -y --gpu 0 --config Config.yml
```

### 2. Predict

Checkpoints are named after the moment they were written, and this example keeps only the best one, so
a glob resolves to exactly one file:

```bash
konfai PREDICTION -y --gpu 0 --config Prediction.yml --models Checkpoints/SEG_BASELINE/*.pt
```

### 3. Evaluate

```bash
konfai EVALUATION -y --config Evaluation.yml
```

This produces:

- `Checkpoints/SEG_BASELINE/`
- `Predictions/SEG_BASELINE/`
- `Evaluations/SEG_BASELINE/`

## What to adapt first

For a real project, you will usually want to update:

1. `dataset_filenames`
2. `train_name`
3. patch size
4. batch size
5. number of classes
6. preprocessing transforms
7. Dice labels in `Evaluation.yml`
8. model channels and scheduler

For multiclass segmentation:

- update `nb_class`
- update `Dice.labels`
- review the label encoding in your dataset

## Why training combines CrossEntropy and Dice

The 40 foreground labels each cover well under 3% of a volume, so `CrossEntropyLoss` on its own is
minimised by predicting background nearly everywhere: after 10 epochs it scores a mean Dice of
**0.05** and finds only four structures.

Adding a `Dice` loss on the `Softmax` output fixes that — the overlap term is scale-free, so a small
organ counts as much as a large one. Five epochs of the two losses together reach a mean Dice of
**0.19** across ten structures, in half the training time.

The two losses read different outputs of the same head, which is why `Config.yml` lists them under
two keys: `UNetBlock_0:Head:Conv` (raw logits, what CrossEntropy expects) and
`UNetBlock_0:Head:Softmax` (class probabilities, what Dice expects).

## How good is the demo result?

`epochs: 5` is sized so the notebook finishes; it is not a trained model. Expect a mean Dice around
`0.19`, with the large structures (bone, muscle, bowel) clearly recognisable and the small ones
missing. Raise `epochs` to 100+ for anything you intend to use.

Two things keep that number optimistic, and both are deliberate for a demo: the five cases are also
the training set except for the one held out by `validation: 0.2`, and `Dice.labels: None` averages
over every label the reference contains — including the eighteen that never appear in a pelvis scan
and therefore score zero. Set `Dice.labels` in `Evaluation.yml` to the labels you actually care about
before reading anything into the score.

## Recommended usage

Use this example when you want to:

- bootstrap a new segmentation experiment quickly
- understand the minimal KonfAI structure for segmentation
- create your own YAML template before moving to stronger architectures or 3D workflows
