# Segmentation

A multiclass segmentation baseline you can read in one sitting and adapt to your
own data: a 2D slice-wise UNet, patch-based training, 41 classes.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fideus-labs/KonfAI/blob/main/examples/Segmentation/Segmentation_demo.ipynb)

**To just run it**, open `Segmentation_demo.ipynb` and run every cell: it fetches
the data, trains, predicts, evaluates and plots the result, about 7 minutes on a
GPU. The [quickstart](https://konfai.readthedocs.io/en/latest/quickstart.html)
does the same from the command line. This page is for what comes after: changing
it into your experiment.

## The files

| File | What it holds |
| --- | --- |
| `Config.yml` | training: model, dataset, losses, optimizer |
| `Prediction.yml` | inference: which outputs are written, under which names |
| `Evaluation.yml` | scoring: predictions against references |
| `UNet.yml` | the model as a declarative graph |
| `Model.py` | the same model as a Python class |
| `Segmentation_demo.ipynb` | the whole loop, cell by cell |

## Your data instead of the demo

The example expects one folder per case, with the group names used in
`groups_src`:

```text
Dataset/CASE_000/CT.mha     # the input image
Dataset/CASE_000/SEG.mha    # the label map: 0 background, 1..40 foreground
```

Point `dataset_filenames` at your own folder laid out that way. `SEG` must hold
integer labels, one input image per case.

The demo data lives at
[VBoussot/konfai-demo](https://huggingface.co/datasets/VBoussot/konfai-demo).
The notebook fetches it; by hand it is:

```bash
python -m pip install -U "huggingface_hub[cli]"
hf download VBoussot/konfai-demo --repo-type dataset \
  --include "Segmentation/**" --local-dir Dataset
mv Dataset/Segmentation/* Dataset/ && rmdir Dataset/Segmentation && rm -rf Dataset/.cache
```

## What to change first

1. `dataset_filenames` and `train_name`, always. `train_name` names the run
   everywhere: `Checkpoints/SEG_BASELINE/`, `Predictions/SEG_BASELINE/`,
   `Evaluations/SEG_BASELINE/`. The three configs must agree on it, or
   evaluation will not find the predictions.
2. `nb_class`, and `Dice.labels` in `Evaluation.yml` to the labels you care
   about.
3. Patch size and batch size, to your GPU.
4. The preprocessing transforms, to your modality.
5. Model channels and the scheduler, once the rest works.

## Two ways to write the model

Both files build the same network, so `Config.yml` and `Prediction.yml` take
either:

```yaml
Model:
  classpath: UNet.yml      # declarative, the default here
  # classpath: Model:UNet  # the same network in Python
```

The YAML form is shareable and safe by construction: it can only reference a
curated set of block types. The Python form is for a model that needs a custom
`forward`, which is what the Synthesis example does. Both expose the same named
output, so `outputs_criterions` does not change when you swap.

## Two losses on two named outputs

Every module output of a KonfAI graph has a name, and a loss attaches to one of
them. Here the same head feeds two:

```yaml
outputs_criterions:
  UNetBlock_0:Head:Conv:      # raw logits
    ...CrossEntropyLoss
  UNetBlock_0:Head:Softmax:   # class probabilities
    ...Dice
```

That is the mechanism to reuse: any number of losses, on any named output,
including intermediate ones for deep supervision. The key is the module's dotted
path in the graph, so `UNet.yml` and the config have to agree on it.

## Reading the demo score

`epochs: 5` is sized so the notebook finishes, so the score says the pipeline
works, not that the model does. Two config keys decide what you are reading:
`validation: 0.2` holds one case out of five, and `Dice.labels: None` averages
over every label present in the reference. Set `Dice.labels` in `Evaluation.yml`
to the labels you care about.
