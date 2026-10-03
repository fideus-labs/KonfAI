# Adopt KonfAI from PyTorch, MONAI, or nnU-Net

Reuse your PyTorch or MONAI model with KonfAI for dataset loading, training, prediction
and evaluation. Start by importing the model; add named internal outputs only when you need them.

## Import an existing component

Name any installed class with `module:Class`. A model that is not a KonfAI `Network` is wrapped: its
arguments go under its class name, and its output is called `Model` (the key losses attach to). A MONAI UNet
(`pip install "konfai[monai]"`). This is a `Model` block to insert under
`Trainer` in a complete training configuration:

```yaml
Model:
  classpath: monai.networks.nets:UNet
  UNet:
    spatial_dims: 2
    in_channels: 1
    out_channels: 2
    channels: [8, 16, 32]
    strides: [2, 2]
    optimizer:
      name: AdamW
      lr: 0.01
    outputs_criterions:
      Model:
        targets_criterions:
          SEG:
            criterions_loader:
              CrossEntropyLoss:
                is_loss: true
```

Losses and metrics import the same way:

```yaml
criterions_loader:
  monai.losses:DiceLoss:
    include_background: false
    to_onehot_y: true
    softmax: true
```

```yaml
criterions_loader:
  torch:nn:L1Loss:
    reduction: mean
```

A wrapped model exposes only its final output. Start here when that is all you need.

## Train a model built in Python

A model built in Python trains and predicts on a KonfAI dataset in two calls, with KonfAI's patching,
blending and run record:

```python
import konfai
import torch
from konfai.metric.measure import CrossEntropyLoss
from konfai.data.transform import Argmax, TensorCast

if __name__ == "__main__":
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 2, 1),
    )
    checkpoints = konfai.train_model(
        model, "./Dataset:mha", inputs="CT", targets="SEG", loss=CrossEntropyLoss(),
        patch=[1, 256, 256], epochs=20, batch_size=8, transforms={"SEG": [TensorCast(dtype="int64")]},
    )
    konfai.predict_model(
        model,
        "./Dataset:mha",
        inputs="CT",
        patch=[1, 256, 256],
        output="./Pred:mha",
        checkpoints=sorted(checkpoints.glob("[0-9]*.pt"))[-1],
        final_transforms=[Argmax(), TensorCast(dtype="uint8")],
    )
```

- `inputs` and `targets` are the dataset's groups; `patch` is what the model sees (its non-1 axes make the
  model 2-D or 3-D).
- The output is written as the model produces it, unless `final_transforms` changes it (here, a `uint8`
  label map).
- `train_model` returns the checkpoint folder; `[0-9]*.pt` is the best checkpoint.
- Without `checkpoints`, `predict_model` uses the weights the model holds, so a model loaded any other way
  predicts as it is.
- The model lives in your process, so this route runs on one GPU and cannot be resumed from another process.
  For several GPUs or to resume, name the model by classpath and use YAML.

## MONAI Bundles, both ways

A [MONAI Bundle](https://docs.monai.io/en/stable/mb_specification.html) (the
model zoo's format: `metadata.json`, `configs/inference.json`, `models/model.pt`)
imports as the `Model` block of a `Prediction.yml` plus a KonfAI checkpoint:

```python
imported = konfai.import_bundle("./spleen_ct_segmentation")
imported.classpath        # 'monai.networks.nets:UNet'
imported.arguments        # the class's arguments, every @reference of the bundle resolved
imported.checkpoint       # models/konfai_model.pt, what `konfai PREDICTION --models` takes
imported.model_tree()     # the Model block, ready for a config tree or a Prediction.yml
imported.untranslated     # the bundle's preprocessing and inferer, reported, not translated
```

The bundle's preprocessing and inferer are MONAI transforms on MONAI's runtime:
spell the equivalent KonfAI stages under `transforms:` (`Standardize`,
`Resample`, `Clip`, and the `Patch` block for the sliding window). The other
way, a loaded KonfAI network's inference head becomes a bundle any MONAI
runtime loads (`models/model.ts`, traced, with its `metadata.json` and an
`inference.json`):

```python
konfai.export_bundle(network, torch.zeros(1, 1, 96, 96, 96), "./my_bundle", name="my_model")
```

Both need the `monai` extra (`pip install "konfai[monai]"`).

## When you need named internal outputs

A KonfAI `Network` names its modules, so losses, metrics and saved outputs can attach to any of them. Write
your architecture as a Python `Network` when it has custom control flow, or as a YAML model when it is a
feed-forward graph. The YAML catalog lists, for each model, whether its weights match MONAI, torchvision,
nnU-Net or segmentation-models-pytorch exactly ({doc}`../reference/components/models`).

## Loading existing weights

`Model.pretrained_from` starts a training from another framework's checkpoint, and checks that the KonfAI
model reproduces the original's outputs ({doc}`../reference/components/models`). From Python,
`konfai.utils.pretrained.transfer_weights_by_execution_order` copies the weights layer by layer and refuses
any mismatch, but does not compare outputs: check them yourself. Keep the original preprocessing, class
order and normalisation when you validate a transferred model.

## A gradual migration path

1. Keep your existing dataset and model; reproduce one inference through a
   KonfAI `Prediction.yml`.
2. Compare tensors and saved medical-image geometry against your reference
   pipeline.
3. Move training losses and metrics into `Config.yml` while leaving the model
   imported as a black box.
4. Convert only the subgraphs whose internal outputs you need to address.
5. Add regional storage and dataset patching if case size requires it.
6. Package the stable prediction/evaluation assets as a KonfAI App.

At each step, keep a reference case and test numerical outputs before changing
the next layer.

## What KonfAI adds, and what it does not

KonfAI adds: named dataset groups with their geometry kept through to the written output; one resolved
config for the whole workflow; patching at the dataset and model levels; streamed reads and writes;
ensembles and test-time augmentation; apps usable locally, from Hugging Face, over HTTP, from Slicer, and
through clients through MCP.

It does not add nnU-Net's automatic planning, MONAI's breadth of components, or Lightning's ecosystem for
arbitrary training loops.

## Trust boundary

Python classpaths and apps run code, and an app installs its `requirements.txt` by default: use trusted
sources only. YAML models are safer: they can only use registered module types.

## Next steps

- {doc}`custom-models`: implementation contracts and complete examples
- {doc}`../reference/components/models`: graph schema and compatibility table
- {doc}`large-images`: regional I/O and memory trade-offs
- {doc}`apps`: package a stable workflow

## Which tool fits which job?

| If your main need is… | Start with… | Where KonfAI fits |
| --- | --- | --- |
| Maximum breadth of medical transforms, losses, and networks | MONAI | Reuse MONAI components while KonfAI owns data layout, patch execution, artifacts, and application workflows. |
| A strong automatically configured supervised segmentation baseline | nnU-Net | Keep nnU-Net for auto-configuration; use compatible KonfAI graphs/checkpoint bridges when you need named internals or a broader workflow. |
| A mature general-purpose training-loop abstraction | PyTorch Lightning | Keep Lightning when generic training orchestration is the problem; use KonfAI when medical data, geometry, regional I/O, prediction datasets, and Apps are central. |
| A declarative end-to-end medical-imaging execution path | KonfAI | Configure train/predict/evaluate together, then package the same workflow for local, remote, Slicer, or automated use. |

They combine: KonfAI can use installed PyTorch and MONAI classes, so adopting it does not mean rewriting a
network or a loss.
