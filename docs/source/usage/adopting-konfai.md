# Adopt KonfAI from PyTorch, MONAI, or nnU-Net

KonfAI is not a replacement for every neighbouring tool. Use it when the
missing piece in your project is a single, inspectable medical-imaging workflow
from on-disk datasets through training, prediction, evaluation, and reusable
application delivery.

## Which tool fits which job?

| If your main need is… | Start with… | Where KonfAI fits |
| --- | --- | --- |
| Maximum breadth of medical transforms, losses, and networks | MONAI | Reuse MONAI components while KonfAI owns data layout, patch execution, artifacts, and application workflows. |
| A strong automatically configured supervised segmentation baseline | nnU-Net | Keep nnU-Net for auto-configuration; use compatible KonfAI graphs/checkpoint bridges when you need named internals or a broader workflow. |
| A mature general-purpose training-loop abstraction | PyTorch Lightning | Keep Lightning when generic training orchestration is the problem; use KonfAI when medical data, geometry, regional I/O, prediction datasets, and Apps are central. |
| A declarative end-to-end medical-imaging execution path | KonfAI | Configure train/predict/evaluate together, then package the same workflow for local, remote, Slicer, or agent-driven use. |

These choices are composable. KonfAI classpaths can instantiate installed
PyTorch and MONAI classes; gradual adoption does not require rewriting a proven
network or loss.

## Lowest-friction adoption: import the component

Use `module:Class` in YAML for any importable class. Examples include:

```yaml
Model:
  classpath: monai.networks.nets:SegResNet
```

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

An ordinary `torch.nn.Module` is wrapped for execution and exposes its final
output. Choose this route first when the existing forward is all you need.

## Bring your model: ten lines, no YAML

A model you already built in Python trains and predicts on a KonfAI dataset
through two calls, with the patching, the overlap blending, the streamed
writes and the run record of every other run:

```python
import konfai, torch
from konfai.metric.measure import CrossEntropyLoss
from konfai.data.transform import TensorCast

model = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(16, 2, 1))
checkpoints = konfai.train_model(
    model, "./Dataset:mha", inputs="CT", targets="SEG", loss=CrossEntropyLoss(),
    patch=[1, 256, 256], epochs=20, batch_size=8, transforms={"SEG": [TensorCast(dtype="int64")]},
)
konfai.predict_model(model, "./Dataset:mha", inputs="CT", patch=[1, 256, 256], output="./Pred:mha",
                     checkpoints=sorted(checkpoints.glob("*.pt"))[-1])
```

`inputs` and `targets` are the dataset's groups; `patch` is what the model is
fed, its non-unit axes deciding whether the model is 2D or 3D; `transforms`
and `augmentations` take the same objects the YAML would name. The workspace
keeps the resolved config as every run does, with the model named by a token
(`konfai.api:live_model`): the object lives in this process, so such a run
stays on one rank and cannot be resumed from another process. A relative
`output` lands under the run's workspace (`Predictions/<name>/`), an absolute
one where it says. Left without
`checkpoints`, `predict_model` writes the weights the module holds in memory
as a checkpoint for the run, so a model loaded any other way (a library's
pretrained weights, a foreign checkpoint) predicts as it stands. For several
GPUs, or to resume, spell the model as a classpath and use the YAML route.

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

Both need the `monai` extra (`pip install konfai[monai]`).

## When you need named internal outputs

KonfAI `Network` objects are routed graphs. Names supplied to `add_module()`
become stable paths that can receive losses, metrics, deep supervision, or
export rules. You have two options:

- wrap your architecture as a Python `Network` when it has custom control flow;
- use a registry-constrained YAML model when it is a feed-forward graph and you
  want the architecture itself to be diffable.

The YAML catalog includes architectures validated at documented equivalence
levels against MONAI, torchvision, nnU-Net's
`dynamic_network_architectures`, and segmentation-models-pytorch. Consult
{doc}`../concepts/yaml-model-builder` before assuming checkpoint compatibility:
some entries are weight-exact, while others are only structurally validated.

## Loading existing weights

For a weight-exact pair,
`konfai.utils.pretrained.transfer_weights_by_execution_order` pairs weighted
leaf modules in forward-execution order and checks every local state shape. It
is useful when graph names differ but execution structure and parameters match.
The config entry point is `Model.pretrained_from` (checkpoint + reference
builder classpath + its args), so a fresh TRAIN seeds from another framework's
checkpoint without any Python: see the
["Start from MONAI, torchvision or nnU-Net weights" section](../reference/components/models.md).

This bridge is strict, not universal. It raises `ConfigError` on a different leaf
count, a per-leaf key or shape mismatch, a target tensor no traced leaf owns, and a
tensor the target ties across two leaves, and it fills every target tensor or
raises, never reporting a partial load as success. What it cannot detect is
**ordering**: two identically-shaped leaves swapped would pair silently, since the
pairing is execution order itself. Unreached *source* branches (an nnU-Net
deep-supervision head) are ignored on purpose. Preserve
the reference preprocessing, class order, normalization, and output convention
when validating a transferred checkpoint.

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

## What KonfAI adds

- named medical dataset groups and physical geometry carried through output writing;
- a resolved configuration snapshot for the complete workflow;
- dataset-level and model-level patching;
- conservative regional reads with bounded-memory fallback;
- prediction-time batching, TTA, ensembles, reductions, reconstruction, and medical-image export;
- a common workspace for training, prediction, and evaluation;
- Apps for local/Hugging Face/HTTP use and an external 3D Slicer client;
- MCP tools that drive the same builders and artifacts as human-operated runs.

## What it does not add

- nnU-Net's automatic dataset fingerprinting and segmentation plan selection;
- MONAI's component breadth;
- Lightning's ecosystem and maturity for arbitrary training-loop patterns;
- a general spatial dependency compiler for every custom transform;
- proof of universal speedups over these tools: the tracked harness
  ({doc}`benchmarks`) reproduces the bounded-memory claim and pins the app
  tables' protocol, but the comparisons are per app and per case, not a
  general claim.

## Trust boundary

Python classpaths and KonfAI Apps import code. Resolving an App can also install
its `requirements.txt` by default. Treat local modules and remote app
repositories as executable code; use only trusted sources. Declarative YAML
model files have a narrower boundary: their node types come from curated
registries and do not evaluate arbitrary imports.

## Next steps

- {doc}`custom-models`: implementation contracts and complete examples
- {doc}`../concepts/yaml-model-builder`: graph schema and compatibility table
- {doc}`large-images`: regional I/O and memory trade-offs
- {doc}`apps`: package a stable workflow
