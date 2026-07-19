# Model graph and output naming

This page explains the naming scheme behind every `outputs_criterions` and
`outputs_dataset` key — how KonfAI addresses individual modules inside a model
graph. Read it before attaching a loss, metric, or exported prediction to a
model output.

KonfAI models are not treated as opaque single-output blocks. A model is a
**named module graph**, and KonfAI lets you attach losses, metrics, and exported
datasets to specific named outputs.

## Networks

The core abstractions live in `konfai.network.network`:

- `Network`
- `ModelLoader`
- `OptimizerLoader`
- `TargetCriterionsLoader`
- `Measure`

The selected model class is configured under `Model.classpath`, then further
configured under a section named after that class.

Example:

```yaml
Model:
  classpath: UNet.yml
  UNet:
    parameters:
      dim: 2
      nb_class: 41
```

## Addressing outputs

Losses and metrics are attached through `outputs_criterions`. Keys in this
mapping correspond to named modules or outputs in the model graph.

Example from the segmentation baseline:

```yaml
outputs_criterions:
  UNetBlock_0:Head:Conv:
    targets_criterions:
      SEG:
        criterions_loader:
          torch:nn:CrossEntropyLoss:
            is_loss: true
```

```{note}
An `outputs_criterions` or `outputs_dataset` key must match a module's dotted
path **exactly** — the `:` separators between graph levels are load-bearing. A
key that does not match any module path raises a configuration error at
runtime.
```

## Targets and metrics

For each output group you can define one or more target groups, then one or more
criteria for each target:

```yaml
outputs_criterions:
  Head:Tanh:
    targets_criterions:
      CT:
        criterions_loader:
          MAE:
            is_loss: true
```

This structure lets you express:

- multiple heads
- multiple targets per head
- multiple losses or metrics per target
- independent scheduler weights per criterion

## Dataset patching vs model patching

KonfAI supports patching at two different levels:

- **dataset patching** with `Dataset.Patch`
- **model patching** with `Model.<Class>.ModelPatch`

Dataset patching controls what reaches the model. Model patching controls how a
network internally re-processes those tensors.

The `examples/Synthesis` GAN variant is the clearest example:

- `Dataset.Patch` provides a 3D chunk to the whole GAN
- `Model.Gan.UNetpp5.ModelPatch` reprocesses the chunk slice-wise inside the generator

## `;accu;` outputs

The `;accu;` marker appears in some advanced workflows, especially when model
patching is enabled. Its semantics are **inferred from the shipped examples and
the network patch/accumulation logic**.

In practice it refers to patch-wise outputs **before final re-assembly**.

This matters in the synthesis GAN example:

- `Generator_A_to_B:;accu;Head:Tanh` is used for patch-wise reconstruction loss
- `Discriminator_pB:Head:Conv` is used after the generator output has been re-assembled

## Prediction outputs

Inference uses a separate `outputs_dataset` mapping to decide what should be
written to disk.

Example:

```yaml
outputs_dataset:
  Head:Tanh:
    OutputDataset:
      name_class: OutSameAsGroupDataset
      group: sCT
      reduction: Mean
```

This lets you control:

- which model output is exported
- how multiple predictions are reduced
- what final transforms are applied before writing files

## Next steps

- {doc}`yaml-model-builder` — define the same named graphs entirely in YAML.
- {doc}`../usage/custom-models` — write your own `Network` subclass and wire it with `add_module`.
- {doc}`../config_guide/prediction` — the full `outputs_dataset` reference for inference runs.
