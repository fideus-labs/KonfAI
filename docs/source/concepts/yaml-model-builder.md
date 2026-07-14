# Declarative YAML model graphs

This page documents the YAML model builder — how to describe a complete
network as a `.yml` file instead of a Python class. Read it when you want a
model that lives entirely in configuration, with the same named outputs and
routing as code-defined models.

`konfai.utils.model_builder` builds a full `konfai.network.network.Network`.
Every YAML entry is installed through `ModuleArgsDict.add_module`, so YAML
models support the same named outputs, branch routing, aliases, checkpoint
metadata, optimizer configuration, and loss attachment as Python models.

The segmentation example is defined in `examples/Segmentation/UNet.yml`; its
training and prediction configs load it with `classpath: UNet.yml`. The older
Python `konfai.models.python.segmentation.UNet` remains available for compatibility.

## Document structure

```yaml
name: RoutedHead
parameters:
  dim: 2
  in_channels: 16
  classes: 3
network:
  dim: ${dim}
  in_channels: ${in_channels}
modules:
  - name: Conv
    type: Conv
    args:
      dim: ${dim}
      in_channels: ${in_channels}
      out_channels: ${classes}
      kernel_size: 1
  - name: Softmax
    type: Softmax
    args: {dim: 1}
  - name: Argmax
    type: ArgMax
    args: {dim: 1}
```

`build_model_from_yaml(yaml_path="model.yml")` returns a `YamlNetwork`, not a
`torch.nn.Sequential`. `ModelLoader` also accepts `.yml` and `.yaml` paths.
**Relative paths are resolved next to the active `KONFAI_config_file`** — the
model `.yml` is looked up relative to the config file that references it, not
the current working directory.

## Routing and nested graphs

Module entries accept the routing fields from `add_module`:

- `in_branch` and `out_branch`
- `alias`
- `pretrained`
- `requires_grad`
- `training`

A nested `modules` list creates a `ModuleArgsDict` subgraph:

```yaml
modules:
  - name: Encoder
    modules:
      - name: Conv
        type: Conv2d
        args: {in_channels: 1, out_channels: 8, kernel_size: 3, padding: 1}
  - name: Preserve
    type: Identity
    out_branch: [1]
  - name: Join
    type: Concat
    in_branch: [0, 1]
```

Module paths remain stable (`Encoder:Conv`, `Join`, and so on) for
`outputs_criterions` and `outputs_dataset`.

## Parameters and safe objects

An exact `${path}` value references `parameters`; list indices use dotted
numbers such as `${channels.2}`. Runtime configuration can override the entire
`parameters` mapping under the model section.

Some KonfAI blocks need configuration objects. They are constructed through a
separate safe object registry:

```yaml
parameters:
  block_configs:
    - $object: BlockConfig
      args:
        kernel_size: 3
        padding: 1
        activation: ReLU
        norm_mode: NONE
modules:
  - name: Block
    type: ConvBlock
    args:
      in_channels: 1
      out_channels: 32
      dim: 2
      block_configs: ${block_configs}
```

`$multiply` provides safe numeric multiplication for derived channel counts.
No YAML value is passed to `eval` or used as an import path.

## Registry

Built-ins, grouped:

- **Dimension-aware factories** (pick the 1-D/2-D/3-D variant from `dim`):
  `Conv`, `ConvTranspose`, `MaxPool`, `AvgPool`, `AdaptiveAvgPool`, `BatchNorm`,
  `InstanceNorm`; explicit `Conv1d`/`Conv2d`/`Conv3d` and `Dropout`/`Dropout1d`/`2d`/`3d`.
- **Normalization / regularization:** `GroupNorm`, `LayerNorm`, `Dropout`.
- **Activations:** `ReLU`, `LeakyReLU`, `PReLU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`.
- **Linear / shape:** `Linear`, `Flatten`, `Upsample`, `Identity`, `Permute`, `View`,
  `Select`, `Unsqueeze`, `ArgMax`.
- **Routing leaves:** `Concat`, `Add`, `Multiply`.
- **Composite blocks:** `ConvBlock`, `ResBlock`, `Attention`.
- **Transformer:** `MultiHeadSelfAttention`, `PositionalEmbedding` (with `LayerNorm`,
  `Linear`, `GELU`) — enough to express a ViT encoder.

Call `list_registered_modules()` for the authoritative, up-to-date list. Applications
may add a trusted `torch.nn.Module` subclass with `register_module(name, cls)`.
Duplicate names and non-module classes raise `ConfigError`.

## Shipped model catalog

`konfai/models/` is split by form: `python/` holds the builtin Python model classes
(referenced as `classpath: segmentation.UNet.UNet`), `yaml/` the declarative catalog.
KonfAI ships a catalog of common medical-imaging architectures as declarative YAML
under `konfai/models/yaml/`. Reference one from any config with a `default|` marker —
the declarative counterpart of a Python model classpath:

```yaml
Model:
  classpath: default|AttentionUNet.yml
```

Every catalog entry is built from the curated registry (no code execution) and is
locked by a test at the strongest available level — weight-exact graph equivalence
against a reference implementation where one exists, otherwise a structural check
(builds, forward on 2-D and 3-D inputs, correct output shape, deep-supervision heads)
with any divergence from the reference documented in the file header:

| Entry | Validation | Loads pretrained from |
|---|---|---|
| `UNet`, `NestedUNet`, `ResNet` | weight-exact vs their KonfAI Python classes | KonfAI checkpoints |
| `SegResNet`, `VNet`, `DynUNet` | weight-exact vs MONAI | MONAI checkpoints (via the bridge) |
| `ResNet18` | weight-exact vs torchvision ResNet-18 | torchvision ImageNet (via the bridge) |
| `PlainConvUNet` | weight-exact vs nnU-Net `dynamic_network_architectures.PlainConvUNet` | nnU-Net / TotalSegmentator / MRSeg checkpoints (via the bridge) |
| `VGG16` | weight-exact vs torchvision (all 5 feature maps exact) | torchvision ImageNet (via the bridge) |
| `ViT` | structural + encoder token-features allclose vs MONAI | — (encoder maths verified) |
| `AttentionUNet`, `UNETR` | structural-strict (graph differs from MONAI, documented) | — |

`VGG16` is the feature-extractor entry: it exposes five named multi-layer outputs
(`Block_0:Out` … `Block_4:Out`, channels 64/128/256/512/512 — the torchvision
`features` slices `[0:4]/[4:9]/[9:16]/[16:23]/[23:30]`) so a perceptual / feature /
IMPACT-style loss can be attached to any of them through `outputs_criterions`.

The MCP server lists the catalog via `list_components(kind="model")` alongside the
Python model classes.

## Which form should I use?

There are three ways to put a common architecture into a KonfAI config, and they are
not redundant — pick by what you need:

| You want to… | Use | Why |
|---|---|---|
| Train/run the vanilla model as-is, one output, one loss | `classpath: monai.networks.nets:SegResNet` (or any installed class) | KonfAI wraps any `nn.Module` in `MinimalModel` automatically — no rebuild needed. Simplest path. |
| Supervise **internal** layers (deep supervision, feature/perceptual losses), edit the architecture without code, or share it safely | `classpath: default|SegResNet.yml` | The YAML builds a KonfAI `Network` whose every submodule is addressable in `outputs_criterions`, editable in YAML, and safe by construction (registry-only, no imported code). |
| Do the above **and** start from someone's pretrained weights | `default|<Name>.yml` + the pretrained bridge (below) | You get the reference's trained weights inside the addressable KonfAI graph. |

An imported `nn.Module` is a black box: only its final output is visible to KonfAI's
loss/evaluation machinery. The YAML form is what unlocks per-node supervision — that is
the reason to rebuild an architecture rather than import it.

## Loading pretrained weights

A catalog entry that is weight-exact to a reference (e.g. `SegResNet.yml` ↔ MONAI
`SegResNet`) can be loaded from that reference's checkpoint even though the two use
different module names, via `konfai.utils.pretrained.transfer_weights_by_execution_order`.
It pairs the two graphs' weighted leaves in forward-execution order and copies them with
a shape check, so no hand-written key map is needed:

```python
from monai.networks.nets import SegResNet
from konfai.utils.model_builder import build_model_from_yaml
from konfai.utils.pretrained import transfer_weights_by_execution_order

reference = SegResNet(spatial_dims=3, init_filters=8, in_channels=1, out_channels=2,
                      blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1))
reference.load_state_dict(torch.load("segresnet_pretrained.pt"))  # your trained checkpoint

net = build_model_from_yaml(yaml_path="konfai/models/yaml/SegResNet.yml",
                            parameters={"dim": 3, "upsample_mode": "trilinear", "nb_class": 2})
example = torch.randn(1, 1, 16, 16, 16)
transfer_weights_by_execution_order(
    target=net, source=reference,
    target_forward=lambda: list(net.named_forward(example)),
    source_forward=lambda: reference(example),
)
```

The transfer is strict: if the two graphs are not weight-exact (different leaf count or a
mismatched shape) it raises `ConfigError` rather than silently mis-loading a network.

## Configuration example

```yaml
Trainer:
  Model:
    classpath: UNet.yml
    UNet:
      parameters:
        dim: 2
        channels: [1, 32, 64, 128, 256]
        nb_class: 41
      optimizer:
        name: AdamW
        lr: 0.001
      outputs_criterions:
        UNetBlock_0:Head:Conv:
          targets_criterions: {}
```

See `examples/Segmentation/UNet.yml` for a complete routed encoder/decoder with
skip connections and nested heads.

## Next steps

- {doc}`model-graph` — how named module paths are addressed by `outputs_criterions` and `outputs_dataset`.
- {doc}`../examples/segmentation` — a complete training run driven by `UNet.yml`.
