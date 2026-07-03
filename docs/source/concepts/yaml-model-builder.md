# Declarative YAML model graphs

`konfai.utils.model_builder` builds a full `konfai.network.network.Network`.
Every YAML entry is installed through `ModuleArgsDict.add_module`, so YAML
models support the same named outputs, branch routing, aliases, checkpoint
metadata, optimizer configuration, and loss attachment as Python models.

The segmentation example is defined in `examples/Segmentation/UNet.yml`; its
training and prediction configs load it with `classpath: UNet.yml`. The older
Python `konfai.models.segmentation.UNet` remains available for compatibility.

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
Relative paths are resolved next to the active `KONFAI_config_file`.

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

Built-ins include dimension-aware `Conv`, `ConvTranspose`, `MaxPool`, and
`AvgPool` factories; explicit `Conv1d`/`Conv2d`/`Conv3d`; `ConvBlock`,
`ResBlock`, `Concat`, `Softmax`, `ArgMax`, and `Identity`.

Call `list_registered_modules()` to inspect the active registry. Applications
may add a trusted `torch.nn.Module` subclass with `register_module(name, cls)`.
Duplicate names and non-module classes raise `ConfigError`.

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

## See also

- {doc}`model-graph`
- {doc}`datasets`
- {doc}`../examples/segmentation`
