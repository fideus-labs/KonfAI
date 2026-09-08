# Models

A KonfAI model is a **named module graph**: a subclass of `konfai.network.Network`
whose modules are wired by `add_module`, or the same graph written as a `.yml`.
Every module has a dotted path, and that path is what a config writes to attach
a loss, a metric or an exported prediction to it. This page is the whole model
story: how outputs are addressed, which networks ship (as Python classes and as
a declarative catalog), how to write a graph in YAML, and how to seed it from
weights trained elsewhere. Reference a built-in by `classpath`
(`classpath: segmentation.UNet.UNet`, or `classpath: default|UNet.yml`).

**"YAML-buildable" column.** A model is buildable from a `.yml` (via the safe
[YAML model builder](#declarative-yaml-model-graphs)) only if it is a pure graph
of registry node types. Models whose graph contains a leaf the registry does
not know (diffusion samplers, StyleGAN, ConvNeXt, VoxelMorph's warping
components) are written as Python classes instead. (Most of them do not
override `forward()`; what keeps them out of YAML is the node types they
compose, not a custom forward pass.) The registry is deliberately small.

## Model graph and output naming

This section explains the naming scheme behind every `outputs_criterions` and
`outputs_dataset` key: how KonfAI addresses individual modules inside a model
graph. Read it before attaching a loss, metric, or exported prediction to a
model output.

KonfAI models are not treated as opaque single-output blocks. A model is a
**named module graph**, and KonfAI lets you attach losses, metrics, and exported
datasets to specific named outputs.

```{mermaid}
flowchart TB
    IN[branch '0', the input]:::branch --> E[UNetBlock_0]:::mod
    E --> H[Head]:::mod
    H --> CV[Conv]:::mod
    CV --> SM[Softmax]:::mod
    SM --> OUT[out_branch]:::branch
    CV -. "UNetBlock_0:Head:Conv" .-> L1[[CrossEntropyLoss]]:::crit
    SM -. "UNetBlock_0:Head:Softmax" .-> L2[[Dice]]:::crit

```

Every module has a dotted path, and that path is the key you write in the config.
Two losses can read two different outputs of the same head, which is what the
dashed arrows above are: `Conv` carries the raw logits CrossEntropy expects,
`Softmax` the probabilities Dice expects.

### Networks

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

### Addressing outputs

Losses and metrics are attached through `outputs_criterions`. Keys in this
mapping correspond to named modules or outputs in the model graph.

Example from the segmentation baseline, which attaches two losses to two different
outputs of the same head: cross entropy wants logits, Dice wants probabilities:

```yaml
outputs_criterions:
  UNetBlock_0:Head:Conv:          # raw logits
    targets_criterions:
      SEG:
        criterions_loader:
          CrossEntropyLoss:
            is_loss: true
  UNetBlock_0:Head:Softmax:       # class probabilities
    targets_criterions:
      SEG:
        criterions_loader:
          Dice:
            is_loss: true
            labels: None          # every label present in the patch
```

A bare name (`CrossEntropyLoss`, `Dice`) resolves inside the criterion package;
`torch:nn:CrossEntropyLoss` would import the torch class directly instead.

An `outputs_criterions` or `outputs_dataset` key must match a module's dotted
path **exactly**: the `:` separators between graph levels are load-bearing. A
key that does not match any module path raises a configuration error at
runtime.

### Targets and metrics

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

### Dataset patching vs model patching

KonfAI supports patching at two different levels:

- **dataset patching** with `Dataset.Patch`
- **model patching** with `Model.<Class>.ModelPatch`

Dataset patching controls what reaches the model. Model patching controls how a
network internally re-processes those tensors.

The `examples/Synthesis` GAN variant is the clearest example:

- `Dataset.Patch` provides a 3D chunk to the whole GAN
- `Model.Gan.UNetpp5.ModelPatch` reprocesses the chunk slice-wise inside the generator

### `;accu;` outputs

The `;accu;` marker appears in some advanced workflows, especially when model
patching is enabled. Its semantics are **inferred from the shipped examples and
the network patch/accumulation logic**.

In practice it refers to patch-wise outputs **before final re-assembly**.

This matters in the synthesis GAN example:

- `Generator_A_to_B:;accu;Head:Tanh` is used for patch-wise reconstruction loss
- `Discriminator_pB:Head:Conv` is used after the generator output has been re-assembled

### Prediction outputs

Inference uses a separate `outputs_dataset` mapping to decide what should be
written to disk.

Example:

```yaml
outputs_dataset:
  Head:Tanh:
    OutputDataset:
      name_class: OutputDataset
      group: sCT
      reduction: Mean
```

This lets you control:

- which model output is exported
- how multiple predictions are reduced
- what final transforms are applied before writing files

## Python models

The built-in `Network` subclasses under `konfai/models/python/`, by family.

### Segmentation: `konfai.models.python.segmentation`

`PlainConvUNet` is the parametric nnU-Net backbone (n_stages / features_per_stage /
strides / n_conv_per_stage as real arguments), weight-exact to
`dynamic_network_architectures.PlainConvUNet` for any topology: a real nnU-Net /
TotalSegmentator / MRSeg checkpoint loads into it through the pretrained bridge. Every
decoder resolution has a deep-supervision head as a named output.

`SMP` wraps any `segmentation_models_pytorch` architecture/encoder pair (Unet,
UnetPlusPlus, FPN, DeepLabV3Plus, … × resnet/efficientnet/timm encoders), with
optional ImageNet encoder weights (`encoder_weights: imagenet`) that survive
training start. 2D-only (SMP's encoder zoo is 2D); use slice-wise patches or
2.5D channels on volumes. Requires `pip install konfai[smp]`.

| Model | Classpath | Purpose | Key args (defaults) | Dims | YAML-buildable |
| --- | --- | --- | --- | --- | --- |
| `UNet` | `segmentation.UNet.UNet` | Classic encoder–decoder U-Net; optional attention gates and deep-supervision heads. | `channels=[1,64,128,256,512,1024]`, `nb_class=2`, `dim=3`, `block_config=BlockConfig()`, `nb_conv_per_stage=2`, `downsample_mode="MAXPOOL"`, `upsample_mode="CONV_TRANSPOSE"`, `attention=False`, `block_type="Conv"` | 2D / 3D | Yes |
| `NestedUNet` | `segmentation.NestedUNet.NestedUNet` | UNet++ with dense nested skips and per-level deep-supervision heads. | as `UNet` + `activation="Softmax"` | 2D / 3D | Yes |
| `UNetPlusPlus` | `segmentation.unetplusplus.UNetPlusPlus` | Parametric UNet++ on a **pretrained ResNet backbone**: **weight-exact vs `smp.UnetPlusPlus`** (resnet18/34). Use this to load an smp / ImpactSynth checkpoint into an addressable KonfAI graph. | `dim=2`, `in_channels=3`, `classes=1`, `encoder_name="resnet34"`, `decoder_channels=[256,128,64,32,16]`, `activation=None` (`"tanh"` for sCT) | 2D or 3D | Yes (params) |
| `ResidualEncoderUNet` | `segmentation.residualencoderunet.ResidualEncoderUNet` | Parametric nnU-Net **residual-encoder** U-Net for any topology: **weight-exact vs `dynamic_network_architectures.ResidualEncoderUNet`**. Loads a real nnU-Net ResEnc / ImpactSeg checkpoint via the bridge. | `dim=3`, `in_channels=1`, `n_stages=6`, `features_per_stage=[32,64,128,256,320,320]`, `strides=[1,2,2,2,2,2]`, `n_blocks_per_stage=[1,3,4,6,6,6]`, `num_classes=2`, `deep_supervision=True` | 2D / 3D | Yes (params) |

Two UNet++ flavours: `NestedUNet` (academic, plain-conv encoder trained from scratch) and
`UNetPlusPlus` (the **smp-faithful** UNet++ with a real pretrained ResNet backbone: the one
that loads `smp.UnetPlusPlus` checkpoints). Pick `UNetPlusPlus` when you need smp
weight-compatibility (e.g. the ImpactSynth app).

The `Model:UNetpp5` used in the `Synthesis` example is a **local** class in
`examples/Synthesis/Model.py` wrapping `segmentation_models_pytorch`; the built-in
`UNetPlusPlus` above is the maintained, smp-weight-exact equivalent.

### Classification: `konfai.models.python.classification`

| Model | Classpath | Purpose | Key args (defaults) | Dims | YAML-buildable |
| --- | --- | --- | --- | --- | --- |
| `ResNet` | `classification.resnet.ResNet` | ResNet-18/34/50/101/152 family with torchvision-compatible weight aliases. | `dim=3`, `in_channels=1`, `depths=[2,2,2,2]`, `widths=[64,64,128,256,512]`, `num_classes=10`, `use_bottleneck=False` | 2D / 3D | Yes |
| `ConvNeXt` | `classification.convNeXt.ConvNeXt` | ConvNeXt (tiny→xlarge presets) with a multi-head classifier (`num_classes` is a list). | `dim=3`, `in_channels=1`, `depths=[3,3,27,3]`, `widths=[128,256,512,1024]`, `drop_p=0.1`, `num_classes=[4,7]` | 2D or 3D (`dim` parameterised) | No (non-registry leaves) |

### Generation: `konfai.models.python.generation`

| Model | Classpath | Purpose | Dims | YAML-buildable |
| --- | --- | --- | --- | --- |
| `VAE` | `generation.vae.VAE` | Convolutional auto-encoder. **Deterministic**: despite the name there is no latent sampling. | 2D / 3D | Yes |
| `LinearVAE` | `generation.vae.LinearVAE` | Fully-connected variational AE (`LatentDistribution` reparam bottleneck). Pairs with the `KLDivergence` loss. | 1D (flat vectors) | No (`LatentDistribution`) |
| `Generator` / `Discriminator` / `Gan` | `generation.gan.*` | PatchGAN discriminator + ResNet-autoencoder generator + composite adversarial graph. | 2D / 3D | No |
| `DiffusionGan`, `DiffusionGanV2`, `DiffusionCycleGan`, `CycleGan*` | `generation.diffusionGan.*` | Adversarial + diffusion + CycleGAN family. | 2D / 3D | No |
| `cStyleGan.Generator` | `generation.cStyleGan.Generator` | Conditional StyleGAN-style generator with weight-modulated convs. | 2D / 3D | No |

### Registration: `konfai.models.python.registration`

| Model | Classpath | Purpose | Dims | YAML-buildable |
| --- | --- | --- | --- | --- |
| `VoxelMorph` | `registration.registration.VoxelMorph` | Learning-based deformable/rigid registration (U-Net flow field + spatial-transformer warp + scaling-and-squaring integration). Pass `dim: 2`. | 2D | No |

### Representation: `konfai.models.python.representation`

| Model | Classpath | Purpose | Dims | YAML-buildable |
| --- | --- | --- | --- | --- |
| `Representation` | `representation.representation.Representation` | Self-supervised / triplet-style representation learner: a frozen conv encoder + trainable linear projection head. | 3D | No |

### Features: `konfai.models.python.features`

| Model | Classpath | Purpose | Dims |
| --- | --- | --- | --- |
| `MIND` | `features.mind.MIND` | Modality-independent neighbourhood-descriptor feature extractor (used as a registration/synthesis feature space). | 2D/3D |

## Declarative YAML model graphs

The YAML model builder describes a complete
network as a `.yml` file instead of a Python class. Use it when you want a
model that lives entirely in configuration, with the same named outputs and
routing as code-defined models.

`konfai.utils.model_builder` builds a full `konfai.network.network.Network`.
Every YAML entry is installed through `ModuleArgsDict.add_module`, so YAML
models support the same named outputs, branch routing, aliases, checkpoint
metadata, optimizer configuration, and loss attachment as Python models.

The segmentation example is defined in `examples/Segmentation/UNet.yml`; its
training and prediction configs load it with `classpath: UNet.yml`. The older
Python `konfai.models.python.segmentation.UNet` remains available for compatibility.

### Document structure

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
**Relative paths are resolved next to the active `KONFAI_config_file`**: the
model `.yml` is looked up relative to the config file that references it, not
the current working directory.

### Routing and nested graphs

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

### Parameters and safe objects

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

### Registry

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
  `Linear`, `GELU`): enough to express a ViT encoder.

Call `list_registered_modules()` for the authoritative, up-to-date list. Applications
may add a trusted `torch.nn.Module` subclass with `register_module(name, cls)`.
Duplicate names and non-module classes raise `ConfigError`.

### Shipped model catalog

`konfai/models/` is split by form: `python/` holds the builtin Python model classes
(referenced as `classpath: segmentation.UNet.UNet`), `yaml/` the declarative catalog.
KonfAI ships a catalog of common medical-imaging architectures as declarative YAML
under `konfai/models/yaml/`. Reference one from any config with a `default|` marker: the declarative counterpart of a Python model classpath:

```yaml
Model:
  classpath: default|AttentionUNet.yml
```

Every catalog entry is built from the curated registry (no code execution) and is
locked by a test at the strongest available level: weight-exact graph equivalence
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
| `ViT` | structural + encoder token-features allclose vs MONAI |: (encoder maths verified) |
| `AttentionUNet`, `UNETR` | structural-strict (graph differs from MONAI, documented) |: |
| `ResidualEncoderUNet` *(parametric)* | weight-exact vs nnU-Net `dynamic_network_architectures.ResidualEncoderUNet` (`deep_supervision` toggle) | nnU-Net ResEnc / ImpactSeg checkpoints (via the bridge) |
| `UNetPlusPlus` *(parametric)* | weight-exact vs `segmentation_models_pytorch.UnetPlusPlus` (ResNet-18/34 encoder, `activation` configurable) | smp / ImpactSynth checkpoints (via the bridge) |

`PlainConvUNet`, `ResidualEncoderUNet` and `UNetPlusPlus` ship **both** ways: as a
fixed-topology entry in the catalog above, and as a **parametric** Python class
(`konfai/models/python/segmentation/`). Reference the class when you want one model to
cover any depth / width / class count and declare the architecture inline: e.g.

```yaml
Model:
  classpath: segmentation.residualencoderunet.ResidualEncoderUNet
  ResidualEncoderUNet:
    dim: 2
    in_channels: 5
    n_stages: 6
    features_per_stage: [24, 48, 96, 192, 256, 256]
    strides: [1, 2, 2, 2, 2, 2]
    n_blocks_per_stage: [1, 2, 2, 3, 3, 3]
    num_classes: 12
    deep_supervision: false
```

`VGG16` is the feature-extractor entry: it exposes five named multi-layer outputs
(`Block_0:Out` … `Block_4:Out`, channels 64/128/256/512/512: the torchvision
`features` slices `[0:4]/[4:9]/[9:16]/[16:23]/[23:30]`) so a perceptual / feature /
IMPACT-style loss can be attached to any of them through `outputs_criterions`.

The MCP server lists the catalog via `list_components(kind="model")` alongside the
Python model classes.

### Which form should I use?

There are three ways to put a common architecture into a KonfAI config, and they are
not redundant: pick by what you need:

| You want to… | Use | Why |
|---|---|---|
| Train/run the vanilla model as-is, one output, one loss | `classpath: monai.networks.nets:SegResNet` (or any installed class) | KonfAI wraps any `nn.Module` in `MinimalModel` automatically: no rebuild needed. Simplest path. |
| Supervise **internal** layers (deep supervision, feature/perceptual losses), edit the architecture without code, or share it safely | `classpath: default\|SegResNet.yml` | The YAML builds a KonfAI `Network` whose every submodule is addressable in `outputs_criterions`, editable in YAML, and safe by construction (registry-only, no imported code). |
| Do the above **and** start from someone's pretrained weights | `default\|<Name>.yml` + the pretrained bridge (below) | You get the reference's trained weights inside the addressable KonfAI graph. |

An imported `nn.Module` is a black box: only its final output is visible to KonfAI's
loss/evaluation machinery. The YAML form is what unlocks per-node supervision, that is
the reason to rebuild an architecture rather than import it.

### Configuration example

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

## Building blocks (`konfai.network.blocks`)

If you author your own model (as a Python `Network` or a YAML graph) these
reusable pieces are the vocabulary:

- **Conv graphs:** `ConvBlock` (`[Conv → Norm → Activation]×N`), `ResBlock`
  (residual with projected skip), `Attention` (Attention-U-Net gate),
  `LatentDistribution` (VAE reparameterisation, exposes `mu`/`log_std`/`z`).
- **`BlockConfig`**: one conv stage: `kernel_size=3, stride=1, padding=1,
  bias=True, activation="ReLU", norm_mode="NONE"`. `activation` accepts a name,
  a `";"`-separated spec (`"LeakyReLU;0.2;True"`), a callable, or `None`.
- **Enums:** `NormMode` (`NONE/BATCH/INSTANCE/GROUP/LAYER/SYNCBATCH/INSTANCE_AFFINE`),
  `UpsampleMode` (`CONV_TRANSPOSE/UPSAMPLE`), `DownsampleMode`
  (`MAXPOOL/AVGPOOL/CONV_STRIDE`).
- **Tensor ops** (leaf modules): `Add`, `Multiply`, `Concat`, `Detach`,
  `ArgMax`, `Select`, `View`, `Permute`, `NormalNoise`, and more.

## Start from MONAI, torchvision or nnU-Net weights (`pretrained_from`)

A fresh TRAIN can seed its model from a checkpoint trained in another
framework, from config alone. `Model.pretrained_from` builds the reference
network, loads the checkpoint into it, and transfers the weights into the
KonfAI graph by forward-execution order (no key map): the bridge fills **every**
target tensor or raises, so a partial transfer is never reported as success.

```yaml
Trainer:
  Model:
    classpath: default|PlainConvUNet.yml
    pretrained_from:
      checkpoint: ./nnunet_fold0.pt        # raw state_dict, or a dict with a 'state_dict' entry;
                                           # an https:// URL is accepted (weights-only load)
      builder: monai.networks.nets:UNet    # classpath of the reference class
      args: {spatial_dims: 3, in_channels: 1, out_channels: 2, channels: [32, 64], strides: [2]}
      input_shape: [96, 96, 96]            # optional; else derived from the model's own
                                           # patch size or downsampling factors
```

The seed runs only on a fresh TRAIN: a RESUME or PREDICTION checkpoint always
wins, and PREDICTION never builds (or needs) the reference. A multi-input graph
or a free-axis patch size cannot derive a synthetic input on its own:
`input_shape` is the escape hatch, and the failure is a `ConfigError` naming
`Model.pretrained_from`. The transfer itself is
`konfai.utils.pretrained.transfer_weights_by_execution_order` (see the API
reference).

### Loading pretrained weights

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

## Fine-tuning across a different head (`allow_head_resize`)

By default a checkpoint load **refuses shape mismatches**: the strict load
raises, naming the tensor and both shapes. To fine-tune across a head whose
shape changed (a different label count, say), opt in with
`Model.allow_head_resize: true`: the load then warm-starts the overlapping
slice of each mismatched tensor and logs a warning per resized tensor. The
loader propagates the opt-in to every nested network and can only enable it,
never disable a model class's own constructor opt-in.

```yaml
Trainer:
  Model:
    classpath: segmentation.UNet.UNet
    allow_head_resize: true
```

## Next steps

- {doc}`losses-metrics`: attach losses and metrics to a model's named outputs.
- {doc}`../../usage/custom-models`: subclass `Network` yourself.
- {doc}`../../usage/adopting-konfai`: the routes into KonfAI from an existing
  PyTorch, MONAI or nnU-Net model.
- {doc}`../../examples/segmentation`: a complete training run driven by `UNet.yml`.
