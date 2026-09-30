# Models

A KonfAI model is a graph of named modules: a `konfai.network.network.Network` subclass built with
`add_module`, or the same graph written as a `.yml`. Each module has a path, and a config attaches losses,
metrics and saved outputs to those paths. Pick a model with `classpath`: `segmentation.UNet.UNet` for a
Python class, `default|UNet.yml` for the YAML catalog, `UNet.yml` for a file next to the config.

## Model graph and output naming

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

A module's path joins the names of its parents with `:`. The segmentation example attaches two losses to
two outputs of the same head: cross entropy to the logits, Dice to the probabilities.

```yaml
outputs_criterions:
  UNetBlock_0:Head:Conv:          # logits
    targets_criterions:
      SEG:
        criterions_loader:
          CrossEntropyLoss:
            is_loss: true
  UNetBlock_0:Head:Softmax:       # probabilities
    targets_criterions:
      SEG:
        criterions_loader:
          Dice:
            is_loss: true
            labels: None          # every label present in the patch
```

The key must match a path exactly; a key that matches none is an error. Each output can have several
targets, and each target several criteria.

A target can also be another output of the model. `Representation` trains an anchor embedding against a
positive and a negative one:

```yaml
outputs_criterions:
  Model:Anchor:
    targets_criterions:
      Model:Positive;Model:Negative:
        criterions_loader:
          torch:nn:TripletMarginLoss:
            margin: 1.0
```

A target output receives no gradient: it flows only through the output the criterion is attached to.

Prediction chooses what to save with `outputs_dataset`, keyed the same way:

```yaml
outputs_dataset:
  Head:Tanh:
    OutputDataset:
      name_class: OutputDataset
      group: sCT
      reduction: Mean
```

### Patching inside the model

`Dataset.Patch` decides what reaches the model; `ModelPatch`, under the model's class, cuts again inside it.
In the `examples/Synthesis` GAN, the dataset hands 3-D blocks to the GAN and the generator works on 2-D
slices. With a `ModelPatch`, an output for one patch carries `;accu;` in its path, before the patches are
put back together: `Generator_A_to_B:;accu;Head:Tanh` is the generator's output patch by patch,
`Discriminator_pB:Head:Conv` a module reading the reassembled volume.

## Python models

Classpaths are relative to `konfai.models.python`. The last column says whether the model also exists in the
YAML catalog.

### Segmentation

| Model | Classpath | Purpose | Key arguments (defaults) | Dims | YAML |
| --- | --- | --- | --- | --- | --- |
| `UNet` | `segmentation.UNet.UNet` | U-Net, with optional attention and deep supervision. | `channels=[1,64,128,256,512,1024]`, `nb_class=2`, `dim=3`, `attention=False` | 2D/3D | yes |
| `NestedUNet` | `segmentation.NestedUNet.NestedUNet` | UNet++ with a head per level. | as `UNet` | 2D/3D | yes |
| `UNetPlusPlus` | `segmentation.unetplusplus.UNetPlusPlus` | UNet++ on a pretrained ResNet, weight-exact to `smp.UnetPlusPlus` (loads ImpactSynth checkpoints). | `dim=2`, `in_channels=3`, `classes=1`, `encoder_name="resnet34"`, `activation=None` | 2D/3D | yes |
| `ResidualEncoderUNet` | `segmentation.residualencoderunet.ResidualEncoderUNet` | nnU-Net residual-encoder U-Net, weight-exact (loads ImpactSeg checkpoints). | `dim=3`, `n_stages=6`, `features_per_stage`, `strides`, `n_blocks_per_stage`, `num_classes=2`, `deep_supervision=True` | 2D/3D | yes |
| `PlainConvUNet` | `segmentation.plainconvunet.PlainConvUNet` | nnU-Net plain U-Net, weight-exact (loads TotalSegmentator and MRSegmentator checkpoints). | `n_stages`, `features_per_stage`, `strides`, `n_conv_per_stage` | 2D/3D | yes |
| `SMP` | `segmentation.smp.SMP` | Any `segmentation_models_pytorch` architecture and encoder, with optional ImageNet weights (`konfai[smp]`). | `arch`, `encoder_name`, `encoder_weights` | 2D | no |

### Other families

| Model | Classpath | Purpose | Dims | YAML |
| --- | --- | --- | --- | --- |
| `ResNet` | `classification.resnet.ResNet` | ResNet-18 to 152, torchvision-compatible. | 2D/3D | yes |
| `ConvNeXt` | `classification.convNeXt.ConvNeXt` | ConvNeXt with one classifier per task. | 2D/3D | no |
| `VAE` | `generation.vae.VAE` | Convolutional auto-encoder (no latent sampling). | 2D/3D | yes |
| `LinearVAE` | `generation.vae.LinearVAE` | Variational auto-encoder on vectors; pairs with `KLDivergence`. | 1D | no |
| `Gan`, `Generator`, `Discriminator` | `generation.gan.*` | GAN with a PatchGAN discriminator. | 2D/3D | no |
| `DiffusionGan`, `CycleGan*`, … | `generation.diffusionGan.*` | Adversarial, diffusion and CycleGAN models. | 2D/3D | no |
| `VoxelMorph` | `registration.registration.VoxelMorph` | Learned deformable registration. | 2D/3D | no |
| `Representation` | `representation.representation.Representation` | Triplet representation learning (above). | 3D | no |
| `MIND` | `features.mind.MIND` | MIND feature descriptor. | 2D/3D | no |

At inference, a GAN runs only its generator. `cStyleGan.Generator` builds but cannot run.

## Declarative YAML model graphs

A `.yml` can describe a whole network. The result is a real `Network`: named outputs, routing, aliases,
optimizer and losses work exactly as for a Python model.

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

- `${name}` reads a parameter (`${channels.2}` for a list item). The config can override `parameters`
  under the model's section.
- A module takes the routing fields of `add_module`: `in_branch`, `out_branch`, `alias`, `pretrained`,
  `requires_grad`, `training`. A nested `modules` list makes a sub-graph (`Encoder:Conv`).
- `$object: BlockConfig` builds a configuration object, `$multiply` multiplies numbers.
- A relative `.yml` path is read next to the config that names it.
- Only registered module types are allowed, and nothing is evaluated or imported: a `.yml` from someone
  else is safe to build. `list_registered_modules()` lists the types (convolutions, pooling, normalization,
  activations, `Linear`, `Concat`, `Add`, `ConvBlock`, `ResBlock`, `Attention`, `MultiHeadSelfAttention`,
  …); `register_module(name, cls)` adds a trusted one.

`examples/Segmentation/UNet.yml` is a complete example.

### The catalog

`default|<Name>.yml` picks a model from KonfAI's catalog (`konfai/models/yaml/`):

```yaml
Model:
  classpath: default|AttentionUNet.yml
```

| Entry | Checked against | Loads weights from |
| --- | --- | --- |
| `UNet`, `NestedUNet`, `ResNet` | KonfAI's Python classes, weight-exact | KonfAI checkpoints |
| `SegResNet`, `VNet`, `DynUNet` | MONAI, weight-exact | MONAI |
| `ResNet18`, `VGG16` | torchvision, weight-exact | torchvision ImageNet |
| `PlainConvUNet` | nnU-Net, weight-exact | nnU-Net, TotalSegmentator, MRSegmentator |
| `ResidualEncoderUNet` | nnU-Net, weight-exact | nnU-Net ResEnc, ImpactSeg |
| `UNetPlusPlus` | `segmentation_models_pytorch`, weight-exact | smp, ImpactSynth |
| `ViT` | MONAI's encoder features | |
| `AttentionUNet`, `UNETR` | structure only (the graph differs from MONAI's, as documented in the file) | |

`VGG16` exposes its five feature maps (`Block_0:Out` to `Block_4:Out`) for perceptual losses.

### Which form to use

| You want to | Use |
| --- | --- |
| Run a model as it is, one output, one loss | its class directly: `classpath: monai.networks.nets:SegResNet` |
| Supervise inner layers, or edit the architecture without code | the catalog: `default\|SegResNet.yml` |
| The same, starting from someone's pretrained weights | the catalog and `pretrained_from` (below) |

A model named by its class is a black box: only its final output can take a loss. The YAML form makes every
module addressable.

## Starting from other weights: `pretrained_from`

A new training can start from a checkpoint trained in another framework:

```yaml
Trainer:
  Model:
    classpath: default|SegResNet.yml
    pretrained_from:
      checkpoint: ./segresnet_pretrained.pt   # a matching MONAI SegResNet state_dict
      builder: monai.networks.nets:SegResNet
      args: {spatial_dims: 3, init_filters: 8, in_channels: 1, out_channels: 2}
      input_shape: [96, 96, 96]            # optional
```

KonfAI builds the original model, loads the checkpoint, copies the weights into the KonfAI graph layer by
layer in execution order, then checks that the KonfAI model reproduces the original's outputs. Any mismatch
is an error: a partial copy is never accepted. It only applies to a new `TRAIN`; `RESUME` and `PREDICTION`
use their own checkpoint. Give `input_shape` when the model cannot guess an input (several inputs, a free
patch axis).

From Python, `konfai.utils.pretrained.transfer_weights_by_execution_order` does the same copy:

```python
from importlib.resources import files

import torch
from monai.networks.nets import SegResNet
from konfai.utils.model_builder import build_model_from_yaml
from konfai.utils.pretrained import transfer_weights_by_execution_order

reference = SegResNet(spatial_dims=3, init_filters=8, in_channels=1, out_channels=2,
                      blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1))
reference.load_state_dict(torch.load("segresnet_pretrained.pt"))

net = build_model_from_yaml(yaml_path=str(files("konfai").joinpath("models/yaml/SegResNet.yml")),
                            parameters={"dim": 3, "upsample_mode": "trilinear", "nb_class": 2})
example = torch.randn(1, 1, 16, 16, 16)
transfer_weights_by_execution_order(
    target=net, source=reference,
    target_forward=lambda: list(net.named_forward(example)),
    source_forward=lambda: reference(example),
)
```

## A different number of classes: `allow_head_resize`

A checkpoint whose tensor shapes differ from the model's is refused. To fine-tune with a different number
of classes, set `allow_head_resize: true`: the matching part of each tensor is loaded, and each resized
tensor is reported.

```yaml
Trainer:
  Model:
    classpath: segmentation.UNet.UNet
    allow_head_resize: true
```

## Building blocks

To write your own model, `konfai.network.blocks` provides:

- `ConvBlock` (convolution, normalization, activation), `ResBlock`, `Attention`, `LatentDistribution`;
- `BlockConfig` for one convolution stage (`kernel_size=3, stride=1, padding=1, activation="ReLU",
  norm_mode="NONE"`);
- `NormMode` (`NONE`, `BATCH`, `INSTANCE`, `GROUP`, `LAYER`, `SYNCBATCH`), `UpsampleMode`, `DownsampleMode`.
  `SYNCBATCH` needs GPUs: use `BATCH` for a multi-process CPU run;
- small modules: `Add`, `Multiply`, `Concat`, `Detach`, `ArgMax`, `Select`, `View`, `Permute`.

## Next steps

- {doc}`losses-metrics`: the losses and metrics.
- {doc}`../../usage/custom-models`: writing your own `Network`.
- {doc}`../../usage/adopting-konfai`: bringing a PyTorch, MONAI or nnU-Net model.
