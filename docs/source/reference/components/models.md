# Models

Built-in networks under `konfai/models/`, referenced in a config by `classpath`
(e.g. `classpath: segmentation.UNet.UNet`, or the bare dotted form). A model is
a subclass of `konfai.network.Network` — a routed `add_module` graph. See
{doc}`../../concepts/model-graph` for how the graph and its named outputs work,
and {doc}`../../concepts/yaml-model-builder` for building a feed-forward model
entirely from YAML.

```{note}
**"YAML-buildable" column.** A model is buildable from a `.yml` (via the safe
{doc}`../../concepts/yaml-model-builder`) only if it is a pure graph of registry
node types. Models with a custom `forward()` (diffusion, StyleGAN, ConvNeXt,
VoxelMorph, …) are written as Python classes. The registry is deliberately small
— see {doc}`../../concepts/yaml-model-builder`.
```

## Segmentation — `konfai.models.python.segmentation`

`PlainConvUNet` is the parametric nnU-Net backbone (n_stages / features_per_stage /
strides / n_conv_per_stage as real arguments), weight-exact to
`dynamic_network_architectures.PlainConvUNet` for any topology — a real nnU-Net /
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
| `UNetPlusPlus` | `segmentation.unetplusplus.UNetPlusPlus` | Parametric UNet++ on a **pretrained ResNet backbone** — **weight-exact vs `smp.UnetPlusPlus`** (resnet18/34). Use this to load an smp / ImpactSynth checkpoint into an addressable KonfAI graph. | `dim=2`, `in_channels=1`, `classes=1`, `encoder_name="resnet34"`, `activation=None` (`"tanh"` for sCT) | 2D | Yes (params) |
| `ResidualEncoderUNet` | `segmentation.residualencoderunet.ResidualEncoderUNet` | Parametric nnU-Net **residual-encoder** U-Net for any topology — **weight-exact vs `dynamic_network_architectures.ResidualEncoderUNet`**. Loads a real nnU-Net ResEnc / ImpactSeg checkpoint via the bridge. | `dim=3`, `in_channels=1`, `n_stages=6`, `features_per_stage=[32,64,128,256,320,320]`, `strides=[1,2,2,2,2,2]`, `n_blocks_per_stage=[1,3,4,6,6,6]`, `num_classes=2`, `deep_supervision=True` | 2D / 3D | Yes (params) |

```{note}
Two UNet++ flavours: `NestedUNet` (academic, plain-conv encoder trained from scratch) and
`UNetPlusPlus` (the **smp-faithful** UNet++ with a real pretrained ResNet backbone — the one
that loads `smp.UnetPlusPlus` checkpoints). Pick `UNetPlusPlus` when you need smp
weight-compatibility (e.g. the ImpactSynth app).
```

```{note}
The `Model:UNetpp5` used in the `Synthesis` example is a **local** class in
`examples/Synthesis/Model.py` wrapping `segmentation_models_pytorch`; the built-in
`UNetPlusPlus` above is the maintained, smp-weight-exact equivalent.
```

## Classification — `konfai.models.python.classification`

| Model | Classpath | Purpose | Key args (defaults) | Dims | YAML-buildable |
| --- | --- | --- | --- | --- | --- |
| `ResNet` | `classification.resnet.ResNet` | ResNet-18/34/50/101/152 family with torchvision-compatible weight aliases. | `dim=3`, `in_channels=1`, `depths=[2,2,2,2]`, `widths=[64,64,128,256,512]`, `num_classes=10`, `use_bottleneck=False` | 2D / 3D | Yes |
| `ConvNeXt` | `classification.convNeXt.ConvNeXt` | ConvNeXt (tiny→xlarge presets) with a multi-head classifier (`num_classes` is a list). | `dim=3`, `in_channels=1`, `depths=[3,3,27,3]`, `widths=[128,256,512,1024]`, `drop_p=0.1`, `num_classes=[4,7]` | 2D | No (custom `forward`) |

## Generation — `konfai.models.python.generation`

| Model | Classpath | Purpose | Dims | YAML-buildable |
| --- | --- | --- | --- | --- |
| `VAE` | `generation.vae.VAE` | Convolutional auto-encoder. **Deterministic** — despite the name there is no latent sampling. | 2D / 3D | Yes |
| `LinearVAE` | `generation.vae.LinearVAE` | Fully-connected variational AE (`LatentDistribution` reparam bottleneck). Pairs with the `KLDivergence` loss. | 1D (flat vectors) | No (`LatentDistribution`) |
| `Generator` / `Discriminator` / `Gan` | `generation.gan.*` | PatchGAN discriminator + ResNet-autoencoder generator + composite adversarial graph. | 2D / 3D | No |
| `DDPM` (+ `MSE`) | `generation.ddpm.DDPM` | Conditional denoising-diffusion U-Net. | 2D / 3D | No |
| `DiffusionGan`, `DiffusionGanV2`, `DiffusionCycleGan`, `CycleGan*` | `generation.diffusionGan.*` | Adversarial + diffusion + CycleGAN family. | 2D / 3D | No |
| `cStyleGan.Generator` | `generation.cStyleGan.Generator` | Conditional StyleGAN-style generator with weight-modulated convs. | 2D / 3D | No |

## Registration — `konfai.models.python.registration`

| Model | Classpath | Purpose | Dims | YAML-buildable |
| --- | --- | --- | --- | --- |
| `VoxelMorph` | `registration.registration.VoxelMorph` | Learning-based deformable/rigid registration (U-Net flow field + spatial-transformer warp + scaling-and-squaring integration). Pass `dim: 2`. | 2D | No |

## Representation — `konfai.models.python.representation`

| Model | Classpath | Purpose | Dims | YAML-buildable |
| --- | --- | --- | --- | --- |
| `Representation` | `representation.representation.Representation` | Self-supervised / triplet-style representation learner: a frozen conv encoder + trainable linear projection head. | 3D | No |

## Building blocks (`konfai.network.blocks`)

If you author your own model — as a Python `Network` or a YAML graph — these
reusable pieces are the vocabulary:

- **Conv graphs:** `ConvBlock` (`[Conv → Norm → Activation]×N`), `ResBlock`
  (residual with projected skip), `Attention` (Attention-U-Net gate),
  `LatentDistribution` (VAE reparameterisation, exposes `mu`/`log_std`/`z`).
- **`BlockConfig`** — one conv stage: `kernel_size=3, stride=1, padding=1,
  bias=True, activation="ReLU", norm_mode="NONE"`. `activation` accepts a name,
  a `";"`-separated spec (`"LeakyReLU;0.2;True"`), a callable, or `None`.
- **Enums:** `NormMode` (`NONE/BATCH/INSTANCE/GROUP/LAYER/SYNCBATCH/INSTANCE_AFFINE`),
  `UpsampleMode` (`CONV_TRANSPOSE/UPSAMPLE`), `DownsampleMode`
  (`MAXPOOL/AVGPOOL/CONV_STRIDE`).
- **Tensor ops** (leaf modules): `Add`, `Multiply`, `Concat`, `Detach`,
  `ArgMax`, `Select`, `View`, `Permute`, `NormalNoise`, and more.

```{note}
`konfai.network.blocks` also defines **debug-only** blocks — `Print`, `Write`,
`Exit` — that have side effects (print / write-to-disk / raise) on every forward.
They are for graph debugging; do not leave them in a trained model.
```

## Next steps

- {doc}`../../concepts/model-graph` — named outputs, `outputs_criterions`, patching
- {doc}`losses-metrics` — attach losses/metrics to a model's named outputs
- {doc}`../../usage/custom-models` — subclass `Network` yourself
