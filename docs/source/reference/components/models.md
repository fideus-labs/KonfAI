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
VoxelMorph, …) must stay as Python classes. The registry is deliberately small —
see {doc}`../../concepts/yaml-model-builder`.
```

## Segmentation — `konfai.models.segmentation`

| Model | Classpath | Purpose | Key args (defaults) | Dims | Stability | YAML-buildable |
| --- | --- | --- | --- | --- | --- | --- |
| `UNet` | `segmentation.UNet.UNet` | Classic encoder–decoder U-Net; optional attention gates and deep-supervision heads. | `channels=[1,64,128,256,512,1024]`, `nb_class=2`, `dim=3`, `block_config=BlockConfig()`, `nb_conv_per_stage=2`, `downsample_mode="MAXPOOL"`, `upsample_mode="CONV_TRANSPOSE"`, `attention=False`, `block_type="Conv"` | 2D / 3D | **Stable** (tested + the `Segmentation` example) | Yes |
| `NestedUNet` | `segmentation.NestedUNet.NestedUNet` | UNet++ with dense nested skips and per-level deep-supervision heads. | as `UNet` + `activation="Softmax"` | 2D / 3D | **Usable** (implemented; blocks reused elsewhere) | Yes |
| `UNetpp` | `segmentation.NestedUNet.UNetpp` | UNet++ on a ResNet-style encoder. | `encoder_channels=[1,64,64,128,256,512]`, `decoder_channels=[256,128,64,32,16,1]`, `layers=[3,4,6,3]`, `dim=2` | 2D | **Experimental** (untested; hardcoded activations/norms; no `nb_class` knob) | Yes (untested) |

```{warning}
The `Model:UNetpp5` used in the `Synthesis` example is a **local** class in
`examples/Synthesis/Model.py` wrapping `segmentation_models_pytorch`, **not** the
built-in `UNetpp` above. Don't confuse the two.
```

## Classification — `konfai.models.classification`

| Model | Classpath | Purpose | Key args (defaults) | Dims | Stability | YAML-buildable |
| --- | --- | --- | --- | --- | --- | --- |
| `ResNet` | `classification.resnet.ResNet` | ResNet-18/34/50/101/152 family with torchvision-compatible weight aliases. | `dim=3`, `in_channels=1`, `depths=[2,2,2,2]`, `widths=[64,64,128,256,512]`, `num_classes=10`, `use_bottleneck=False` | 2D / 3D | **Usable** (implemented; not tested in-repo) | Yes |
| `ConvNeXt` | `classification.convNeXt.ConvNeXt` | ConvNeXt (tiny→xlarge presets) with a multi-head classifier (`num_classes` is a list). | `dim=3`, `in_channels=1`, `depths=[3,3,27,3]`, `widths=[128,256,512,1024]`, `drop_p=0.1`, `num_classes=[4,7]` | **2D only** | **Broken by default** (see below) | No (custom `forward`) |

```{warning}
**`ConvNeXt` crashes in 3D, which is its default.** `LayerScaler.gamma` is
hardcoded to 2 spatial dimensions (`torch.ones(dimensions, 1, 1)`,
`convNeXt.py`), so a default `ConvNeXt()` on `[B,C,D,H,W]` input raises a
`RuntimeError`. Use it in 2D, or fix the layer for 3D. Tracked as finding N6 in
`AUDIT_KONFAI.md`.
```

## Generation — `konfai.models.generation`

| Model | Classpath | Purpose | Dims | Stability | YAML-buildable |
| --- | --- | --- | --- | --- | --- |
| `VAE` | `generation.vae.VAE` | Convolutional auto-encoder. **Deterministic** — despite the name there is no latent sampling. | 2D / 3D | **Stable** (autoencoder) | Yes |
| `LinearVAE` | `generation.vae.LinearVAE` | True fully-connected variational AE (`LatentDistribution` reparam bottleneck). Pairs with the `KLDivergence` loss. | 1D (flat vectors) | **Stable** (tested) | No (`LatentDistribution`) |
| `Generator` / `Discriminator` / `Gan` | `generation.gan.*` | PatchGAN discriminator + ResNet-autoencoder generator + composite adversarial graph. | 2D / 3D | **Experimental** (research; `SYNCBATCH` needs DDP; hardcoded channels) | No |
| `DDPM` (+ `MSE`) | `generation.ddpm.DDPM` | Conditional denoising-diffusion U-Net. | 2D / 3D (nominal) | **Broken by default** — `__init__` raises `NotImplementedError` (not runnable) | No |
| `DiffusionGan`, `DiffusionGanV2`, `DiffusionCycleGan`, `CycleGan*` | `generation.diffusionGan.*` | Adversarial + diffusion + CycleGAN research family. | 2D / 3D | **Experimental** (undocumented; some wire losses via hardcoded key strings) | No |
| `cStyleGan.Generator` | `generation.cStyleGan.Generator` | Conditional StyleGAN-style generator with weight-modulated convs. | 2D / 3D | **Experimental** (no wrapper/test/example) | No |

```{warning}
`DDPM()` raises on construction — *"DDPM is experimental and not currently
runnable"* (`ddpm.py`). It is kept as scaffolding, not a usable model today.
```

## Registration — `konfai.models.registration`

| Model | Classpath | Purpose | Dims | Stability | YAML-buildable |
| --- | --- | --- | --- | --- | --- |
| `VoxelMorph` | `registration.registration.VoxelMorph` | Learning-based deformable/rigid registration (U-Net flow field + spatial-transformer warp + scaling-and-squaring integration). | **2D only** | **Broken by default** — `__init__` raises unless `dim==2`, but defaults to `dim=3` | No |

```{warning}
`VoxelMorph()` with the default `dim=3` raises `NotImplementedError`: its
`SpatialTransformer`/`VecInt`/`ResizeTransform` are 2-D-hardcoded. You must pass
`dim: 2`. For production registration, use the **IMPACT-Reg app** instead (see
{doc}`../../ecosystem/index`).
```

## Representation — `konfai.models.representation`

| Model | Classpath | Purpose | Dims | Stability | YAML-buildable |
| --- | --- | --- | --- | --- | --- |
| `Representation` | `representation.representation.Representation` | Self-supervised / triplet-style representation learner: a frozen conv encoder + trainable linear projection head. | 3D (hardcoded) | **Experimental** (example-grade; feature dims fixed to a 32-channel encoder) | No |

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
`Exit` — that have side effects (print/write-to-disk/raise) on every forward.
They are for graph debugging; never leave them in a trained model.
```

## See also

- {doc}`../../concepts/model-graph` — named outputs, `outputs_criterions`, patching
- {doc}`../../concepts/yaml-model-builder` — build a model from YAML
- {doc}`../../usage/custom-models` — subclass `Network` yourself
- {doc}`losses-metrics` — attach losses/metrics to a model's named outputs
