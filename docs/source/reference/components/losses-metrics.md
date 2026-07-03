# Losses & metrics

Losses and metrics are both **criteria** — subclasses of
`konfai.metric.measure.Criterion` in `konfai/metric/measure.py`. You attach them
to a **named model output** and one or more **target dataset groups**, under
`outputs_criterions:` (training) or `metrics:` (evaluation). Bare names resolve
inside `konfai.metric.measure`; you can also point at any library, e.g.
`torch:nn:L1Loss` or `monai.losses:DiceLoss`.

## Loss vs metric — how KonfAI actually decides

```{important}
Whether a criterion is **back-propagated is decided by the `is_loss:` flag in the
config, not by the Python return type.** `is_loss: true` adds the returned tensor
to the training loss; `is_loss: false` detaches it and only logs it. In an
`Evaluation.yml`, every criterion is a metric.

The return *shape* controls **logging**: a criterion may return a bare `Tensor`,
or a tuple `(tensor, scalar_or_dict)`. The tuple form is what lets `Dice`, `TRE`,
etc. log clean per-label values while still driving the gradient with the tensor.
```

So most pixelwise criteria are **dual-use**: the same class is a training loss or
a logged metric depending on `is_loss`. The tables note the typical role.

## Attaching a criterion (training)

```yaml
outputs_criterions:
  UNetBlock_0:Head:Conv:          # named model output (dotted path; ':' or '.')
    targets_criterions:
      SEG:                        # target group ("CT;MASK" to add a mask)
        criterions_loader:
          CrossEntropyLoss:       # criterion name (bare → konfai.metric.measure)
            is_loss: true         # true → back-propagated; false → metric only
            schedulers:
              Constant: { nb_step: 0, value: 1 }   # weight schedule
            group: 0              # loss/optimizer group (e.g. GAN G vs D)
            start: 0              # first active iteration
            stop: None            # last active iteration (None = never)
            accumulation: false
            # any remaining keys are forwarded to the criterion's constructor:
            reduction: mean
```

The reserved keys (`is_loss`, `group`, `start`, `stop`, `accumulation`, plus the
`schedulers:` subtree) are consumed by KonfAI; **all other keys are the
criterion's own constructor arguments**. For evaluation the shape is the same,
without `is_loss`/`schedulers`/`group`:

```yaml
metrics:
  sCT:
    targets_criterions:
      CT;MASK:                    # ';' joins target + mask into one masked metric
        criterions_loader:
          MAE:  { reduction: mean }
          PSNR: { dynamic_range: None }
          SSIM: { dynamic_range: None }
```

## Pixelwise / regression

All subclass `MaskedLoss` and return `(Tensor, float)` (dual-use). Extra target
groups act as a mask.

| Name | Purpose | Key args (defaults) | Stability |
| --- | --- | --- | --- |
| `MSE` | Masked mean-squared error. | `reduction="mean"` | **Stable** |
| `MAE` | Masked mean-absolute error. | `reduction="mean"` | **Stable** (both examples) |
| `ME` | Signed mean error `(x−y).mean()` (bias). | — | **Usable** |
| `PSNR` | Peak SNR over the mask. Default `dynamic_range` falls back to `4095` (HU range). | `dynamic_range=None` | **Stable** |
| `MAESaveMap` | MAE that also emits a voxelwise error map. | `reduction="mean", dataset=None, group=None` | **Experimental** — returns a 3-tuple the normal loss path can't unpack; only for a dedicated save-map consumer. |

## Segmentation / classification

| Name | Role | Purpose | Key args (defaults) | Stability |
| --- | --- | --- | --- | --- |
| `Dice` | `(Tensor, dict)` dual-use | Soft Dice per label; loss `= 1 − mean(dice)`, per-label dict logged. Resamples target to output (nearest). | `labels=None` (None → all present labels) | **Stable** (tested + example) |
| `CrossEntropyLoss` | `Tensor` loss | Wraps `nn.CrossEntropyLoss` (squeezes the target channel). | `weight=None, reduction="mean"` | **Stable** (Segmentation example) |
| `FocalLoss` | `Tensor` loss | Multi-class focal loss. | `gamma=2.0, alpha=[0.5,2.0,0.5,0.5,1], reduction="mean"` | **Usable, fragile** — `alpha` default is a **length-5** list indexed by label id; mismatched `nb_class` → wrong weights / `IndexError`. |
| `Accuracy` | `Tensor` metric | Running classification accuracy. | — | **Fragile** — accumulates across the whole run with **no reset**; the value is cumulative. |
| `DiceSaveMap` | 3-tuple | Dice + voxelwise error map. | `labels=None, dataset=None, group=None` | **Experimental** — 3-tuple; not usable on the normal loss path. |

## Adversarial / style

| Name | Purpose | Key args (defaults) | Stability |
| --- | --- | --- | --- |
| `BCE` | `BCEWithLogitsLoss` against a constant real/fake target. | `target=0` | **Usable** (GAN block) |
| `PatchGanLoss` | LSGAN-style MSE against a constant target. | `target=0` | **Stable** (Synthesis GAN example) |
| `WGP` | `mean((output−1)²)` WGAN-style penalty. | — | **Usable** |
| `Gram` | Gram-matrix (style) loss. | — | **Usable** (used inside `IMPACTSynth`) |
| `PerceptualLoss` | Feature-space perceptual loss over a pretrained KonfAI `Network`. | `model_loader`, `path_model`, `modules`, `shape` | **Experimental / heavy** — custom multi-model forward; `preprocessing()` is a no-op; the default `path_model` is a placeholder that must be replaced with a real checkpoint. |

## Registration / distributions

| Name | Role | Purpose | Key args (defaults) | Stability |
| --- | --- | --- | --- | --- |
| `TRE` | `(Tensor, dict)` metric | Target Registration Error between predicted/target landmark coordinates. | — | **Stable** (tested) |
| `GradientImages` | `Tensor` loss | Image-gradient smoothness loss (2D/3D auto); regulariser, or gradient-difference if a target is given. | — | **Stable** |
| `MutualInformationLoss` | `Tensor` loss | Parzen-window Gaussian mutual information (returns `−MI`). Subclasses `nn.Module` directly. | `num_bins=23, sigma_ratio=0.5` | **Usable** |
| `KLDivergence` | `Tensor` loss | VAE KL term. **Rewires the graph** on init, inserting a `LatentDistribution` block; computes closed-form KL from `mu`/`log_std`. | `shape` (**required**), `dim=100, mu=0, std=1` | **Stable** (tested) |

## Uncertainty / bookkeeping

| Name | Role | Purpose | Key args | Stability |
| --- | --- | --- | --- | --- |
| `Variance` | `(Tensor, float)` metric | Channel-wise variance mean (ensemble/uncertainty). | `name="Variance"` | **Stable** |
| `Mean` | `(Tensor, float)` metric | Mean of the output tensor. | `name="Mean"` | **Stable** |
| `TripletLoss` | `Tensor` loss | `nn.TripletMarginLoss` over a 3-tuple output. | — | **Usable** (representation) |
| `L1LossRepresentation` | `Tensor` loss | L1 between two representations + variance-collapse regulariser. | — | **Usable** |

## IMPACT feature-based criteria (network required)

These download TorchScript feature extractors from Hugging Face at construction
(`hf_hub_download`), so they need **network access**; the sanity check uses GPU 0.
All are `CriterionWithAttribute` and consume per-group `Attribute` statistics.

| Name | Purpose | Key args (defaults) | Stability |
| --- | --- | --- | --- |
| `IMPACTReg` | Feature-space registration loss over the layers of a TorchScript model. | `name="Reg", model_name="TS/M291.pt", shape=[0,0], in_channels=3, loss="torch:nn:L1Loss", weights=[0,1]` | **Usable** (external HF dependency; powers the IMPACT-Reg app) |
| `IMPACTSynth` | Content (MSE) + style (Gram) perceptual synthesis loss over two TorchScript models. | `model_content_name`, `model_style_name` (**required**), plus per-branch shapes/channels/weights | **Usable** (in tests; heavy plumbing; Synthesis example) |
| `SAM_Perceptual` | SAM2-feature perceptual criterion. | `train=False, model_name="SAM2.1_Small.pt", weights=None` | **Usable** — **2D only**; external HF dependency; no tests |

## Optional-dependency criteria

Imported lazily; a missing package raises a `MeasureError` with an install hint.

| Name | Extra | Purpose | Key args | Stability |
| --- | --- | --- | --- | --- |
| `SSIM` | `konfai[ssim]` (scikit-image) | Masked structural similarity. Default `dynamic_range → 4024`. | `dynamic_range=None` | **Stable** (tested) |
| `LPIPS` | `konfai[lpips]` | Learned perceptual similarity (AlexNet by default), tiled over patches. | `model="alex"` | **Usable** — pinned to GPU 0; untested here. |
| `FID` | `konfai[fid]` (scipy + torchvision) | Fréchet Inception Distance (InceptionV3). | — | **Usable** — CUDA-hardcoded; untested here. |

```{note}
`IMPACT*` and `SAM_Perceptual` pin the sanity check to GPU 0; `LPIPS`/`FID` are
also hardcoded to CUDA device 0. They will not work on CPU-only machines.
```

## See also

- {doc}`schedulers` — the weight schedulers used in the `schedulers:` subtree
- {doc}`../../concepts/model-graph` — the named outputs criteria attach to
- {doc}`../../config_guide/training` and {doc}`../../config_guide/evaluation`
- {doc}`../api/extension-points` — write your own `Criterion`
