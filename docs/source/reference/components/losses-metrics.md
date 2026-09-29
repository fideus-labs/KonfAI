# Losses & metrics

Losses and metrics are both **criteria**: subclasses of
`konfai.metric.measure.Criterion` in `konfai/metric/measure/`. You attach them
to a **named model output** and one or more **target dataset groups**, under
`outputs_criterions:` (training) or `metrics:` (evaluation). Bare names resolve
inside `konfai.metric.measure`; you can also point at any library, e.g.
`torch:nn:L1Loss` or `monai.losses:DiceLoss`.

## Loss vs metric: how KonfAI actually decides

```{important}
Whether a criterion is **back-propagated is decided by the `is_loss:` flag in the
config, not by the Python return type.** `is_loss: true` adds the returned tensor
to the training loss; `is_loss: false` detaches it and only logs it. In an
`Evaluation.yml`, every criterion is a metric.

The return *shape* controls **logging**: a criterion may return a bare `Tensor`,
or a tuple `(tensor, value)` where the value is a float, a 0-d tensor read back
lazily, a per-label dict, or a `(values, labels)` pair (also read back lazily).
The tuple form is what lets `Dice`, `TRE`, etc. log clean per-label values while
still driving the gradient with the tensor.
```

So most pixelwise criteria are **dual-use**: the same class is a training loss or
a logged metric depending on `is_loss`. Leaving `is_loss` out takes the criterion's
own role: a loss for most of them, a metric for `PSNR` and `SSIM`. The resolved
value is written back to the config.

- `PSNR` is a metric only: it rises as the output nears the target, so minimising it
  would push the output away. `is_loss: true` on it is refused; train on `MSE`, which
  it is a function of.
- `SSIM` becomes a loss only with `is_loss: true`: it then minimises `1 - SSIM` with
  its gradient, and still reports the SSIM.

## Attaching a criterion (training)

```yaml
outputs_criterions:
  UNetBlock_0:Head:Conv:          # named model output (dotted path; ':' or '.')
    targets_criterions:
      SEG:                        # target group ("CT;MASK" to add a mask)
        criterions_loader:
          CrossEntropyLoss:       # criterion name (bare → konfai.metric.measure)
            is_loss: true         # true → back-propagated; false → metric only; left out → the criterion's role
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
          PSNR: { dynamic_range: 4095 }
          SSIM: { dynamic_range: 4095 }
```

Each criterion is logged and reported as `output:target:Name`. Two criteria of one class on
one output and target (the list spelling, `- Dice: {labels: [1]}` then `- Dice: {labels: [2]}`)
keep one entry each, the second as `Dice#2`: their losses add up, and each has its own row.

## How a logged value averages batches

The boards, the checkpoint score, early stopping and `ReduceLROnPlateau` read a
criterion over a run of batches: a validation pass, or the last `it_validation`
training steps. A criterion that declares `batch_mean` reports the mean of one
value per patch of the batch, so each batch weighs its patch count and the value
is the mean over every patch, whatever the batch size, a partial last batch, or
the number of ranks the validation is split over. Any other criterion weighs one
per batch.

- Declared: `MAE` and `MSE` with `reduction: mean`, `ME`, `SSIM`, `TRE`, `BCE`,
  `KLDivergence`, `FocalLoss` with `reduction: mean`, `CrossEntropyLoss` with
  `reduction: mean` and no `weight`, `Variance`, `Mean`, `PatchGanLoss`,
  `IMPACTReg`, `IMPACTSynth` and `SAM_Perceptual`. A masked mean runs over the
  patches the mask reaches: it is the mean over every patch when each patch holds
  a mask voxel.
- One per batch: `Dice` and `DiceSaveMap` (a ratio of the batch's pooled sums),
  `PSNR` and `LPIPS` (pooled over the batch without a mask), `GradientImages`,
  `Gram` and `PerceptualLoss`, `Accuracy`, a `sum` reduction, a class-weighted
  `CrossEntropyLoss`, and a criterion from another library (`torch:nn:L1Loss`).

Under several ranks the validation is split without duplicates: each patch is
scored once, and a rank a batch short runs one more batch, which it does not
score, so the ranks still run as many forwards.

## Pixelwise / regression

All subclass `MaskedLoss` and return `(Tensor, float)` (dual-use, except `PSNR`,
a metric only). Extra target
groups act as a mask: a voxel is inside wherever the mask is not 0 (0/1 and 0/255
alike), and several masks keep the voxels inside all of them. The same rule holds for
every `mask` in KonfAI (`Clip`, `Standardize`, `Mask`).

| Name | Purpose | Key args (defaults) |
| --- | --- | --- |
| `MSE` | Masked mean-squared error. | `reduction="mean"` |
| `MAE` | Masked mean-absolute error. | `reduction="mean"` |
| `ME` | Signed mean error `(x−y).mean()` (bias). |: |
| `PSNR` | Peak SNR over the mask. `dynamic_range` defaults to `4095` (HU range), which `None` also names. Metric only: `is_loss: true` is refused. | `dynamic_range=4095.0` |
| `MAESaveMap` | MAE that also returns a voxelwise L1 error map (a 3-tuple, for a save-map consumer). | `reduction="mean", dataset=None, group=None` |

## Segmentation / classification

| Name | Role | Purpose | Key args (defaults) |
| --- | --- | --- | --- |
| `Dice` | `(Tensor, dict)` dual-use | Soft Dice per label; loss `= 1 − mean(dice)`, per-label dict logged. Resamples target to output (nearest). A one-channel output is read as a label map: a float one with non-integer values (a sigmoid probability) is refused, so give it two channels or threshold it first. | `labels=None` (None → all present labels) |
| `CrossEntropyLoss` | `Tensor` loss | Wraps `nn.CrossEntropyLoss` (squeezes the target channel). | `weight=None, reduction="mean"` |
| `FocalLoss` | `Tensor` loss | Multi-class focal loss. `alpha` is an optional per-label weight list indexed by label id; `None` weights every class equally, and a list shorter than the class count is refused. | `gamma=2.0, alpha=None, reduction="mean"` |
| `Accuracy` | `Tensor` metric | This batch's classification accuracy. It keeps no state: the logging window averages it over the batches and resets between train and validation, so one figure never blends epochs or splits. |: |
| `DiceSaveMap` | 3-tuple | Dice + voxelwise error map (for a save-map consumer). | `labels=None, dataset=None, group=None` |

## Adversarial / style

| Name | Purpose | Key args (defaults) |
| --- | --- | --- |
| `BCE` | `BCEWithLogitsLoss` against a constant real/fake target. | `target=0` |
| `PatchGanLoss` | LSGAN-style MSE against a constant target. | `target=0` |
| `Gram` | Gram-matrix (style) loss. |: |
| `PerceptualLoss` | Feature-space perceptual loss over a pretrained KonfAI `Network` (custom multi-model forward; requires a real `path_model` checkpoint). | `model_loader`, `path_model`, `modules`, `shape` |

## Registration / distributions

| Name | Role | Purpose | Key args (defaults) |
| --- | --- | --- | --- |
| `TRE` | `(Tensor, dict)` metric | Target Registration Error between predicted/target landmark coordinates. |: |
| `GradientImages` | `Tensor` loss | Image-gradient smoothness loss (2D/3D auto); regulariser, or gradient-difference if a target is given. |: |
| `monai.losses:GlobalMutualInformationLoss` | `Tensor` loss | Parzen-window mutual information, by classpath (needs MONAI installed). | see MONAI |
| `KLDivergence` | `Tensor` loss | VAE KL term. **Rewires the graph** on init, inserting a `LatentDistribution` block; computes the closed-form KL between the latent Gaussian (the block's `mu` output and its `log_std` output, which holds the log-variance) and the prior N(`mu`, `std`²) its arguments set. | `shape` (**required**), `dim=100, mu=0, std=1` |

## Uncertainty / bookkeeping

| Name | Role | Purpose | Key args |
| --- | --- | --- | --- |
| `Variance` | `(Tensor, value)` metric | Channel-wise variance mean (ensemble/uncertainty). | `name="Variance"` |
| `Mean` | `(Tensor, value)` metric | Mean of the output tensor. | `name="Mean"` |
| `torch:nn:TripletMarginLoss` | `Tensor` loss | Triplet margin loss, by classpath. | see PyTorch |

## IMPACT feature-based criteria

These download TorchScript feature extractors from Hugging Face at construction
(`hf_hub_download`), so they need **network access**. The sanity check that probes the
extractor runs on the **CPU**: deliberately, since touching a GPU there crashed
CPU-only hosts and pinned every DDP rank to the same device.
All are `CriterionWithAttribute` and read the `Image*` statistics the `Statistics` transform
records. A target is read with its own. The output is read with its own when it has any (a
prediction under `EVALUATION`, for `IMPACTReg` and the content pass of `IMPACTSynth`), with the
reference's otherwise; `SAM_Perceptual` reads both with the reference's, and the style pass of
`IMPACTSynth` both with the style image's.
The mask, when given, is the target after the image (`Reference;Mask`), after the
content and style images for `IMPACTSynth`, whatever its type.

| Name | Purpose | Key args (defaults) |
| --- | --- | --- |
| `IMPACTReg` | Feature-space registration loss over the layers of a TorchScript model. | `name="Reg", model_name="TS/M291.pt", shape=[0,0], in_channels=3, loss="torch:nn:L1Loss", weights=[0,1]` |
| `IMPACTSynth` | Content (MSE) + style (Gram) perceptual synthesis loss over two TorchScript models. | `model_content_name`, `model_style_name` (**required**), plus per-branch shapes/channels/weights |
| `SAM_Perceptual` | SAM2-feature perceptual criterion (2D only). | `train=False, model_name="SAM2.1_Small.pt", weights=None` |

## Optional-dependency criteria

Imported lazily; a missing package raises a `MeasureError` with an install hint.

| Name | Extra | Purpose | Key args |
| --- | --- | --- | --- |
| `SSIM` | `konfai[ssim]` (scikit-image) | Masked structural similarity. `dynamic_range` defaults to `4095`, which `None` also names. A metric unless `is_loss: true`, which minimises `1 - SSIM`. | `dynamic_range=4095.0` |
| `LPIPS` | `konfai[lpips]` | Learned perceptual similarity (AlexNet by default), tiled over patches: a 2-D image whole, a volume slice by slice. | `model="alex"` |
| `torchmetrics.image.fid:FrechetInceptionDistance` | `torchmetrics` | Fréchet Inception Distance, by classpath. FID is defined over dataset-level feature distributions, so compute it over the whole prediction set, never per case. | see torchmetrics |

`LPIPS` rescales the output and the target to [-1, 1], each with its own min and max taken over
the whole batch. In training, one image's value therefore depends on the other images of its batch;
a batch of one is rescaled per image. The rescaling also removes any global gain and offset of
intensity (an output equal to `0.5 * target + 100` scores about 0), so pair it with `MAE` or
`PSNR` when intensities matter. A constant output or target (a background patch in a batch of one)
has no range. As a metric (`is_loss: false`, or in `Evaluation.yml`) it scores NaN, which the
averages skip. As a loss (`is_loss: true`, or `is_loss` left out: `LPIPS` is a loss by default) it
is rescaled to 0, so the loss and its gradient stay finite; any other input scores the same in both
roles.
Under a mask (`CT;MASK`), the voxels outside it are set to 0 in both images and each image of the
batch is rescaled and scored on its own; the value is the mean over the images whose mask is not
empty.

None of these pins a device. The `IMPACT*` sanity check probes its TorchScript
extractor on the CPU and discards the result; `LPIPS` follows the device
of the tensor it is given: the rank's GPU under DDP, or the CPU. `SAM_Perceptual`
runs no sanity check at all.

## Schedulers

KonfAI has **two distinct scheduler families**, resolved by different loaders.
Don't confuse them. Use this section when filling in a `schedulers:` block: first
check *which* of the two blocks you are in, then pick a name from the matching
table.

### A. Criterion-weight schedulers

These schedule the **weight of a loss** over training, in the `schedulers:`
subtree of a criterion (`konfai/metric/schedulers.py`, base class `Scheduler`):

```yaml
CrossEntropyLoss:
  is_loss: true
  schedulers:
    Constant: { nb_step: 0, value: 1 }
```

| Name | Purpose | Config args (defaults) |
| --- | --- | --- |
| `Constant` | Fixed weight for all iterations. | `value=1` (+ `nb_step`) |
| `CosineAnnealing` | Cosine anneal from `start_value` to `eta_min` over `t_max`. | `start_value=1, eta_min=1e-5, t_max=100` (+ `nb_step`) |

Each entry carries an `nb_step` (window width). Multiple schedulers can be
**chained** into consecutive iteration windows; an `nb_step: 0` (or `None`)
window is the terminal, always-on schedule.

### B. Learning-rate schedulers

These schedule the **optimizer learning rate**, in the model's `schedulers:`
block. The loader searches **both** `torch.optim.lr_scheduler` **and**
`konfai.metric.schedulers`, so you can use any torch scheduler (`StepLR`,
`ReduceLROnPlateau`, `CosineAnnealingLR`, …) **plus** the two KonfAI additions:

```yaml
schedulers:
  StepLR: { step_size: 20, gamma: 0.5 }     # any torch LR scheduler works
```

| Name | Purpose | Config args (defaults) |
| --- | --- | --- |
| `Warmup` | Linear LR warmup wrapper (`LambdaLR`). | `warmup_steps=10, last_epoch=-1` (+ `nb_step`) |
| `PolyLRScheduler` | nnU-Net-style polynomial LR decay `lr = initial_lr·(1 − step/max_steps)^exponent`. | `initial_lr` (**required**), `max_steps` (**required**), `exponent=0.9` (+ `nb_step`) |

The `optimizer` itself is injected by the framework, you do not write it under
`schedulers:`.

## Next steps

- {doc}`models`: the named outputs criteria attach to
- {doc}`../../config_guide/training`: the `optimizer:` / `schedulers:` blocks
- {doc}`../../usage/custom-models`: write your own `Criterion`
