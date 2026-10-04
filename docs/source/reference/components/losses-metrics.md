# Losses and metrics

Losses and metrics are the same kind of object, a criterion (`konfai.metric.measure`). A criterion is
attached to a model output and a target, under `outputs_criterions:` in training or `metrics:` in
evaluation. A bare name (`Dice`) is a KonfAI criterion; `torch:nn:L1Loss` or `monai.losses:DiceLoss` name
one from another library.

## Loss or metric

`is_loss: true` makes a criterion a loss (its value is back-propagated); `is_loss: false` only logs it. In
`Evaluation.yml` every criterion is a metric. Without `is_loss`, a criterion takes its own role: a loss,
except `Accuracy`, `PSNR` and `SSIM`, which are metrics.

- `PSNR` cannot be a loss (it grows as the output improves): train on `MSE`.
- `Accuracy` cannot be a loss (choosing the highest-scoring class has no gradient): train on `CrossEntropyLoss`.
- `SSIM` with `is_loss: true` minimises `1 - SSIM` and still reports the SSIM.

## Attaching a criterion

```yaml
outputs_criterions:
  UNetBlock_0:Head:Conv:          # a model output
    targets_criterions:
      SEG:                        # the target group ("CT;MASK" adds a mask)
        criterions_loader:
          CrossEntropyLoss:
            is_loss: true
            schedulers:
              Constant: { nb_step: 0, value: 1 }   # the loss weight over time
            group: 0              # losses of one group are summed (a GAN's generator vs discriminator)
            start: 0              # first iteration it is active
            stop: None            # last iteration (None: never stops)
            accumulation: false
            reduction: mean       # any other key is the criterion's own argument
```

In evaluation, the same without `is_loss`, `schedulers` or `group`:

```yaml
metrics:
  sCT:
    targets_criterions:
      CT;MASK:
        criterions_loader:
          MAE:  { reduction: mean }
          PSNR: { dynamic_range: 4095 }
          SSIM: { dynamic_range: 4095 }
```

Each value is logged as `output:target:Name`. The same criterion twice on one output (in list form) is logged
as `Dice` and `Dice#2`.

A mask is a target placed after the image (`CT;MASK`): a voxel counts where the mask is not 0. Several masks
keep the voxels inside all of them.

Logged values are averaged over the patches of the validation or of the last training steps, whatever the
batch size and the number of GPUs. Criteria computed on the batch as a whole (`Dice`, `PSNR`, `LPIPS`, a `sum`
reduction, a criterion from another library) are averaged per batch instead.

## Regression

| Name | Purpose | Key arguments (defaults) |
| --- | --- | --- |
| `MSE` | Mean squared error. | `reduction="mean"` |
| `MAE` | Mean absolute error. | `reduction="mean"` |
| `ME` | Mean signed error (bias). | |
| `PSNR` | Peak signal-to-noise ratio. Metric only. | `dynamic_range=4095.0` |
| `SSIM` | Structural similarity. A metric unless `is_loss: true`. | `dynamic_range=4095.0` |
| `MAESaveMap` | MAE that also writes the error map. | `reduction="mean", dataset=None, group=None` |

With a positive `dynamic_range`, an exact match scores `PSNR = +inf`, including with a mask or streamed
evaluation. Evaluation JSON stores this as `null`, like other non-finite scores, and excludes it from
finite aggregates. The case still reports its finite metrics (`MSE = 0`, for example).

## Segmentation and classification

| Name | Purpose | Key arguments (defaults) |
| --- | --- | --- |
| `Dice` | Dice per label; the loss is `1 - mean`. Takes probabilities (after a `Softmax`) or a label map; logits are refused. | `labels=None` (every label present) |
| `CrossEntropyLoss` | Cross entropy, on logits. | `weight=None, reduction="mean"` |
| `FocalLoss` | Focal loss; `alpha` weights each label. | `gamma=2.0, alpha=None, reduction="mean"` |
| `Accuracy` | Fraction of correct classes, chosen from logits or probabilities. Metric only. | |
| `DiceSaveMap` | Dice that also writes the error map. | `labels=None, dataset=None, group=None` |

## Adversarial and perceptual

| Name | Purpose | Key arguments (defaults) |
| --- | --- | --- |
| `BCE` | Binary cross entropy against a constant real/fake target. | `target=0` |
| `PatchGanLoss` | Least-squares GAN loss. | `target=0` |
| `Gram` | Style loss on Gram matrices. | |
| `PerceptualLoss` | Feature loss through a pretrained KonfAI network. | `model_loader`, `path_model`, `modules`, `shape` |
| `LPIPS` | Learned perceptual similarity (`konfai[lpips]`), slice by slice on volumes. It ignores global intensity scale: pair it with `MAE` or `PSNR`. | `model="alex"` |

## Registration and other

| Name | Purpose | Key arguments (defaults) |
| --- | --- | --- |
| `TRE` | Target registration error between landmarks. | |
| `GradientImages` | Gradient smoothness (or gradient difference with a target). | |
| `KLDivergence` | VAE KL term, against the prior N(`mu`, `std`²). | `shape` (required), `dim=100, mu=0, std=1` |
| `Variance`, `Mean` | Variance or mean of the output (uncertainty). | |
| `monai.losses:GlobalMutualInformationLoss` | Mutual information (needs MONAI). | |
| `torch:nn:TripletMarginLoss` | Triplet loss. | |
| `torchmetrics.image.fid:FrechetInceptionDistance` | FID, over the whole prediction set, never per case. | |

## IMPACT criteria

These compare images in the feature space of a pretrained network. They download it from Hugging Face on
first use, so they need network access once. They read the image statistics that the `Statistics` transform
records, so add `Statistics` to the chains of the groups they compare. A mask goes after the images
(`Reference;Mask`).

| Name | Purpose | Key arguments (defaults) |
| --- | --- | --- |
| `IMPACTReg` | Registration loss on the features of a TorchScript model. | `model_name="TS/M291.pt", shape=[0,0], in_channels=3, loss="torch:nn:L1Loss", weights=[0,1]` |
| `IMPACTSynth` | Content and style loss for synthesis, on two models. | `model_content_name`, `model_style_name` (required) |
| `SAM_Perceptual` | Perceptual criterion on SAM2 features (2D). | `train=False, model_name="SAM2.1_Small.pt", weights=None` |

## Schedulers

There are two kinds, in two different places.

**Loss weights**, in a criterion's `schedulers:` block:

| Name | Purpose | Arguments (defaults) |
| --- | --- | --- |
| `Constant` | A fixed weight. | `value=1`, `nb_step` |
| `CosineAnnealing` | From `start_value` down to `eta_min` over `t_max`. | `start_value=1, eta_min=1e-5, t_max=100`, `nb_step` |

`nb_step` is how many iterations a scheduler lasts; several chain one after the other, and `nb_step: 0` lasts
until the end.

**Learning rates**, in the model's `schedulers:` block. Any `torch.optim.lr_scheduler` class works
(`StepLR`, `ReduceLROnPlateau`, `CosineAnnealingLR`, …), plus two from KonfAI:

```yaml
schedulers:
  StepLR: { step_size: 20, gamma: 0.5 }
```

| Name | Purpose | Arguments (defaults) |
| --- | --- | --- |
| `Warmup` | Linear warm-up. | `warmup_steps=10`, `nb_step` |
| `PolyLRScheduler` | nnU-Net polynomial decay. | `initial_lr`, `max_steps` (required), `exponent=0.9`, `nb_step` |

## Next steps

- {doc}`models`: the outputs criteria attach to.
- {doc}`../../config_guide/training`: the `optimizer` and `schedulers` blocks.
- {doc}`../../usage/custom-models`: writing your own criterion.
