# Transforms

Transforms are deterministic processing stages (`konfai/data/transform/`). They run in every workflow, on
the whole case (`transforms:`) or on each patch (`patch_transforms:`), under a dataset group:

```yaml
Dataset:
  groups_src:
    CT:
      groups_dest:
        CT:
          transforms:
            Standardize: {}
          is_input: true
```

The {doc}`../../examples/visual-gallery` shows each one on real images.

In the tables below:

- **Shape**: the stage changes the spatial shape (it implements `transform_shape()`).
- **Inv**: the stage can be undone with `inverse: true` (to bring a prediction back to the original space).
- **Stream**: the stage works region by region, so the case never has to be loaded whole
  ({doc}`../../usage/large-images`).

`†` changes the number of channels, not the spatial shape. `‡` changes the tensor's rank, so the case is
loaded whole.

## Intensity

| Name | Purpose | Key arguments (defaults) | Shape | Inv | Stream |
| --- | --- | --- | --- | --- | --- |
| `Clip` | Clamp to a range: numbers, `"min"`/`"max"`, or `"percentile:<p>"`. | `min_value=-1024, max_value=1024, save_clip_min=False, save_clip_max=False, mask=None` | no | no | yes, except with a percentile |
| `Normalize` | Map linearly to `[min, max]`. | `min_value=-1, max_value=1, channels=None, lazy=False, inverse=True` | no | yes | yes |
| `UnNormalize` | Map `[-1, 1]` to `[min, max]`. | `min_value=-1024, max_value=3071` | no | no | yes |
| `Standardize` | Zero mean, unit standard deviation, from the volume, a mask, or given values. | `mean=None, std=None, mask=None, lazy=False, inverse=True` | no | yes | yes |
| `HistogramMatching` | Match the histogram of a reference group. | `reference_group` | no | no | no |

## Geometry

| Name | Purpose | Key arguments (defaults) | Shape | Inv | Stream |
| --- | --- | --- | --- | --- | --- |
| `Resample` | Change the grid, and optionally apply a map ({doc}`../../config_guide/transform`). | `spacing`, `shape`, `reference`, `reference_group`, `reference_dataset`, `transforms`, `field`, `field_group`, `align="extent"`, `interpolation=None`, `fill=0.0`, `inverse=True` | yes | yes (the grid change) | yes |
| `Padding` | Pad; `mode` accepts `"constant:<value>"`, `reflect`, `replicate`, `circular`. | `padding=[0,0,0,0,0,0], mode="constant", inverse=True` | yes | yes | yes |
| `Crop` | Crop to the foreground bounding box. Must come before any stage that changes the grid. | `inverse=True` | yes | yes | yes |
| `Canonical` | Reorient to RAS (3-D). An oblique volume is resampled onto the nearest axis-aligned grid. | `inverse=True, fill=0.0` | yes | yes | yes, if not oblique |
| `Permute` | Permute spatial axes (`dims="1\|0\|2"`); the geometry follows. | `dims="1\|0\|2", inverse=True` | yes | yes | yes |
| `Flip` | Flip spatial axes; the geometry follows. | `dims="1\|0\|2", inverse=True` | no | yes | yes |
| `Squeeze` | `tensor.squeeze(dim)`. | `dim`, `inverse=True` | yes | yes | no ‡ |
| `Flatten` | Flatten to 1-D. | | yes | no | no ‡ |

`interpolation` defaults to nearest for `uint8`, `int64` and `bool` volumes and to linear otherwise: set
`interpolation: nearest` for a label map stored in another type. `inverse` undoes the grid change of a
`Resample`, not the map it applied.

## Labels and masks

| Name | Purpose | Key arguments (defaults) | Shape | Inv | Stream |
| --- | --- | --- | --- | --- | --- |
| `TensorCast` | Change the dtype. | `dtype="float32", inverse=True` | no | yes | yes |
| `OneHot` | One-hot encode a one-channel label map. | `num_classes, inverse=True` | no † | yes | yes |
| `Argmax` | `argmax(dim)`, keeping the axis. | `dim=0` | no † | no | yes over channels |
| `Softmax` | `softmax(dim)`. | `dim=0` | no | no | yes over channels |
| `FlatLabel` | Set the chosen labels (or all non-zero) to 1. | `labels=None` | no | no | yes |
| `SelectLabel` | Remap labels, given as `"(old,new)"`. | `labels` | no | no | yes |
| `Mask` | Set voxels outside a mask to `value_outside`. The mask is a group or a file. | `path="./default.mha", value_outside=0` | no | no | yes |
| `Dilate` | Binary dilation. | `dilate=1` | no | no | yes |
| `Sum` | Sum over `dim`. | `dim=0` | no † | no | yes over channels |
| `MergeLabels` | Merge the label maps of a `combine: Concat` ensemble of disjoint tasks into one (TotalSegmentator's five tasks). | | no | no | yes |
| `Gradient` | Gradient magnitude, or its components. | `per_dim=False` | no † | no | yes |

## Ensembles and uncertainty

These work on a stack of ensemble members (prediction post-processing).

| Name | Purpose | Key arguments | Shape | Stream |
| --- | --- | --- | --- | --- |
| `InferenceStack` | Write the stack of members; return their mean or median (`mode`: `mean`, `median`, `Seg`). | `dataset=None, name=None, mode="mean"` | no | no |
| `Variance`, `StandardDeviation` | Per-voxel variance or standard deviation over the members. | | no † | yes |
| `SegmentationDisagreement` | Per-voxel disagreement between segmentations. | `ignore_background=False` | no † | yes |
| `Magnitude` | Vector magnitude over the channels (a displacement field). | | no † | yes |
| `Norm` | Vector magnitude over the last axis. | | yes | no ‡ |
| `Percentage` | `tensor / baseline * 100`. | `baseline` | no | yes |

## Writing and changing the number of cases

| Name | Purpose | Key arguments (defaults) |
| --- | --- | --- |
| `Save` | Write the volume to a dataset and pass it on. Once written, later runs read from it. | `dataset=None, group=None, scale_factors=None, downsample_method=None` |
| `Write` | Like `Save`, but `dataset` is required: the output of a TRANSFORM run. | same as `Save` |
| `Reduce` | Combine every case into one volume (TRANSFORM only). | `operator="Median", output, grid="strict", grid_tolerance=1e-6` |
| `Expand` | Turn each case into `nb` copies (TRANSFORM only). | `nb=2, pattern="{name}_{a:02d}", seed=None` |
| `Statistics` | Record the volume's min, max, mean and std for the criteria that read them (`IMPACTS`, `IMPACTSynth`, `IMPACTReg`, `SAM_Perceptual`). | |
| `KonfAIInference` | Run a packaged app on the case (needs `konfai-apps`). It loads the model for every case: for a cohort, use `PREDICTION`. | |

`scale_factors` writes an OME-Zarr pyramid; label dtypes are downsampled by majority, others by mean
([OME-Zarr](storage-backends.md#multiscale-levels)). `Reduce` and `Expand` are described in the
{doc}`transform guide <../../config_guide/transform>`.

## Augmentations

Augmentations are random and applied during training (`konfai/data/augmentation/`). Each draw is made once
per case, so all the patches of a case share it within an epoch.

```yaml
Dataset:
  augmentations:
    DataAugmentation_0:
      data_augmentations:
        Flip:
          f_prob: [0, 0.5, 0.5]   # flip probability per axis
          prob: 1                 # goes under the augmentation, not beside it
      nb: 1                       # augmented copies per case
```

`PlacedMask`, `Permute` and a quarter-turn `Rotate` change the shape; the others keep the geometry.

### Spatial

| Name | Purpose | Key arguments (defaults) | Stream |
| --- | --- | --- | --- |
| `Translate` | Random shift, in voxels. | `t_min=-10, t_max=10, is_int=False` | yes |
| `Rotate` | Random rotation about the centre, in world units. | `a_min=0, a_max=360, is_quarter=False` | yes |
| `Scale` | Random isotropic scale. | `s_std=0.2` | yes |
| `Flip` | Random flip per axis; `vector_field` also negates the flipped component. | `f_prob=[0.33,0.33,0.33], vector_field=False` | yes |
| `Elastix` | Random elastic warp. | `grid_spacing=16, max_displacement=16` | yes |
| `Permute` | Random axis permutation (3-D). | `prob_permute=[0.5,0.5]` | yes |
| `PlacedMask` | Place a mask volume at random; outside is `value`. | `mask`, `value` | no |

### Intensity

| Name | Purpose | Key arguments (defaults) |
| --- | --- | --- |
| `Brightness` | Additive brightness. | `b_std` |
| `Contrast` | Multiplicative contrast. | `c_std` |
| `LumaFlip` | Invert the values. | |
| `HUE` | Rotate the hue (RGB). | `hue_max` |
| `Saturation` | Scale the saturation (RGB). | `s_std` |
| `GaussianNoise` | Additive Gaussian noise. | `std_min=0.0, std_max=0.1` |
| `GaussianBlur` | Gaussian blur; `in_plane` for a 2.5-D stack. | `sigma_min=0.5, sigma_max=1.0, in_plane=False` |
| `SimulateLowResolution` | Downsample each plane and back up. | `factor_min=1.0, factor_max=2.0` |
| `Gamma` | Gamma on the case's range. | `gamma_min=0.7, gamma_max=1.5, eps=1e-6` |
| `ContrastAroundMean` | Scale the distance to the mean. | `factor_min=0.75, factor_max=1.25` |
| `Noise` | Diffusion-style noising; its `prob` is the largest noise step. | `n_std, noise_step=1000` |
| `CutOUT` | Fill a random box with `value` (`cutout_size` is a fraction of each axis). | `cutout_size, value` |

All intensity augmentations stream.

## Next steps

- {doc}`../../config_guide/index`: where `transforms:` and `augmentations:` go in a config.
- {doc}`../../usage/custom-models`: writing your own transform or augmentation.
