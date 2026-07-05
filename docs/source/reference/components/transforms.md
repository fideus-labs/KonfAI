# Transforms

Transforms are **deterministic** pre/post-processing steps in
`konfai/data/transform.py`. They run every time (train / predict / evaluate),
either whole-image at load time (`transforms:`) or per extracted patch
(`patch_transforms:`), under a dataset group:

```yaml
Dataset:
  groups_src:
    CT:
      groups_dest:
        CT:
          transforms:
            Standardize: { }        # bare name → konfai.data.transform.Standardize
          is_input: true
```

A transform whose base class is `TransformInverse` can be run in reverse during
post-processing / reassembly when you pass `inverse: true`.

```{important}
A transform that changes the **spatial** shape must implement `transform_shape()`
correctly — patch planning depends on an exact prediction. The **Shape** column
below flags those. Transforms that only change the **channel** count (`OneHot`,
`Argmax`, `Sum`, …) intentionally do *not* override it (patching strips the
channel axis first). **Inv** flags whether a working `inverse()` exists.
```

## Intensity & normalisation

| Name | Purpose | Key args (defaults) | Shape | Inv |
| --- | --- | --- | --- | --- |
| `Clip` | Clamp intensities to a fixed or data-dependent range. | `min_value=-1024, max_value=1024, mask=None` | no | no |
| `Normalize` | Linear map to `[min,max]`; caches Min/Max. | `min_value=-1, max_value=1, channels=None, lazy=False, inverse=True` | no | **yes** |
| `UnNormalize` | Map `[-1,1] → [min,max]` (fixed). | `min_value=-1024, max_value=3071` | no | no |
| `Standardize` | Zero-mean / unit-std from cached, given, or mask-derived stats. | `mean=None, std=None, mask=None, lazy=False, inverse=True` | no | **yes** |
| `HistogramMatching` | SimpleITK histogram match to a reference group. | `reference_group` | no | no |

## Geometry & resampling

| Name | Purpose | Key args (defaults) | Shape | Inv |
| --- | --- | --- | --- | --- |
| `Padding` | `F.pad`; updates Origin. `mode` supports `"constant:<val>"`. | `padding=[0,0,0,0,0,0], mode="constant", inverse=True` | **yes** | **yes** |
| `Crop` | Crop to foreground bounding box; caches the box; updates Origin. | `inverse=True` | **yes** | **yes** (pads back) |
| `ResampleToResolution` | Resample to a target voxel spacing (per-axis `<0` = keep). | `spacing=[1,1,1], inverse=True` | **yes** | **yes** |
| `ResampleToShape` | Resample to a target shape (per-axis `0/<0` = keep). | `shape=[100,256,256], inverse=True` | **yes** | **yes** |
| `ResampleTransform` | Warp by stored SimpleITK transforms read from the dataset. | `transforms`, `inverse=True` | no | no |
| `Canonical` | Reorient to canonical direction (3-D); updates Origin/Direction. | `inverse=True` | no | **yes** |
| `Permute` | Permute spatial axes. `dims` is a pipe-separated axis list. | `dims="1\|0\|2", inverse=True` | **yes** | **yes** |
| `Flip` | Flip spatial axes. | `dims="1\|0\|2", inverse=True` | no | **yes** (self-inverse) |
| `Squeeze` | `tensor.squeeze(dim)`. Does not override `transform_shape` — use on the channel axis or in post-processing. | `dim` (required), `inverse=True` | no | **yes** |
| `Flatten` | Flatten to 1-D. | — | **yes** | no |

## Labels & masks

| Name | Purpose | Key args (defaults) | Shape | Inv |
| --- | --- | --- | --- | --- |
| `TensorCast` | Cast dtype; caches original for inverse. | `dtype="float32", inverse=True` | no | **yes** |
| `OneHot` | One-hot encode a label map (changes **channels**). | `num_classes, inverse=True` | no† | **yes** |
| `Argmax` | `argmax(dim)` then unsqueeze. | `dim=0` | no† | no |
| `Softmax` | `softmax(dim)`. | `dim=0` | no | no |
| `FlatLabel` | Binarise selected labels (else `>0`) → 1. | `labels=None` | no | no |
| `SelectLabel` | Remap labels; entries are `"(old,new)"` strings. | `labels` | no | no |
| `Mask` | Set voxels where mask==0 to `value_outside`. | `path="./default.mha", value_outside=0` | no | no |
| `Dilate` | Binary dilation via max-pool (2D/3D). | `dilate=1` | no | no |
| `Sum` | Sum over `dim` (merges multi-model label maps). | `dim=0` | no† | no |
| `Gradient` | Gradient-magnitude image (or components). | `per_dim=False` | no† | no |

## Ensemble / uncertainty post-processing

Operate on a stacked `[N, …]` ensemble axis (prediction post-processing).

| Name | Purpose | Key args | Shape |
| --- | --- | --- | --- |
| `InferenceStack` | Aggregate an ensemble stack (mean / median / seg-argmax); writes an `InferenceStack` volume. | `dataset, name, mode="mean"` | no |
| `Norm` | Vector magnitude over the trailing axis (drops it). | — | **yes** |
| `Variance` | Per-voxel variance over N. | — | no† |
| `StandardDeviation` | Per-voxel std over N. | — | no† |
| `SegmentationDisagreement` | Per-voxel label disagreement across N segmentations. | `ignore_background=False` | no† |
| `Percentage` | `tensor / baseline * 100`. | `baseline` | no |

## Side-effect & advanced

| Name | Purpose |
| --- | --- |
| `Statistics` | Records ImageMin/Max/Mean/Std to the attribute cache and returns the tensor unchanged (feeds the perceptual criteria `SAM_Perceptual`, `IMPACTSynth`, `IMPACTReg`). Order in the transform list matters. |
| `Save` | Marker — a no-op passthrough (a checkpoint hint, not a saver). |
| `KonfAIInference` | Run a nested KonfAI app inference in a spawned subprocess. Needs `konfai-apps` and `num_workers: 0`; defaults to a specific HF repo. |

`†` changes the **channel** dimension, not spatial — no `transform_shape`
override needed.

## Next steps

- {doc}`augmentations` — the random, train-time counterpart
- {doc}`../../concepts/datasets` — where transforms sit in a dataset config
- {doc}`../api/extension-points` — write your own `Transform` (implement
  `__call__` **and** `transform_shape()`)
