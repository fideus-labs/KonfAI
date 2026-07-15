# Augmentations

Augmentations are **random, train-time** data augmentations in
`konfai/data/augmentation.py` (base class `DataAugmentation`). Unlike transforms,
they are sampled per case and applied only during training:

```yaml
Dataset:
  augmentations:
    DataAugmentation_0:
      data_augmentations:
        Flip:                     # bare name → konfai.data.augmentation.Flip
          f_prob: [0, 0.5, 0.5]   # per-axis flip probability
        prob: 1
      nb: 1                       # number of augmented copies per sample
```

```{note}
**Lifecycle.** Parameters are drawn **once per case index** and cached, so every
patch of a case shares the same draw within an epoch (mandatory for patch
consistency). `reset_state` re-samples each epoch. A subclass implements
`_state_init` (sample params) and `_compute` (apply); `_inverse` supports
test-time augmentation reassembly.
```

```{important}
**Only `Mask` and `Permute` may change spatial shape** (verified against the
code). Everything else preserves geometry.
```

```{note}
**Patch streaming.** The **Stream** column says whether a copy's patches can be
cut straight from disk or whether the augmentation needs the case loaded whole
(see {doc}`transforms`). An augmentation answers per **copy**, not per
transform: the answer follows the draw, so two copies of the same case can differ
and the same copy can differ next epoch. A copy the draw did not select is the
identity, which streams. The draw is planned as part of the group's chain, so
a region augmentation and a region transform (`Dilate`, `Canonical`, a resampler)
in the same chain load the volume whole — only one of them can shape the read. An
augmentation you write yourself starts at the whole volume and streams nothing
until it declares otherwise.
```

## Spatial (Euler transforms)

Reversible affine warps via `grid_sample` (nearest-neighbour for label tensors).

| Name | Purpose | Key args (defaults) | Shape | Inv | Stream |
| --- | --- | --- | --- | --- | --- |
| `Translate` | Random translation (voxels). | `t_min=-10, t_max=10, is_int=False` | no | **yes** | **yes** — a halo of the drawn shift (plus a voxel for interpolation), while that stays within half the patch |
| `Rotate` | Random rotation (degrees). | `a_min=0, a_max=360, is_quarter=False` | no | **yes** | **yes** with `is_quarter: true` (an index remap); no for a free angle — the shift grows with distance from the centre |
| `Scale` | Random log2-normal isotropic scale. | `s_std=0.2` | no | **yes** | no — the shift grows with distance from the centre, so no fixed halo covers it |
| `Flip` | Per-axis random flip; optional vector-field channel negation. | `f_prob=[0.33,0.33,0.33], vector_field=False` | no | **yes** (self-inverse) | **yes** — index remap; no with `vector_field: true` (negating a channel changes values) |
| `Elastix` | Random BSpline elastic warp (SimpleITK). | `grid_spacing=16, max_displacement=16` | no | no | no — the displacement field is built at the full shape and indexed by absolute position |
| `Permute` | Random spatial-axis permutation (**3-D only**). | `prob_permute=[0.5,0.5]` | **yes** | **yes** | **yes** — index remap |
| `Mask` | Randomly place a mask volume; outside → `value` (SimpleITK). | `mask` (required), `value` (required) | **yes** | no | no — the output grid is the mask's, and the mask is already resident |

## Intensity (colour transforms)

Apply a per-index colour affine to RGB(3-ch) or L(1-ch) tensors. Their inverse is
a no-op (colour changes don't move voxels). The draw is a colour matrix applied to
each voxel on its own — no neighbour, no coordinate, no extent — so every one of
them streams: a voxel comes out the same whatever region it was read in.

| Name | Purpose | Key args (defaults) |
| --- | --- | --- |
| `Brightness` | Additive brightness. | `b_std` (required) |
| `Contrast` | Multiplicative contrast (log2-normal). | `c_std` (required) |
| `LumaFlip` | Random luma (value) inversion. | — |
| `HUE` | Random hue rotation. | `hue_max` (required) |
| `Saturation` | Random saturation scale. | `s_std` (required) |

## Other

| Name | Purpose | Key args (defaults) | Notes | Stream |
| --- | --- | --- | --- | --- |
| `Noise` | Diffusion-style forward noising (zero-terminal-SNR β schedule). | `n_std` (required), `noise_step=1000` | Its `prob` is the max noise timestep, not an apply probability; it always applies. | no — the field is drawn per call, so overlapping patches would draw unrelated fields and the blend would suppress the variance |
| `CutOUT` | Random cutout box filled with `value`. | `c_prob`, `cutout_size`, `value` (all required) | Gating uses the base probability. | no — the box is normalised to the tensor in hand, so per patch it would land in every patch |

```{note}
`Elastix` and `Mask` require SimpleITK. The `vector_field` flag on `Flip` should
only be enabled for single-channel or genuine vector-field groups.
```

## Next steps

- {doc}`transforms` — deterministic preprocessing (runs every workflow)
- {doc}`../../concepts/datasets` — where the `augmentations:` block sits in a config
- {doc}`../api/extension-points` — write your own `DataAugmentation`
