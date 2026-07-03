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

## Spatial (Euler transforms)

Reversible affine warps via `grid_sample` (nearest-neighbour for label tensors).

| Name | Purpose | Key args (defaults) | Shape | Inv |
| --- | --- | --- | --- | --- |
| `Translate` | Random translation (voxels). | `t_min=-10, t_max=10, is_int=False` | no | **yes** |
| `Rotate` | Random rotation (degrees). | `a_min=0, a_max=360, is_quarter=False` | no | **yes** |
| `Scale` | Random log2-normal isotropic scale. | `s_std=0.2` | no | **yes** |
| `Flip` | Per-axis random flip; optional vector-field channel negation. | `f_prob=[0.33,0.33,0.33], vector_field=False` | no | **yes** (self-inverse) |
| `Elastix` | Random BSpline elastic warp (SimpleITK). | `grid_spacing=16, max_displacement=16` | no | no |
| `Permute` | Random spatial-axis permutation (**3-D only**). | `prob_permute=[0.5,0.5]` | **yes** | **yes** |
| `Mask` | Randomly place a mask volume; outside → `value` (SimpleITK). | `mask` (required), `value` (required) | **yes** | no |

## Intensity (colour transforms)

Apply a per-index colour affine to RGB(3-ch) or L(1-ch) tensors. Their inverse is
a no-op (colour changes don't move voxels).

| Name | Purpose | Key args (defaults) |
| --- | --- | --- |
| `Brightness` | Additive brightness. | `b_std` (required) |
| `Contrast` | Multiplicative contrast (log2-normal). | `c_std` (required) |
| `LumaFlip` | Random luma (value) inversion. | — |
| `HUE` | Random hue rotation. | `hue_max` (required) |
| `Saturation` | Random saturation scale. | `s_std` (required) |

## Other

| Name | Purpose | Key args (defaults) | Notes |
| --- | --- | --- | --- |
| `Noise` | Diffusion-style forward noising (zero-terminal-SNR β schedule). | `n_std` (required), `noise_step=1000` | Its `prob` is the max noise timestep, not an apply probability; it always applies. |
| `CutOUT` | Random cutout box filled with `value`. | `c_prob`, `cutout_size`, `value` (all required) | Gating uses the base probability. |

```{note}
`Elastix` and `Mask` require SimpleITK. The `vector_field` flag on `Flip` should
only be enabled for single-channel or genuine vector-field groups.
```

## See also

- {doc}`transforms` — deterministic preprocessing (runs every workflow)
- {doc}`../../concepts/datasets`
- {doc}`../api/extension-points` — write your own `DataAugmentation`
