# Tuning a preset

Each preset is a complete recipe; start from one without changes. When a registration needs adjusting, `--set` changes
one parameter of it for one run, and `show NAME` lists every parameter the preset takes, with its value, its range or
choices, and what it does:

```bash
impact-reg-konfai show FireANTs_SyN
impact-reg-konfai register FireANTs_SyN -f fixed.nii.gz -m moving.nii.gz --gpu 0 \
    --set deformable_iterations=[400,200,100] --set smooth_warp_sigma=1.0
```

`--set NAME=VALUE` applies to every preset of the run; `--set PRESET:NAME=VALUE` to one only, since each engine names
its parameters its own way. Every override is checked against every preset before anything runs, and a misspelled name
is refused with the closest matches. A nested parameter takes its full path from the config root.

## What to change first

| You see | Try first | Parameter, by engine |
|---|---|---|
| the images start far apart (centimetres) | a preset with a rigid stage; for FireANTs a larger linear step | FireANTs `affine_lr` (millimetres per step: 0.03 moves up to 10 mm per stage) |
| the alignment is right globally but local structures are off | more deformable iterations | FireANTs `deformable_iterations`, elastix `max_iterations`, ConvexAdam `iterations` |
| the deformation folds or looks noisy | more regularisation | FireANTs `smooth_warp_sigma`, `smooth_grad_sigma`; ConvexAdam `regularization_weight`; elastix `final_grid_spacing` (larger is smoother) |
| the deformation is too stiff | less regularisation, a finer grid | the same parameters the other way |
| the registration is pulled by structures outside the region of interest, or by what only one image shows (a CT's table or head rest, a CT past the MRI's field of view) | a mask | `--fixed-mask`, `--moving-mask` |
| multimodal pair, intensity presets fail | an IMPACT preset | see [Choosing a preset](presets.md) |

## Parameters by concept

The names below are those of the shipped presets; `show NAME` gives each preset's full list.

| Concept | elastix (`Generic_*`, `Elastix_IMPACT_*`) | FireANTs (`FireANTs_*`) | ConvexAdam (`ConvexAdam_*`) |
|---|---|---|---|
| Transform model | `parameter_maps` (rigid, B-spline) | `linear_method` (rigid_affine, rigid, none), `deformable_method` (syn, greedy, none) | `stages` (coarse, fine), `linear` |
| Optimisation | `max_iterations` | `affine_iterations`, `deformable_iterations`, `affine_lr`, `deformable_lr` | `iterations`, `learning_rate` |
| Multi-resolution | the parameter map's resolutions, or `levels` (IMPACT presets: per level, the iterations and the models) | `scales` (downsampling per level), `levels` (the models per scale) | `grid_spacing`, `displacement_half_width` (coarse search), `grid_shrink`, `levels` (the models of the coarse and fine stages) |
| Metric | intensity: the parameter map; IMPACT: the `models` of each level | `affine_metric`, `deformable_metric`, `cc_kernel`, `deformable_masked` | the distance of `models`, `balance_coarse_layers` (each coarse-search layer divided by its spread over the candidate displacements, times `layers_weight`) |
| Features | the IMPACT loss (below) | the IMPACT loss (below); `feature_patch`, `feature_chunk`, `feature_overlap` (Static extraction) | the IMPACT loss (below) |
| Regularisation | `final_grid_spacing` | `smooth_warp_sigma`, `smooth_grad_sigma` | `regularization_weight`, `control_grid_smoothing` |
| Initialisation | the rigid map | `moments_init` (cof, com, none) | `linear`, `linear_iterations`, `linear_sampling` |
| Sampling | `spatial_samples` | | |
| Random draws | `seed` | `seed` | `seed` |
| Escape hatch | `parameter_overrides` (any elastix parameter) | | |

Memory is not a preset parameter: runs are sized automatically, and `--max-voxels` overrides it (see
[Large images](large-images.md)). The device is `--gpu` or `--cpu`.

## Feature layers

`layers_mask` is a string of 0 and 1 over the network's outputs, shallow to deep: `'01'` requests the first two layers
and keeps the second. For the TotalSegmentator models, the early layers keep texture and suit CT/CBCT pairs, the last
layers carry organ-level information and suit MR/CT pairs. A mask longer than the model's outputs is refused.

## The IMPACT loss

The three engines read the same IMPACT settings, with the same meaning.

Each model under `models` (or under each entry of `levels`, which replaces `models` level by level):

| Setting | Meaning |
|---|---|
| `ref` | the feature model, `repo:file` on Hugging Face or a local TorchScript file |
| `layers_mask` | the layers kept, see above |
| `layers_weight` | one weight for every kept layer, or one per kept layer |
| `distance` | `L1`, `L2` (default), `Dice`, `Cosine`, `L1Cosine`, `NCC`, `LNCC`; Dice compares raw features and can be negative |
| `pca` | principal components each kept layer is reduced to, fitted on the fixed image (0 keeps every channel) |
| `subset_features` | channels of each kept layer drawn at random at every iteration (0 compares all of them) |
| `voxel_size` | the resolution (mm) the image is resampled to before the model sees it; left out, the image as it is |
| `feature_normalization` | `none`, `l2` or `standardized`, each voxel's feature vector, before the PCA |

And for the loss as a whole: `mode` (`Static`: features extracted once, then warped; `Jacobian`: the network inside the
loss), `normalize` (each layer divided by its value when a level starts, so every layer starts at 1 and the weights are
shares; on by default; ConvexAdam's coarse search keeps its raw cost, its coupling schedule being absolute),
`feature_map_update_interval` (Static: extract the moving features again every that many iterations), `lncc_kernel` (the LNCC window) and `mixed_precision` (the models in float16).

A setting an engine cannot honour is refused before the run. The one left: `LNCC` correlates windows of a dense
feature map, so it is refused wherever the loss is scored on random points (elastix, or `voxel_sampling` below 1).

## Making it permanent

A set of overrides that works for a cohort is a new preset: copy the preset's folder from the cache (or from
[`VBoussot/ImpactReg`](https://huggingface.co/VBoussot/ImpactReg)), edit its `Prediction.yml`, and point
`KONFAI_IMPACTREG_REPO` at the directory holding it.
