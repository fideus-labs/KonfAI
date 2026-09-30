---
name: impact-reg
description: >-
  Register medical (or microscopy) images with IMPACT-Reg (the impact-reg-konfai CLI): choose a preset for the pair,
  run it at native resolution whatever the image size, check the result, and tune it when it falls short. Knows the
  presets of VBoussot/ImpactReg (elastix-IMPACT, FireANTs, ConvexAdam), which feature layers suit which modality pair,
  the transform conventions, how large volumes are tiled, and how to evaluate (MAE, Dice, TRE, Jacobian, ensemble
  spread). Use when the user wants to register a pair or a cohort, align MR/CT or CBCT/CT, pick registration
  hyperparameters, register a volume too large for the GPU, evaluate or compare registrations, or warp segmentations
  through a transform. Triggers: "register these images", "align MR to CT", "CBCT/CT registration", "which preset",
  "tune the registration", "registration is off", "large volume registration", "impact-reg-konfai", "IMPACT".
---

# Registering images with IMPACT-Reg

`impact-reg-konfai` runs registration **presets**: complete recipes (engine, stages, metric, feature models) published
on Hugging Face `VBoussot/ImpactReg`. Work from a preset; change it only where a measurement says so. User-facing docs:
`apps/impact_reg/docs/` in the KonfAI repository ([how it works](../../../apps/impact_reg/docs/how-it-works.md),
[presets](../../../apps/impact_reg/docs/presets.md), [parameters](../../../apps/impact_reg/docs/parameters.md),
[large images](../../../apps/impact_reg/docs/large-images.md), [evaluation](../../../apps/impact_reg/docs/evaluation.md)).

## The loop

1. **Look at the pair first.** Modalities, voxel sizes, field of view, how far apart the images start (centimetres
   or millimetres), whether masks or labels exist. `impact-reg-konfai presets` lists the presets; `presets NAME` says
   what one runs, needs (GPU, models, installs), costs per voxel and tunes.
2. **Give the fixed image a body mask** (`--fixed-mask`) whenever it shows what the other image does not: the table
   or head rest of a CT, a CT that extends past the MRI's field of view. A deformable stage otherwise pulls the moving
   image onto them. On the tutorial MR/CT pair it took `Elastix_IMPACT_Static` from 4.0 to 2.1 mm TRE (Dice 0.88 to
   0.91), `FireANTs_SyN` from 5.2 to 3.2 mm and `FireANTs_Anatomix` from 3.2 to 1.8 mm. ConvexAdam ignores masks
   (itk-impact has no mask API).
3. **Pick the preset from the pair** (below), always with `--gpu` when a GPU exists.
4. **Register**, then **measure before trusting**: `eval` with whatever ground truth exists (labels: Dice; landmarks:
   TRE; same modality: MAE), always against the same metrics without `--transform` (the starting misalignment). A
   deformable result must not fold: `eval` reports the Jacobian's folded fraction for field transforms.
5. **Tune one thing at a time** with `--set`, re-measure, keep what helps (see *Tuning order*).
6. **Several good presets?** Ensemble them (`register A B C --keep-fields`) and map where they disagree
   (`uncertainty`): high spread marks where the registration should not be trusted.

## Choosing a preset

| Pair | Start with | Then |
|---|---|---|
| any pair, far apart | `Generic_Rigid` (seconds, CPU) | a deformable preset; every shipped deformable preset has its own rigid or affine stage |
| CT/CT, MR/MR (same contrast) | `ConvexAdam_Composite` (lungs: TRE 2.1 mm in 40 s), `FireANTs_SyN` (brains) | `FireANTs_Anatomix` (the best on the lungs, five times slower); `Elastix_IMPACT_Static` between two patients; `Generic_Rigid_BSpline` on the CPU |
| MR/CT | `Elastix_IMPACT_Static` (deep TotalSegmentator features + MIND) with a body mask on the CT: best on both MR/CT sets | `FireANTs_IMPACT_MRCT` (TS/M730 layer 7 + MIND); `ConvexAdam_IMPACT_MRCT` on the abdomen; not `FireANTs_SyN` or `FireANTs_IMPACT`, both below the unregistered pair on head and neck MR/CT |
| CT/CBCT | `FireANTs_SyN` | `FireANTs_IMPACT` (early TotalSegmentator CT layers): at most 0.01 more Dice, 2.5 to 6.5 times slower; `ConvexAdam_IMPACT_CBCT` (TS/M730 layer 2) in half a minute |
| microscopy, non-medical | `FireANTs_SyN` | `ConvexAdam_Composite`, `Generic_Rigid_BSpline`; not the TotalSegmentator-based presets, made for human CT/MR anatomy at millimetre scale |

Intensity metrics (FireANTs' local correlation, mutual information) compare grey values; across modalities they
compare values that do not correspond. IMPACT compares network features instead. Every mutual-information stage
clamps each image's extreme voxels first (0.01-99.99 percentiles; FireANTs 0.5-99.5), as does ConvexAdam before MIND:
on two raw ExaSPIM light-sheet brains, lone voxels at 21,668 beside tissue under 33 had put the whole brain in one
histogram bin (brain Dice after `Generic_Rigid` 0.742 against 0.879 as they came; 0.916 clamped). The rigid stage
of an elastix IMPACT preset keeps the raw intensities its models need: clamp such images before an IMPACT preset.
elastix samples its metric at random: a single `Generic_*` run can land far from another on a hard pair
(AbdomenMRCT case 2: Dice 0.22 or 0.56 with the seed alone); compare presets over several cases, not one run.

## Feature layers (the IMPACT guideline)

- **CT/CBCT: early layers**, TotalSegmentator MR (`TS/M730`) layer 2 (`layers_mask '01'`): same tissue contrast,
  degraded by noise and artefacts; texture and edges carry the alignment. On the benchmark's three CBCT/CT sets they matched FireANTs'
  local correlation within 0.01 Dice: a CBCT keeps the CT's contrast, so start with the intensity preset and move to
  the early layers when streaks or scatter defeat it.
- **MR/CT: deep, organ-level layers of a segmentation network**, `TS/M730` layer 7 (`'0000001'`), plus MIND for detail
  inside organs. Measured in all three engines on AbdomenMRCT and head and neck MR/CT: FireANTs 0.562 -> 0.602 and
  0.674 -> 0.684 against anatomix + MIND, ConvexAdam 0.543 -> 0.627 on the abdomen against MIND alone (0.659 -> 0.630 on
  head and neck). Compare these layers with a bounded distance (soft Dice) or normalised features: in L2 their
  activations outweighed ConvexAdam's regularisation and folded 16 % of the voxels. They need the whole image around
  them: on native tiles ConvexAdam lost 0.065, which is why the presets built on them register a large pair on a
  coarse copy only. The segmentation head alone has a short capture range: a pair that starts far apart falls into the
  wrong organ. `Elastix_IMPACT_Static` compares the last decoder layer (`0000001`) at its coarse levels and the head
  (`00000001`) at its fine ones; the head at every level lost AbdomenMRCT case_0002 (Dice 0.28 against 0.65).
- `TS/M730` serves both guidelines. The one measured exception: in FireANTs, CT/CBCT layer 2 of TotalSegmentator CT
  (`TS/M291`) beat `TS/M730` on head and neck (0.800 against 0.736) and matched it elsewhere, so `FireANTs_IMPACT`
  keeps `TS/M291`.
- `layers_mask` is positional over the network's outputs, shallow to deep. Its path depends on the engine, and
  `impact-reg-konfai presets NAME` prints every one: `Predictor.Model.RegistrationNet.models.N.layers_mask` for FireANTs
  and ConvexAdam, `Predictor.Model.RegistrationNet.resolutions.R.models.N.layers_mask` for elastix (one per level).

## Large images: coarse globally, then native tiles

Never answer "too large" by downsampling the pair. `register` handles it: when a pair exceeds what the preset
registers whole in the free memory (its `app.json` cost per voxel), it runs the whole preset once on a coarse copy
(from the OME-Zarr pyramid when there is one), pre-warps the moving image onto the native grid, refines with the
preset's deformable stage on native tiles, and composes the fields. Presets without a local stage stop at the global
pass. Levers: `--max-voxels` (whole/tiled threshold), `--patch-size` (tile), `--tmp-dir` (intermediates: ~60 bytes a
voxel at the peak, on disk), streamable formats (`.mha`, `.nii`, OME-Zarr). Pin `--max-voxels` and `--patch-size` for
reproducibility: the automatic plan follows the free memory. Keep `TMPDIR` short (PyTorch socket path limit).

## Tuning order

1. **Start far apart?** Make sure the preset has a rigid stage (all shipped ones do); for FireANTs the linear step is
   `affine_lr` in millimetres per iteration.
2. **Globally right, locally off:** more deformable iterations (`deformable_iterations`, `max_iterations`,
   `iterations`).
3. **Folding or noise:** more regularisation (`smooth_warp_sigma`, `regularization_weight`, a larger
   `final_grid_spacing`).
4. **Pulled by irrelevant structures:** `--fixed-mask` / `--moving-mask` (elastix and FireANTs; not ConvexAdam).
5. **Multimodal and still off:** change the features (model, layers), not the optimiser.

`--set NAME=VALUE` goes to every preset of a run, `--set PRESET:NAME=VALUE` to one; every name is checked before
anything runs. A setting that works for a cohort belongs in a copied preset folder (`KONFAI_IMPACTREG_REPO=<dir>`).

## Conventions to get right

- `Transform.h5` is an ITK displacement field on the **fixed** grid mapping a **fixed** point to its **moving**
  partner, physical components (mm). Moved = moving resampled through it. Landmarks: fixed pushed through the
  transform land on the moving ones.
- `eval` and `apply` take the **original** moving data; they apply the transform themselves. `apply --labels` for
  segmentations (nearest neighbour). There is no inverse: register the other way round for the other direction.
- Ensembles average fields voxel by voxel: ensemble presets of the same nature (not a rigid with deformables).
- `register.json` beside the outputs records the inputs of each case (`P000`, ... in input order), versions,
  overrides and runtime.
- In KonfAI Studio (konfai-mcp), `run_app_infer` on a registration app runs this same `register` (Transform.h5, Moved,
  register.json) and `run_registration_evaluate` is `eval`; the apps' own `run_app_evaluate` is refused, since their
  bundled configs read the moving data without the transform.
