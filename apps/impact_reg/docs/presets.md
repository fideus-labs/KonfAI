# The presets

A preset is a complete registration recipe published in [`VBoussot/ImpactReg`](https://huggingface.co/VBoussot/ImpactReg):
an engine, its stages, the similarity each stage scores, and the settings measured to work. `impact-reg-konfai list`
lists them; `show NAME` prints what one runs, needs and tunes, with every parameter's path for `--set`.

## What each one runs

| Preset | Engine | Stages | Similarity | Masks | A pair too large for the device |
|---|---|---|---|---|---|
| `Elastix#rigid` | elastix | rigid | mutual information | yes | whole preset on a coarse copy |
| `Elastix#rigid_bspline` | elastix | rigid, B-spline (16 mm grid) | mutual information | yes | coarse copy, then B-spline on native tiles |
| `Elastix#impact_static` | elastix-IMPACT | rigid (MI), B-spline (20 mm) | IMPACT: MIND (L1) and TotalSegmentator MR `TS/M730` (soft Dice), last decoder layer at the two coarse levels, segmentation head at the two fine ones; features extracted once (Static); plus MI | yes | coarse copy only |
| `Elastix#impact_jacobian` | elastix-IMPACT | rigid (MI), B-spline (10 mm) | IMPACT: early `TS/M730` layers (`'01'`, L1), differentiated through the network (Jacobian) | yes | coarse copy only |
| `ConvexAdam#coarse` | itk-impact | affine (MI), coupled-convex coarse search | IMPACT on MIND | yes | coarse copy only |
| `ConvexAdam#composite` | itk-impact | affine (MI), coarse search, Adam refinement (80 steps) | IMPACT on MIND (L2) | yes | coarse copy, then the refinement on native tiles |
| `ConvexAdam#impact_cbct` | itk-impact | affine (MI), coarse search, Adam refinement | IMPACT on TotalSegmentator MR `TS/M730` layer 2 (`'01'`, L2) | yes | coarse copy, then the refinement on native tiles |
| `ConvexAdam#impact_mrct` | itk-impact | affine (MI), coarse search, Adam refinement (regularisation 5) | IMPACT on `TS/M730` layer 7 (`'0000001'`, soft Dice) and MIND (L2) | yes | coarse copy only |
| `FireANTs#syn` | FireANTs | rigid, affine (MI), SyN | local correlation | yes | coarse copy, then SyN on native tiles |
| `FireANTs#impact` | FireANTs | rigid, affine (MI), SyN | IMPACT: early layers of TotalSegmentator CT `TS/M291` (`'01'`, L1) | yes | coarse copy, then SyN on native tiles |
| `FireANTs#anatomix` | FireANTs | rigid, affine (MI), SyN (step 0.5, warp smoothed at 1 voxel) | IMPACT on anatomix and MIND, extracted once and reduced to their main components | yes | coarse copy, then SyN on native tiles |
| `FireANTs#impact_mrct` | FireANTs | rigid, affine (MI), SyN (step 0.5, warp smoothed at 1 voxel) | IMPACT on `TS/M730` layer 7 (`'0000001'`) and MIND, extracted once and reduced to their main components | yes | coarse copy, then SyN on native tiles |

Every deformable preset starts with its own rigid or affine stage, so a pair that starts centimetres apart needs no
separate rigid run. Masks restrict the similarity to a region (`--fixed-mask`, `--moving-mask`), in every engine. The
last column is how `register` handles a pair larger than what the preset registers whole in the memory free: see
[Large images](large-images.md).

## Which one for which pair

| Pair | First choice | Also good | Avoid |
|---|---|---|---|
| any pair that starts far apart | the deformable preset for the pair: each aligns rigidly first | `Elastix#rigid` alone when the anatomy is rigid (head, a bone) | |
| MR/CT | `Elastix#impact_static` with a body mask on the CT | `FireANTs#impact_mrct`; `ConvexAdam#impact_mrct` on the abdomen | `FireANTs#syn`, whose local correlation compares grey values that do not correspond across modalities, and `FireANTs#impact`, whose early CT layers do not carry over: on head and neck MR/CT both ended below the pair as it came (Dice 0.62 and 0.50 against 0.64) |
| CT/CBCT | `FireANTs#syn` | `FireANTs#impact`: at most 0.01 more Dice, 2.5 to 6.5 times slower, and 22 GB of GPU memory; `ConvexAdam#impact_cbct` in half a minute | `ConvexAdam#coarse` alone: on two of the three sets it ends at or below the pair as it came |
| CT/CT | `ConvexAdam#composite`: within 0.01 Dice and 0.1 mm of the best on the lungs, in 40 s and 6 GB | `FireANTs#anatomix`, the best on the lungs, five times slower; `Elastix#impact_static` between two patients | `Elastix#rigid` alone on breathing lungs (landmark error 15.3 mm, 12.8 as they came) |
| MR/MR | `ConvexAdam#composite` or `FireANTs#syn`, level on the brain | `FireANTs#anatomix` | |
| microscopy (ExaSPIM light-sheet) | `FireANTs#syn` | `ConvexAdam#composite`, `Elastix#rigid_bspline` | the TotalSegmentator presets, made for human anatomy at millimetre scale |

**The feature layers follow the IMPACT guideline**: TotalSegmentator MR (`TS/M730`) layer 2 for CT/CBCT, layer 7 with
MIND for MR/CT, in each engine. For MR/CT it holds in all three: layer 7 with MIND took FireANTs from 0.562 to 0.602
Dice on AbdomenMRCT and 0.674 to 0.684 on head and neck, and ConvexAdam from 0.543 to 0.627 on the abdomen (0.659 to
0.630 on head and neck). `Elastix#impact_static` goes one layer further at its two fine levels, the segmentation head,
which beat layer 7 everywhere on both sets (0.772 and 0.693 against 0.731 and 0.663). For CT/CBCT, layer 2 lifted
ConvexAdam by 0.005 to 0.021; in FireANTs it matched TotalSegmentator CT (`TS/M291`) on the thorax and the abdomen and
lost on head and neck (0.736 against 0.800), so `FireANTs#impact` keeps `TS/M291`. The deep layers need the whole
image around them: `ConvexAdam#impact_mrct`, like the elastix IMPACT presets, registers a pair too large for the device
on a coarse copy only (on native tiles it fell from 0.641 to 0.576 on head and neck).

**The settings were chosen for general use**, on cases the benchmark below does not report: `Elastix#impact_static`'s
20 mm grid halves the folded voxels of its 14 mm one at the same MR/CT Dice, and `FireANTs#anatomix`'s smaller step and
smoother warp keep its field from folding (4.5 % of the voxels on abdominal MR/CT before, 0.06 % after) at the same
accuracy. `ConvexAdam#composite` keeps the regularisation that suits CT/CT, its pair; on another pair,
`--set regularization_weight=2.5` trades a little Dice for a smoother field (three to six times fewer folded voxels for
about 0.02 less Dice, measured with itk-impact 0.1.1).

**The intensity stages read clamped intensities.** Each image is clamped to its 0.01 and 99.99 percentiles before a
mutual-information stage bins it, and before ConvexAdam's MIND divides it by its range, so that a few extreme voxels do
not squeeze the tissue into one bin. On two raw ExaSPIM brains, whose tissue lies under 33 beside lone voxels at 21,668,
that took `Elastix#rigid` from a brain Dice of 0.742 (0.879 before registration) to 0.916 and `ConvexAdam#composite`
from 0.220 to 0.956. Not ANTs' 0.5 and 99.5: over a volume mostly air, the 99.5th percentile of an abdominal CBCT is
-12 HU, and clamping there flattens every tissue. FireANTs' linear stages keep ANTs' clamp; the elastix IMPACT presets
keep the raw intensities, which their feature models need.

**Give the CT a body mask** whenever it shows what the other image does not (a table, a head rest, a field of view past
the MRI's): a deformable stage otherwise pulls the moving image onto them. On the tutorial MR/CT pair it took
`Elastix#impact_static` from 4.0 to 2.1 mm landmark error, `FireANTs#syn` from 5.2 to 3.2 mm and
`FireANTs#anatomix` from 3.2 to 1.8 mm.

## Measured

Nine public datasets from RegistrationBenchmarks, each preset with its published settings, one to three cases a
dataset, on an NVIDIA RTX PRO 5000 Blackwell Laptop GPU (24 GB). **Dice** is the mean over the labelled structures, the
moving labels warped through the transform; **TRE** the distance between the dataset's landmarks (ThoraxCBCT's automatic
keypoints are left out: no method improves them, even where the Dice does); **Folded** the share of voxels whose
Jacobian determinant is negative; **Time** the whole `register` call, start-up and writing included; **VRAM** the
peak GPU memory of its processes, PyTorch's cache included. The best Dice of each dataset (the best TRE for BraTSReg) is
in bold. The `Elastix#impact_*` rows ran an elastix-IMPACT build with the Static fix that release 1.1.0 carries; the
`ConvexAdam#*` rows ran in a fresh environment, as pip resolved the presets' requirements then (itk-impact 0.1.5). The
three presets on TotalSegmentator MR features are measured on the MR/CT and CBCT/CT sets they are made for.

Each row is one run. elastix samples its metric at random, and the draw matters: `Elastix#rigid_bspline`'s AbdomenMRCT
case 2 lands at a Dice of 0.22 or of 0.56 depending on the seed alone. A run is reproducible for a given `seed`, which
every preset sets.

**MR/CT, abdomen** (AbdomenMRCT, 3 cases; before: Dice 0.38)

| Preset | Dice | TRE (mm) | Folded | Time | VRAM |
|---|---|---|---|---|---|
| `Elastix#rigid` | 0.337 |  | 0.00 % | 6 s | 1.8 GB |
| `Elastix#rigid_bspline` | 0.592 |  | 3.85 % | 13 s | 1.8 GB |
| **`Elastix#impact_static`** | 0.772 |  | 2.38 % | 98 s | 8.2 GB |
| `Elastix#impact_jacobian` | 0.448 |  | 0.28 % | 312 s | 2.2 GB |
| `ConvexAdam#coarse` | 0.539 |  | 0.02 % | 23 s | 1.8 GB |
| `ConvexAdam#composite` | 0.543 |  | 1.11 % | 23 s | 2.0 GB |
| `ConvexAdam#impact_mrct` | 0.627 |  | 0.06 % | 27 s | 7.9 GB |
| `FireANTs#syn` | 0.451 |  | 0.22 % | 39 s | 6.4 GB |
| `FireANTs#impact` | 0.462 |  | 0.02 % | 139 s | 16.2 GB |
| `FireANTs#anatomix` | 0.562 |  | 0.02 % | 66 s | 10.7 GB |
| `FireANTs#impact_mrct` | 0.602 |  | 0.02 % | 66 s | 9.9 GB |

**MR/CT, head and neck** (SynthRAD2025_HN_MRCT, 1 case; before: Dice 0.64)

| Preset | Dice | TRE (mm) | Folded | Time | VRAM |
|---|---|---|---|---|---|
| `Elastix#rigid` | 0.633 |  | 0.00 % | 8 s | 5.6 GB |
| `Elastix#rigid_bspline` | 0.658 |  | 0.00 % | 16 s | 5.6 GB |
| **`Elastix#impact_static`** | 0.693 |  | 0.00 % | 114 s | 12.8 GB |
| `Elastix#impact_jacobian` | 0.688 |  | 0.00 % | 308 s | 3.6 GB |
| `ConvexAdam#coarse` | 0.647 |  | 0.00 % | 36 s | 5.6 GB |
| `ConvexAdam#composite` | 0.659 |  | 1.87 % | 37 s | 6.1 GB |
| `ConvexAdam#impact_mrct` | 0.630 |  | 0.00 % | 47 s | 23.0 GB |
| `FireANTs#syn` | 0.622 |  | 0.00 % | 96 s | 20.5 GB |
| `FireANTs#impact` | 0.496 |  | 0.00 % | 393 s | 23.1 GB |
| `FireANTs#anatomix` | 0.674 |  | 0.00 % | 114 s | 22.9 GB |
| `FireANTs#impact_mrct` | 0.684 |  | 0.00 % | 667 s | 21.2 GB |

**CBCT/CT, thorax** (ThoraxCBCT, 2 cases; before: Dice 0.35)

| Preset | Dice | TRE (mm) | Folded | Time | VRAM |
|---|---|---|---|---|---|
| `Elastix#rigid` | 0.426 |  | 0.00 % | 13 s | 5.4 GB |
| `Elastix#rigid_bspline` | 0.492 |  | 0.00 % | 20 s | 5.4 GB |
| `Elastix#impact_static` | 0.413 |  | 0.00 % | 98 s | 7.3 GB |
| `Elastix#impact_jacobian` | 0.500 |  | 0.00 % | 347 s | 5.4 GB |
| `ConvexAdam#coarse` | 0.388 |  | 0.00 % | 46 s | 8.0 GB |
| `ConvexAdam#composite` | 0.450 |  | 2.30 % | 51 s | 9.4 GB |
| `ConvexAdam#impact_cbct` | 0.455 |  | 0.60 % | 56 s | 17.3 GB |
| `FireANTs#syn` | 0.531 |  | 0.00 % | 217 s | 20.3 GB |
| **`FireANTs#impact`** | 0.532 |  | 0.00 % | 1422 s | 23.3 GB |
| `FireANTs#anatomix` | 0.504 |  | 0.00 % | 322 s | 22.8 GB |

**CBCT/CT, abdomen** (SynthRAD2025_CBCTCT, 1 case; before: Dice 0.65)

| Preset | Dice | TRE (mm) | Folded | Time | VRAM |
|---|---|---|---|---|---|
| `Elastix#rigid` | 0.654 |  | 0.00 % | 6 s | 2.5 GB |
| `Elastix#rigid_bspline` | 0.650 |  | 0.00 % | 11 s | 2.5 GB |
| `Elastix#impact_static` | 0.677 |  | 0.00 % | 89 s | 5.6 GB |
| `Elastix#impact_jacobian` | 0.668 |  | 0.00 % | 314 s | 2.5 GB |
| `ConvexAdam#coarse` | 0.637 |  | 0.00 % | 23 s | 2.5 GB |
| `ConvexAdam#composite` | 0.685 |  | 0.03 % | 25 s | 2.7 GB |
| `ConvexAdam#impact_cbct` | 0.705 |  | 0.00 % | 26 s | 6.8 GB |
| `FireANTs#syn` | 0.705 |  | 0.00 % | 50 s | 9.0 GB |
| **`FireANTs#impact`** | 0.706 |  | 0.00 % | 123 s | 22.3 GB |
| `FireANTs#anatomix` | 0.678 |  | 0.00 % | 76 s | 16.1 GB |

**CBCT/CT, head and neck** (SynthRAD2025_HN_CBCTCT, 1 case; before: Dice 0.68)

| Preset | Dice | TRE (mm) | Folded | Time | VRAM |
|---|---|---|---|---|---|
| `Elastix#rigid` | 0.685 |  | 0.00 % | 6 s | 2.4 GB |
| `Elastix#rigid_bspline` | 0.742 |  | 0.00 % | 11 s | 2.4 GB |
| `Elastix#impact_static` | 0.738 |  | 0.00 % | 67 s | 4.8 GB |
| `Elastix#impact_jacobian` | 0.749 |  | 0.00 % | 259 s | 2.2 GB |
| `ConvexAdam#coarse` | 0.684 |  | 0.00 % | 23 s | 2.4 GB |
| `ConvexAdam#composite` | 0.730 |  | 0.33 % | 24 s | 2.7 GB |
| `ConvexAdam#impact_cbct` | 0.751 |  | 0.00 % | 26 s | 6.8 GB |
| `FireANTs#syn` | 0.791 |  | 0.00 % | 29 s | 8.7 GB |
| **`FireANTs#impact`** | 0.800 |  | 0.00 % | 75 s | 22.1 GB |
| `FireANTs#anatomix` | 0.740 |  | 0.00 % | 60 s | 15.0 GB |

**CT/CT, lungs (inspiration/expiration)** (Lung250M, 1 case; before: Dice 0.06, TRE 12.8 mm)

| Preset | Dice | TRE (mm) | Folded | Time | VRAM |
|---|---|---|---|---|---|
| `Elastix#rigid` | 0.057 | 15.26 | 0.00 % | 10 s | 5.3 GB |
| `Elastix#rigid_bspline` | 0.637 | 4.36 | 0.00 % | 15 s | 5.3 GB |
| `Elastix#impact_static` | 0.659 | 3.13 | 0.00 % | 92 s | 5.3 GB |
| `Elastix#impact_jacobian` | 0.451 | 8.36 | 0.00 % | 331 s | 5.3 GB |
| `ConvexAdam#coarse` | 0.490 | 4.52 | 0.00 % | 37 s | 5.3 GB |
| `ConvexAdam#composite` | 0.801 | 2.12 | 0.02 % | 39 s | 5.7 GB |
| `FireANTs#syn` | 0.725 | 3.91 | 0.00 % | 141 s | 19.7 GB |
| `FireANTs#impact` | 0.720 | 4.26 | 0.00 % | 332 s | 22.2 GB |
| **`FireANTs#anatomix`** | 0.807 | 2.06 | 0.00 % | 186 s | 20.2 GB |

**CT/CT, abdomen (two patients)** (AbdomenCTCT, 2 cases; before: Dice 0.20)

| Preset | Dice | TRE (mm) | Folded | Time | VRAM |
|---|---|---|---|---|---|
| `Elastix#rigid` | 0.139 |  | 0.00 % | 7 s | 2.3 GB |
| `Elastix#rigid_bspline` | 0.247 |  | 0.08 % | 16 s | 2.3 GB |
| **`Elastix#impact_static`** | 0.359 |  | 0.00 % | 120 s | 11.2 GB |
| `Elastix#impact_jacobian` | 0.250 |  | 0.07 % | 305 s | 2.3 GB |
| `ConvexAdam#coarse` | 0.320 |  | 0.12 % | 59 s | 2.3 GB |
| `ConvexAdam#composite` | 0.344 |  | 1.80 % | 64 s | 2.5 GB |
| `FireANTs#syn` | 0.281 |  | 0.36 % | 32 s | 8.1 GB |
| `FireANTs#impact` | 0.255 |  | 0.12 % | 73 s | 22.7 GB |
| `FireANTs#anatomix` | 0.322 |  | 0.24 % | 41 s | 13.1 GB |

**MR/MR, brain (two subjects)** (OASIS, 2 cases; before: Dice 0.50)

| Preset | Dice | TRE (mm) | Folded | Time | VRAM |
|---|---|---|---|---|---|
| `Elastix#rigid` | 0.520 |  | 0.00 % | 6 s | 2.1 GB |
| `Elastix#rigid_bspline` | 0.675 |  | 0.00 % | 8 s | 2.1 GB |
| `Elastix#impact_static` | 0.662 |  | 0.00 % | 53 s | 2.1 GB |
| `Elastix#impact_jacobian` | 0.694 |  | 0.00 % | 232 s | 2.1 GB |
| `ConvexAdam#coarse` | 0.582 |  | 0.00 % | 20 s | 2.1 GB |
| `ConvexAdam#composite` | 0.738 |  | 0.03 % | 21 s | 2.1 GB |
| **`FireANTs#syn`** | 0.741 |  | 0.01 % | 24 s | 7.4 GB |
| `FireANTs#impact` | 0.732 |  | 0.00 % | 64 s | 18.9 GB |
| `FireANTs#anatomix` | 0.737 |  | 0.08 % | 31 s | 12.4 GB |

**MR/MR, brain tumour (follow-up)** (BraTSReg, 1 case; before: TRE 2.0 mm)

| Preset | Dice | TRE (mm) | Folded | Time | VRAM |
|---|---|---|---|---|---|
| `Elastix#rigid` |  | 1.78 | 0.00 % | 6 s | 2.6 GB |
| `Elastix#rigid_bspline` |  | 2.03 | 0.00 % | 9 s | 2.6 GB |
| `Elastix#impact_static` |  | 1.81 | 0.00 % | 53 s | 2.2 GB |
| `Elastix#impact_jacobian` |  | 1.88 | 0.00 % | 245 s | 2.6 GB |
| `ConvexAdam#coarse` |  | 1.79 | 0.00 % | 23 s | 2.4 GB |
| **`ConvexAdam#composite`** |  | 1.41 | 0.00 % | 25 s | 2.8 GB |
| `FireANTs#syn` |  | 1.52 | 0.00 % | 32 s | 9.4 GB |
| `FireANTs#impact` |  | 1.52 | 0.00 % | 79 s | 23.3 GB |
| `FireANTs#anatomix` |  | 1.44 | 0.00 % | 81 s | 16.1 GB |
