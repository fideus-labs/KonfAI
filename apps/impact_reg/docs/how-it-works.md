# How IMPACT-Reg works

## The pipeline

```text
impact-reg-konfai register PRESET -f fixed -m moving
        │
        ▼
preset (a KonfAI app on Hugging Face VBoussot/ImpactReg: app.json, Prediction.yml, engine parameters)
        │   resolved and cached by konfai-apps; its requirements pip-installed on first use
        ▼
konfai-apps infer            one subprocess per preset: reads the pair, runs the engine, writes the field
        │
        ├── elastix-IMPACT   elastix binary with the IMPACT metric (B-spline, ASGD)
        ├── FireANTs         rigid → affine → SyN on the GPU, optionally on IMPACT features
        └── ConvexAdam       itk-impact: coarse discrete search, then Adam refinement
        │
        ▼
displacement field on the fixed grid   (several presets: averaged voxel by voxel)
        │
        ▼
Moved image · eval (MAE, Dice, TRE, Jacobian) · uncertainty (spread of an ensemble) · apply (more images)
```

A pair larger than a preset registers whole is split: the preset runs once on a coarse copy of the whole pair, then
its deformable stage refines the result on native-resolution tiles (see [Large images](large-images.md)).

## The transform and its conventions

`Output/<case>/Transform.h5` is an ITK `DisplacementFieldTransform` stored as HDF5, readable by SimpleITK, ITK, 3D Slicer
and ANTs:

| Property | Value |
|---|---|
| Grid | the fixed image's: same size, origin, spacing and direction |
| Direction | maps a point of the **fixed** image to the matching point of the **moving** image (ITK's pull-back convention) |
| Moved image | `moved(x) = moving(x + D(x))` for every fixed voxel `x`: resampling the moving image through the transform |
| Vector components | physical x, y, z (LPS, as ITK), not voxel axes; the direction matrix is already applied |
| Units | the images' physical unit: millimetres for NIfTI, MHA, NRRD, DICOM; what the store declares for OME-Zarr |
| Outside the grid | zero displacement |
| Size on disk | 24 bytes per fixed voxel (float64), whatever the preset: 175 MB for 240 × 180 × 176 |

Consequences worth knowing:

- **Landmarks** go the same way as the transform: a fixed landmark `p` lands on `p + D(p)`, which should be its moving
  partner. `eval` measures TRE exactly like that.
- **Segmentations of the moving image** follow the moved image: `impact-reg-konfai apply --labels` warps them with
  nearest-neighbour interpolation. There is no inverse transform: to bring fixed-side data onto the moving image,
  register the other way round.
- **Ensembles** average the members' fields voxel by voxel on the common fixed grid (not a diffeomorphic mean).

## IMPACT: registering on what a network sees

Intensity metrics compare grey values, which works when both images show anatomy with the same contrast (CT/CT, MR/MR
of one sequence). Across modalities, the same organ is bright in one image and dark in the other. IMPACT compares
instead the **features** a pretrained network computes from each image, which describe anatomy rather than grey
values.

```text
fixed image ──► network ──► feature maps F(x) ─┐
                                                ├─► distance per voxel (L1, L2, Dice, cosine, NCC) ─► metric
moving image ─► network ──► feature maps M(x) ─┘                                   ▲
      ▲                                                                            │
      └────────────── transform, updated by the optimiser to lower the metric ◄───┘
```

1. **Image.** Each image keeps its own geometry. Before the features are computed, both images are handed over with
   their voxel axes in the same (LPS) order: a network computes its channels along the axes it is given, so a
   flipped or permuted copy of the same anatomy would otherwise give different channels.
2. **Network.** A TorchScript file from [`VBoussot/impact-torchscript-models`](https://huggingface.co/VBoussot/impact-torchscript-models)
   turns an image into a list of feature maps, one per layer, shallow to deep. It normalises its own input.
3. **Feature maps.** `layers_mask` picks layers by position: `'01'` asks for the first two layers and keeps the
   second, `'1'` keeps the only one. Shallow layers keep texture and edges (half resolution, a receptive field of a
   few voxels); deep layers carry organ-level meaning.
4. **Similarity.** The fixed and moving features are compared voxel by voxel over the channels, optionally after PCA,
   several models or layers summed with their weights.
5. **Optimisation.** The engine moves the transform to lower that distance, often together with an intensity term
   (mutual information) and a regulariser.

**Static** extracts each image's features once and registers the feature volumes: fast, and the features never see
the moving image deformed. **Jacobian** recomputes the moving features while optimising and differentiates the loss
through the network: slower, and the features of the warped image stay consistent with its appearance.

### The feature models the presets use

| Model | File | Size | What it is | Layers used |
|---|---|---|---|---|
| MIND-SSC | `MIND/R1D2_3D.pt` | 15 kB | handcrafted local self-similarity descriptor, radius 1, dilation 2 | 1 layer, 12 channels, full resolution |
| TotalSegmentator MR | `TS/M730.pt` | 123 MB | nnU-Net trained on MRI (`total_mr`, organs) | layer 1: 64 channels at 1/2 resolution; later layers: organ-level features |
| TotalSegmentator CT | `TS/M291.pt` | 125 MB | nnU-Net trained on CT (`total`, organs) | layer 1: 64 channels at 1/2 resolution |
| anatomix | `Anatomix/Anatomix.pt` | 24 MB | contrast-agnostic 3D U-Net trained on synthetic anatomy | 1 layer, 16 channels, full resolution |

They are downloaded once to the Hugging Face cache (`~/.cache/huggingface`) when a preset first needs them.

### Which features for which pair

| Pair | Features that work | Why |
|---|---|---|
| CT / CBCT | early layers (TotalSegmentator layer 1, Jacobian) | the same tissue contrast, degraded: texture and edges carry the alignment |
| MR / CT | deep layers of a segmentation network, plus MIND | contrasts differ; organ-level features do not |
| any pair, no GPU budget for a network | MIND | cheap, contrast-invariant by construction |

## The three engines

| | elastix-IMPACT | FireANTs | ConvexAdam (itk-impact) |
|---|---|---|---|
| Transform | rigid, then B-spline | rigid, affine, then SyN (diffeomorphic) | affine, then a dense field |
| Optimiser | adaptive stochastic gradient descent on random points | Adam, multi-resolution | coarse discrete search, then Adam |
| Similarity | mutual information or IMPACT | local correlation, mutual information or IMPACT | IMPACT |
| IMPACT modes | Static or Jacobian | Static or Jacobian | Static or Jacobian (fine stage) |
| Runs on | CPU, IMPACT metric on a CUDA GPU | GPU (CPU works, slowly) | GPU or CPU |
| Presets | `Generic_Rigid`, `Generic_Rigid_BSpline`, `Elastix_IMPACT_Static`, `Elastix_IMPACT_Jacobian` | `FireANTs_SyN`, `FireANTs_IMPACT`, `FireANTs_Anatomix`, `FireANTs_IMPACT_MRCT` | `ConvexAdam_Coarse`, `ConvexAdam_Composite`, `ConvexAdam_IMPACT_CBCT`, `ConvexAdam_IMPACT_MRCT` |

The three engines compute one IMPACT loss: the same models, layers, PCA, feature normalisation and distances, each
layer divided by its value when a level starts (`normalize`). elastix and itk-impact share itk-impact's C++
implementation, and FireANTs uses KonfAI's IMPACT measure (`konfai.metric.measure.impact`), tested against it. What
still differs is how each engine samples the image: elastix draws random points, ConvexAdam and FireANTs score the whole grid unless
`voxel_sampling` asks them for random points too.
