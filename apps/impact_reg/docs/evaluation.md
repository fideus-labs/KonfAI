# Evaluating a registration

## `eval`

`eval` scores a transform on any subset of three modalities, each given as a fixed side and a moving side. The moving
side is always the **original** moving data: `eval` warps it through `--transform` itself (without `--transform`, it
measures the misalignment before registration).

| Modality | Options | Metric | Unit |
|---|---|---|---|
| image | `-f` / `-m`, optional `--mask` (on the fixed grid) | MAE between the fixed image and the warped moving one, inside the mask | the images' intensity unit |
| segmentation | `--gt-fixed-seg` / `--gt-moving-seg` | Dice of every label of the fixed segmentation, and their mean | 0 to 1 |
| landmarks | `--gt-fixed-fid` / `--gt-moving-fid` | TRE: each fixed landmark pushed through the transform, against its moving partner | mm |

When the transform is a displacement field, `eval` also reports its **Jacobian determinant**: the fraction of folded
voxels (determinant ≤ 0), the minimum, and the standard deviation of its logarithm. A good deformable registration has
no folding.

```bash
impact-reg-konfai eval --transform Output/P000/Transform.h5 \
    -f fixed.nii.gz -m moving.nii.gz \
    --gt-fixed-seg fixed_organs.nii.gz --gt-moving-seg moving_organs.nii.gz \
    --gt-fixed-fid fixed.fcsv --gt-moving-fid moving.fcsv \
    -o Evaluation
```

**Outputs.** One folder per case and modality, `Evaluation/<case>/Evaluation/{Image,Segmentation,Landmarks,Jacobian}/`,
each holding the metrics as `ImpactReg/Metric_TRAIN.json` (and the voxel-wise map, `ImpactReg/Output/<case>/MAE_map.mha`
or `Dice_map.mha`, where one applies), and
`Evaluation/Evaluation_summary.json` gathering every case and the cohort's statistics (mean, median, min, max,
quartiles).

**Several cases.** Each option takes one entry per case (paired by position) or a single entry used for every case,
such as one atlas or one transform. Other counts are refused.

**Landmark files** are 3D Slicer markups `.fcsv`. The `# CoordinateSystem` header is honoured (LPS, or RAS, which is
converted); rows are paired by order, so both files must list the same points in the same order.

**Transforms** accepted: the `Transform.h5` of `register`, a displacement-field image, or an ITK linear transform
(`.h5`, `.tfm`, `.txt`: translation, rigid, similarity, affine).

## `uncertainty`

Several presets registering one pair give several fields. Where they agree, the registration is reliable; where they
disagree, it is not. `uncertainty` measures that disagreement voxel by voxel: the root mean square distance of the
members' displacement vectors to their mean (sample, N−1), in mm.

```bash
impact-reg-konfai register ConvexAdam_Composite FireANTs_Anatomix Elastix_IMPACT_Static \
    -f fixed.nii.gz -m moving.nii.gz -o Output --keep-fields --gpu 0
impact-reg-konfai uncertainty --dvf Output/P000/Ensemble/*.h5 -o Output/P000
```

The map is written as `Output/P000/uncertainty/Uncertainty.mha`, on the fixed grid. Ensemble members that register
different things (a rigid preset beside deformable ones) make the spread mean little: ensemble presets of the same
nature.

## `apply`

`register` writes the moved image of the moving image only. `apply` warps anything else from the moving side through
the same transform: its segmentation (`--labels`, nearest neighbour, no invented label values), another sequence, a
mask.

```bash
impact-reg-konfai apply --transform Output/P000/Transform.h5 -f fixed.nii.gz \
    -i moving_organs.nii.gz --labels -o Output/P000
```

Each image is written as `<output>/<its name>`, in its own format. The transform goes from the fixed side to the
moving side, so it only brings moving-side data onto the fixed grid.
