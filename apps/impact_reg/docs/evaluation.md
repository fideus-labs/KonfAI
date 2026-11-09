# Evaluating a registration

## `eval`

`eval` scores a transform on any subset of three modalities, each given as a fixed side and a moving side. The moving
side is always the **original** moving data: `eval` warps it through `--transform` itself (without `--transform`, it
measures the misalignment before registration).

| Modality | Options | Metric | Unit |
|---|---|---|---|
| image | `-f` / `-m`, optional `--mask` (on the fixed grid) | the fixed image against the warped moving one, inside the mask: MAE (one modality, one scale), NMI (any modalities), NCC and LNCC (one modality); and the share of the fixed grid sent outside the moving image | intensity unit; 1 to 2; -1 to 1; 0 to 1; 0 to 1 |
| segmentation | `--gt-fixed-seg` / `--gt-moving-seg` | every label of the fixed segmentation, and their mean: Dice, Jaccard, relative volume difference; HD95, ASSD; surface Dice at 2 mm | Dice, Jaccard 0 to 1, relative volume difference 0 and up; mm; 0 to 1 |
| landmarks | `--gt-fixed-fid` / `--gt-moving-fid` | TRE: each fixed landmark pushed through the transform, against its moving partner | mm |

When the transform is a displacement field or a B-spline, `eval` also reports its **measures without reference**: from the
determinant of its Jacobian, the fraction of folded voxels (determinant ≤ 0), the minimum and the standard deviation of
its logarithm; its bending energy (the mean squared second derivative, 0 for an affine map); and the length of its
displacements (mean, 95th percentile, maximum). A good deformable registration has no folding. `--transform` given
alone measures these and nothing else. A field is measured on its own grid; a B-spline (elastix's native form) has
none and is sampled on the fixed grid, which `-f` or `--gt-fixed-seg` gives.

The cohort summary adds `worst30` to every metric: the mean of its worst 30 % of cases, the robustness Learn2Reg
reports as DSC30 and TRE30.

```bash
impact-reg-konfai eval --transform Output/P000/Transform.h5 \
    -f fixed.nii.gz -m moving.nii.gz \
    --gt-fixed-seg fixed_organs.nii.gz --gt-moving-seg moving_organs.nii.gz \
    --gt-fixed-fid fixed.fcsv --gt-moving-fid moving.fcsv \
    -o Evaluation
```

**Without impact-reg.** Each modality is `konfai-apps eval` of the transform (`-i`) against the fixed and the moving
data (`--gt`, in that order), through the configs every engine app ships: `Evaluation_with_images.yml`,
`Evaluation_with_seg.yml`, `Evaluation_with_fid.yml`, and `Evaluation_no_reference.yml`, which takes the field alone.
`Evaluation_inverse_consistency.yml` takes a registration and the one registered back (`-i FORWARD --gt BACKWARD --gt
FIXED`): the distance each fixed voxel ends from where it started, in mm. `konfai-apps pipeline` registers and scores
in one call.

```bash
konfai-apps eval VBoussot/ImpactReg:ConvexAdam --evaluation-file Evaluation_with_seg.yml \
    -i Output/P000/Transform.h5 --gt fixed_organs.nii.gz --gt moving_organs.nii.gz -o Evaluation
konfai-apps pipeline VBoussot/ImpactReg:ConvexAdam -i fixed.nii.gz -i moving.nii.gz \
    --gt fixed_organs.nii.gz --gt moving_organs.nii.gz --evaluation-file Evaluation_with_seg.yml -o Output
```

**Outputs.** One folder per case and modality, `Evaluation/<case>/Evaluation/{Image,Segmentation,Landmarks,NoReference}/`,
each holding the metrics as `ImpactReg/Metric_TRAIN.json` (and the voxel-wise map, `ImpactReg/Output/<case>/MAE_map.mha`
or `Dice_map.mha`, where one applies), and
`Evaluation/Evaluation_summary.json` gathering every case and the cohort's statistics (mean, median, min, max,
quartiles, worst 30 %).

**Several cases.** Each option takes one entry per case (paired by position) or a single entry used for every case,
such as one atlas or one transform. Other counts are refused.

**Landmark files** are 3D Slicer markups `.fcsv`. The `# CoordinateSystem` header is honoured (LPS, or RAS, which is
converted); rows are paired by order, so both files must list the same points in the same order.

**Transforms** accepted: the `Transform.h5` of `register`, a displacement-field image, an ITK linear transform
(`.h5`, `.tfm`, `.txt`: translation, rigid, similarity, affine), or an ITK B-spline or composite of these (elastix's
rigid then B-spline).

## `uncertainty`

Several presets registering one pair give several fields. Where they agree, the registration is reliable; where they
disagree, it is not. `register --uncertainty` keeps every preset's field beside their mean, as the ensemble's stack
(`InferenceStack.mha`, or `.ome.zarr` beside a store: one volume whose channels are the members' components, the
member count in its header). `uncertainty` measures the disagreement from it, through konfai-apps' `uncertainty` and
the configs shipped in `evaluation/`:

| Given | Measured |
|---|---|
| the stack alone | the spread of the displacement vectors, voxel by voxel: the root mean square distance of the members' vectors to their mean (sample, N−1), in mm; its mean and 95th percentile |
| `--gt-fixed-fid` | the spread of the fixed landmarks pushed through every member, mm per landmark |
| `--gt-fixed-fid --gt-moving-fid` | whether that spread predicts the error: its Spearman correlation with the TRE |
| `--gt-moving-seg` | how the moving segmentation, pulled through every member onto the fixed grid, disagrees |

```bash
impact-reg-konfai register ConvexAdam#composite FireANTs#anatomix \
    -f fixed.nii.gz -m moving.nii.gz -o Output --uncertainty --gpu 0
impact-reg-konfai uncertainty -i Output/P000/InferenceStack.mha \
    --gt-fixed-fid fixed.fcsv --gt-moving-fid moving.fcsv -o Uncertainty
```

Case `P000` is written to `Uncertainty/P000/Uncertainty/<Map|Landmarks|Segmentation>/`, on the fixed grid, and every
measure to `Uncertainty_summary.json`. Ensemble members that register different things (a rigid preset beside
deformable ones) make the spread mean little: ensemble presets of the same nature.

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
