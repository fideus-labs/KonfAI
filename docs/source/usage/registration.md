# Register two images with IMPACT-Reg

IMPACT-Reg aligns a **moving** image to a **fixed** image using a published registration
preset. The fixed image defines the output grid. This guide runs an existing preset;
{doc}`../examples/registration` shows how to train a registration model.

## Install and choose a preset

```bash
python -m pip install impact-reg-konfai
impact-reg-konfai list
impact-reg-konfai show Generic_Rigid_BSpline
```

`list` gives the available presets. `show` describes one preset's engine, requirements
and adjustable parameters. The first run downloads its models and engine dependencies.
The installation requirements differ between elastix, ConvexAdam and FireANTs; consult
the [IMPACT-Reg installation guide](https://github.com/fideus-labs/KonfAI/blob/c621fa29577e44a26f259a7a2a747a7a9854d39d/apps/impact_reg/docs/troubleshooting.md)
before changing engines.

## Register the pair

For a moving image and its fixed reference:

```bash
impact-reg-konfai register Generic_Rigid_BSpline \
  -f ./fixed.mha -m ./moving.mha \
  -o ./Registration --cpu 1
```

Replace the paths with your pair. This preset runs on CPU. For a feature-based
MR/CT registration, inspect `Elastix_IMPACT_Static` with `show` and use `--gpu 0`.
When the CT includes a table or anatomy outside the MR field of view, supply its body
mask with `--fixed-mask ./ct_body.nii.gz` to restrict the metric region.

For a single pair, inspect:

- `Registration/P000/Moved.<ext>`: the moving image resampled onto the fixed grid;
- `Registration/P000/Transform.h5`: the transform used to produce it;
- `register.json` under the output directory: the record of the run.

Overlay the moved and fixed images and check anatomical alignment. Use the
[evaluation guide](https://github.com/fideus-labs/KonfAI/blob/c621fa29577e44a26f259a7a2a747a7a9854d39d/apps/impact_reg/docs/evaluation.md)
when you have reference labels or landmarks.

## Apply the transform to labels

Use the original moving-side labels. `--labels` selects nearest-neighbour interpolation:

```bash
impact-reg-konfai apply \
  --transform ./Registration/P000/Transform.h5 \
  -f ./fixed.mha -i ./moving_labels.mha \
  --labels -o ./WarpedLabels
```

## Large images and output resolution

A large pair can be optimized on a coarser grid to fit the declared memory budget.
Some presets then refine the result in native-resolution tiles. The output transform
is placed on the fixed grid in either case: its grid alone does not tell you the
resolution used for optimization. See the large-image guide below for each preset's
route.

Loading the moved image and ensemble fields into Slicer requires memory for those
volumes, even when the registration itself used tiles.

## Choose the next step

- [Presets](https://github.com/fideus-labs/KonfAI/blob/c621fa29577e44a26f259a7a2a747a7a9854d39d/apps/impact_reg/docs/presets.md):
  choose an engine and feature model for your image pair.
- [Large images](https://github.com/fideus-labs/KonfAI/blob/c621fa29577e44a26f259a7a2a747a7a9854d39d/apps/impact_reg/docs/large-images.md):
  understand the coarse pass, native-resolution tiles and memory requirements.
- [Parameters](https://github.com/fideus-labs/KonfAI/blob/c621fa29577e44a26f259a7a2a747a7a9854d39d/apps/impact_reg/docs/parameters.md):
  change a preset with `--set`.
- [IMPACT-Reg notebook](https://github.com/fideus-labs/KonfAI/blob/main/examples/ImpactReg/register_demo.ipynb):
  download a demo pair and inspect its registration.

Several presets can be listed in one `register` command. Their displacement fields are
averaged; pass `--keep-fields` to retain each field for a later uncertainty calculation.
Field disagreement measures variation between those registrations, not registration accuracy.
